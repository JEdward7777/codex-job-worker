"""
W2V2-BERT ASR Training Handler

Handles ASR training jobs for the W2V2-BERT model (and traditional Wav2Vec2).
Downloads training data from GitLab, trains the model, and uploads the
checkpoint back to GitLab.

Supports both:
- facebook/w2v-bert-2.0 (more accurate, requires more VRAM)
- facebook/wav2vec2-base (less accurate, requires less VRAM)
"""

import os
import sys
import csv
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from handlers.base import create_tar_archive
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Import training API
from train_w2v2bert_asr import train_w2v2bert_asr_api


def run(job_context: Dict[str, Any], callbacks) -> Dict[str, Any]:
    """
    Execute ASR training job.

    Args:
        job_context: Job configuration from GitLab manifest
        callbacks: JobCallbacks object for heartbeat, file operations, etc.

    Returns:
        Dictionary with:
            - success: bool
            - error_message: str (if success is False)
    """
    work_dir = callbacks.get_work_dir()

    print("=" * 60)
    print("W2V2-BERT ASR Training Handler")
    print("=" * 60)

    try:
        # Extract job configuration
        # The manifest places most fields at the top level of the job dict.
        model_config = job_context.get('model', {})
        training_config = job_context.get('training', {})

        # Get training parameters with defaults
        # 'epochs' is a top-level manifest field per the spec.
        num_epochs = job_context.get('epochs', 5)
        batch_size = job_context.get('batch_size', 8)
        learning_rate = job_context.get('learning_rate', 3e-4)
        gradient_accumulation_steps = job_context.get('gradient_accumulation_steps', 2)
        warmup_steps = job_context.get('warmup_steps', 500)
        save_steps = job_context.get('save_steps', 500)
        eval_steps = job_context.get('eval_steps', 500)
        val_split = job_context.get('val_split', 0.1)
        test_split = job_context.get('test_split', 0.1)
        max_duration_seconds = job_context.get('max_duration_seconds')
        use_8bit_optimizer = job_context.get('use_8bit_optimizer', False)

        # Get model parameters
        base_model = model_config.get('base_model', 'facebook/w2v-bert-2.0')
        base_checkpoint = model_config.get('base_checkpoint')

        # Determine if using traditional Wav2Vec2 or W2V2-BERT
        # User can explicitly set this, or it's auto-detected from model name
        use_wav2vec2_base = model_config.get('use_wav2vec2_base')
        if use_wav2vec2_base is None:
            use_wav2vec2_base = 'bert' not in base_model.lower()

        # Text normalization options (handler-specific, not in manifest spec)
        text_config = job_context.get('text_normalization', {})
        lowercase = text_config.get('lowercase', True)
        remove_punctuation = text_config.get('remove_punctuation', True)
        remove_numbers = text_config.get('remove_numbers', False)

        # Get filter configuration (per spec, training filters live under 'training')
        include_verses = training_config.get('include_verses', [])
        exclude_verses = training_config.get('exclude_verses', [])

        model_arch = "Wav2Vec2" if use_wav2vec2_base else "Wav2Vec2-BERT"
        print("\nTraining Configuration:")
        print(f"  Model architecture: {model_arch}")
        print(f"  Base model: {base_model}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Validation split: {val_split}")
        print(f"  Test split: {test_split}")
        print(f"  8-bit optimizer: {use_8bit_optimizer}")
        print(f"  Base checkpoint: {base_checkpoint or 'None (training from scratch)'}")
        if max_duration_seconds:
            print(f"  Max audio duration: {max_duration_seconds}s")
        if include_verses:
            print(f"  Include verses: {len(include_verses)} specified")
        if exclude_verses:
            print(f"  Exclude verses: {len(exclude_verses)} specified")
        print()

        # Step 1: Download training data from GitLab
        print("Step 1: Downloading training data from GitLab...")
        callbacks.heartbeat(message="Downloading training data", stage="download")

        data_dir = work_dir / "data"
        audio_dir = data_dir / "audio"
        data_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Download .codex files and audio
        download_result = _download_training_data(
            job_context=job_context,
            callbacks=callbacks,
            output_dir=data_dir,
            audio_dir=audio_dir,
            include_verses=include_verses,
            exclude_verses=exclude_verses
        )

        if not download_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download training data: {download_result['error_message']}"
            }

        metadata_csv = download_result['metadata_csv']
        sample_count = download_result['sample_count']

        print(f"  Downloaded {sample_count} training samples")

        if sample_count == 0:
            return {
                'success': False,
                'error_message': "No training samples found. Check that cells have both audio and text."
            }

        # Step 2: Download base checkpoint if specified
        resume_checkpoint = None
        if base_checkpoint:
            print("\nStep 2: Downloading base checkpoint...")
            callbacks.heartbeat(message="Downloading base checkpoint", stage="download")

            checkpoint_result = _download_checkpoint(
                job_context=job_context,
                callbacks=callbacks,
                checkpoint_path=base_checkpoint,
                output_dir=work_dir / "base_checkpoint"
            )

            if checkpoint_result['success']:
                resume_checkpoint = checkpoint_result['local_path']
                print(f"  Downloaded checkpoint to: {resume_checkpoint}")
            else:
                print(f"  Warning: Failed to download checkpoint: {checkpoint_result['error_message']}")
                print("  Training will start from scratch.")
        else:
            print("\nStep 2: No base checkpoint specified, training from scratch")

        # Step 3: Train the model
        print("\nStep 3: Training W2V2-BERT ASR model...")
        callbacks.heartbeat(epochs_completed=0, message="Starting training", stage="training")

        output_dir = work_dir / "model_output"

        def training_heartbeat(epoch: int):
            """Heartbeat callback for training progress."""
            callbacks.heartbeat(epochs_completed=epoch, message=f"Training epoch {epoch}", stage="training")

        train_result = train_w2v2bert_asr_api(
            csv_path=str(metadata_csv),
            output_dir=str(output_dir),
            model_name=base_model,
            audio_column='file_name',
            text_column='transcription',
            audio_base_path=str(audio_dir),
            use_wav2vec2_base=use_wav2vec2_base,
            train_split=1.0 - val_split - test_split,
            val_split=val_split,
            test_split=test_split,
            max_duration_seconds=max_duration_seconds,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            remove_numbers=remove_numbers,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            use_8bit_optimizer=use_8bit_optimizer,
            resume_from_checkpoint=resume_checkpoint,
            heartbeat_callback=training_heartbeat
        )

        if not train_result['success']:
            return {
                'success': False,
                'error_message': f"Training failed: {train_result['error_message']}"
            }

        print(f"  Training completed: {train_result['epochs_completed']} epochs")
        print(f"  Final model: {train_result['final_model_path']}")
        if train_result.get('test_results'):
            print(f"  Test WER: {train_result['test_results'].get('eval_wer', 'N/A')}")
            print(f"  Test CER: {train_result['test_results'].get('eval_cer', 'N/A')}")

        # Step 4: Upload model to GitLab
        print("\nStep 4: Uploading model to GitLab...")
        callbacks.heartbeat(message="Uploading model", stage="upload")

        upload_result = _upload_model(
            job_context=job_context,
            callbacks=callbacks,
            model_path=train_result['final_model_path'],
            job_id=job_context['job_id']
        )

        if not upload_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to upload model: {upload_result['error_message']}"
            }

        print(f"  Uploaded model to: {upload_result['remote_path']}")

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'epochs_completed': train_result['epochs_completed'],
            'model_path': upload_result['remote_path'],
            'test_results': train_result.get('test_results'),
            'train_metrics': train_result.get('train_metrics')
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\nTraining failed with error: {error_msg}")
        return {
            'success': False,
            'error_message': error_msg
        }


def _download_training_data(
    job_context: Dict[str, Any],
    callbacks,
    output_dir: Path,
    audio_dir: Path,
    include_verses: list,
    exclude_verses: list
) -> Dict[str, Any]:
    """
    Download training data from GitLab using GitLabDatasetDownloader.

    Returns:
        Dictionary with success, metadata_csv, sample_count, error_message
    """
    try:
        scanner = callbacks.scanner

        # Create a GitLabDatasetDownloader instance for data operations
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=scanner.server_url,
            access_token=scanner.access_token,
            project_id=str(job_context['project_id']),
            config_overrides={
                'dataset.output_dir': str(output_dir),
                'dataset.audio_dir': 'audio',
                'dataset.csv_filename': 'metadata.csv',
                'dataset.text_source': 'value',
            }
        )

        # List all files in the repository to build audio map
        print("  Listing repository files...")
        items = downloader.list_repository_tree()

        # Build audio files map using the existing method
        print("  Building audio files map...")
        audio_files_map = downloader.build_audio_files_map(items)

        # Get list of .codex files from job config or find them in repository
        job_config = job_context.get('job_config', {})
        codex_files = job_config.get('codex_files', [])

        if not codex_files:
            codex_files = [item['path'] for item in items if item['name'].endswith('.codex')]
            if not codex_files:
                return {
                    'success': False,
                    'metadata_csv': None,
                    'sample_count': 0,
                    'error_message': "No .codex files found in repository"
                }
            print(f"  Found {len(codex_files)} .codex files in repository")

        # Create metadata CSV
        metadata_csv = output_dir / "metadata.csv"
        samples = []

        # Convert include/exclude lists to sets for faster lookup
        include_set = set(include_verses) if include_verses else None
        exclude_set = set(exclude_verses) if exclude_verses else None

        for codex_path in codex_files:
            print(f"  Processing: {codex_path}")

            # Download and parse .codex file using downloader
            codex_data = downloader.download_json_file(codex_path)
            if not codex_data:
                print(f"    Warning: Could not download {codex_path}")
                continue

            # Use the downloader's extract_audio_transcriptions method
            # which properly handles audio_info including isDeleted/isMissing checks
            pairs = downloader.extract_audio_transcriptions(codex_data, audio_files_map)

            # Filter pairs based on include/exclude verses if needed
            if include_set or exclude_set:
                filtered_pairs = []
                for pair in pairs:
                    verse_id = pair.get('verse_id', '')
                    if include_set and verse_id not in include_set:
                        continue
                    if exclude_set and verse_id in exclude_set:
                        continue
                    filtered_pairs.append(pair)
                pairs = filtered_pairs

            print(f"    Found {len(pairs)} audio-transcription pairs")

            # Download audio files and add to samples
            for pair_i, pair in enumerate(pairs):
                if pair_i % 100 == 0 or pair_i == len(pairs)-1:
                    print( f"Downloading pair {pair_i} of {len(pairs)}")

                audio_url = pair['audio_url']
                transcription = pair['transcription']
                verse_id = pair['verse_id']

                # Skip pairs without transcription text
                if not transcription or not transcription.strip():
                    continue

                # Get filename from URL
                audio_filename = Path(audio_url).name
                local_audio_path = audio_dir / audio_filename

                # Download if not already present
                if not local_audio_path.exists():
                    if downloader.download_file(audio_url, local_audio_path):
                        pass  # Success
                    else:
                        print(f"    Warning: Could not download audio for {verse_id}")
                        continue

                samples.append({
                    'file_name': audio_filename,  # Just filename, audio_base_path handles the rest
                    'transcription': transcription,
                    'verse_id': verse_id
                })

            callbacks.heartbeat(message=f"Downloaded {len(samples)} samples", stage="download")

        # Write metadata CSV
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'transcription', 'verse_id'])
            writer.writeheader()
            writer.writerows(samples)

        return {
            'success': True,
            'metadata_csv': metadata_csv,
            'sample_count': len(samples),
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'metadata_csv': None,
            'sample_count': 0,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def _download_checkpoint(
    job_context: Dict[str, Any],
    callbacks,
    checkpoint_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download a checkpoint from GitLab.

    The checkpoint is expected to be a .pth.tar archive (created by
    create_tar_archive during training upload). Uses GitLabDatasetDownloader
    which handles Git LFS transparently, then extracts the archive.

    Returns:
        Dictionary with success, local_path, error_message
    """
    try:
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create downloader for file operations (handles LFS transparently)
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=scanner.server_url,
            access_token=scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        checkpoint_filename = os.path.basename(checkpoint_path)
        local_path = output_dir / checkpoint_filename

        print(f"    Downloading checkpoint: {checkpoint_path}")
        if not downloader.download_file(checkpoint_path, local_path):
            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download checkpoint from {checkpoint_path}"
            }

        # Check if it's a tar archive and extract it
        if checkpoint_path.endswith(('.pth.tar', '.tar.gz', '.tar')):
            from handlers.base import extract_tar_archive
            print("    Extracting checkpoint archive...")
            extract_tar_archive(str(local_path), output_dir)
            # Clean up the archive after extraction
            local_path.unlink(missing_ok=True)
            return {
                'success': True,
                'local_path': str(output_dir),
                'error_message': None
            }

        return {
            'success': True,
            'local_path': str(local_path),
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'local_path': None,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def _upload_model(
    job_context: Dict[str, Any],
    callbacks,
    model_path: str,
    job_id: str
) -> Dict[str, Any]:
    """
    Upload a trained model to GitLab using LFS as a .pth.tar archive.

    For HuggingFace models, this creates a tar archive of the model directory
    and uploads it as a single LFS file.

    Returns:
        Dictionary with success, remote_path, error_message
    """
    try:
        model_dir = Path(model_path)

        # Create tar archive of the model directory
        archive_filename = "asr_model.pth.tar"
        archive_path = model_dir.parent / archive_filename

        print(f"    Creating model archive: {archive_filename}")
        create_tar_archive(model_dir, archive_path)

        # Remote path for the archive
        remote_path = f"gpu_jobs/job_{job_id}/model/{archive_filename}"

        # Create uploader instance
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        # Upload archive using LFS
        result = uploader.upload_batch(
            files=[{
                'local_path': str(archive_path),
                'remote_path': remote_path,
                'lfs': True,  # Model archives always use LFS
            }],
            commit_message=f"Upload ASR model for job {job_id}"
        )

        # Clean up local archive
        if archive_path.exists():
            archive_path.unlink()

        if result['success']:
            return {
                'success': True,
                'remote_path': remote_path,
                'error_message': None
            }
        else:
            return {
                'success': False,
                'remote_path': None,
                'error_message': result.get('error_message', 'Upload failed')
            }

    except Exception as e:
        return {
            'success': False,
            'remote_path': None,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }
