"""
StableTTS Training Handler

Handles TTS training jobs for the StableTTS model.
Downloads training data from GitLab, preprocesses it, trains the model,
and uploads the checkpoint back to GitLab.
"""

import os
import sys
import csv
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from gitlab_to_hf_dataset import GitLabDatasetDownloader
from handlers.base import (
    download_pretrained_model,
    STABLETTS_DEFAULT_REPO_ID,
    STABLETTS_DEFAULT_FILENAME,
)

# Import training and preprocessing APIs
from preprocess_stabletts import preprocess_stabletts_api
from train_stabletts import train_stabletts_api


def run(job_context: Dict[str, Any], callbacks) -> Dict[str, Any]:
    """
    Execute TTS training job.

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
    print("StableTTS Training Handler")
    print("=" * 60)

    try:
        # Extract job configuration
        job_config = job_context.get('job_config', {})
        model_config = job_context.get('model', {})
        training_config = job_config.get('training', {})

        # Get training parameters with defaults
        num_epochs = training_config.get('epochs', 100)
        batch_size = training_config.get('batch_size', 32)
        learning_rate = training_config.get('learning_rate', 1e-4)
        val_split = training_config.get('val_split', 0.1)
        save_interval = training_config.get('save_interval', 10)

        # Get model parameters
        model_type = model_config.get('type', 'StableTTS')
        base_checkpoint = model_config.get('base_checkpoint')
        language = model_config.get('language', 'english')
        use_uroman = model_config.get('use_uroman', False)
        uroman_lang = model_config.get('uroman_lang', None)

        # Get filter configuration
        include_verses = job_config.get('include_verses', [])
        exclude_verses = job_config.get('exclude_verses', [])

        print("\nTraining Configuration:")
        print(f"  Model type: {model_type}")
        print(f"  Language: {language}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Validation split: {val_split}")
        print(f"  Base checkpoint: {base_checkpoint or 'None (will use default pretrained model)'}")
        print(f"  Use uroman: {use_uroman}")
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

        # Step 2: Preprocess data for StableTTS
        print("\nStep 2: Preprocessing data for StableTTS...")
        callbacks.heartbeat(message="Preprocessing data", stage="preprocess")

        feature_dir = work_dir / "features"
        output_json = feature_dir / "filelist.json"

        preprocess_result = preprocess_stabletts_api(
            input_csv=str(metadata_csv),
            audio_base_dir=str(audio_dir),
            output_json=str(output_json),
            output_feature_dir=str(feature_dir),
            language=language,
            use_uroman=use_uroman,
            uroman_language=uroman_lang,
            resample=False,
            num_workers=2,
            heartbeat_callback=lambda: callbacks.heartbeat(message="Preprocessing data", stage="preprocess")
        )

        if not preprocess_result['success']:
            return {
                'success': False,
                'error_message': f"Preprocessing failed: {preprocess_result['error_message']}"
            }

        print(f"  Preprocessed {preprocess_result['processed_count']} / {preprocess_result['total_count']} samples")

        # Step 3: Download base checkpoint if specified, otherwise fetch default pretrained model
        pretrained_model = None
        if base_checkpoint:
            print("\nStep 3: Downloading base checkpoint from GitLab...")
            callbacks.heartbeat(message="Downloading base checkpoint", stage="download")

            checkpoint_result = _download_checkpoint(
                job_context=job_context,
                callbacks=callbacks,
                checkpoint_path=base_checkpoint,
                output_dir=work_dir / "base_checkpoint"
            )

            if checkpoint_result['success']:
                pretrained_model = checkpoint_result['local_path']
                print(f"  Downloaded checkpoint to: {pretrained_model}")
            else:
                print(f"  Warning: Failed to download checkpoint: {checkpoint_result['error_message']}")
                print("  Will fall back to default pretrained model from HuggingFace.")
                base_checkpoint = None  # Fall through to default download below

        if not base_checkpoint:
            # No base checkpoint specified (or GitLab download failed) — download default
            # pretrained model from HuggingFace Hub. The model is cached locally so
            # subsequent jobs on the same worker won't re-download it.
            print("\nStep 3: Downloading default pretrained model from HuggingFace Hub...")
            callbacks.heartbeat(message="Downloading pretrained model", stage="download")

            # Allow YAML manifest to override the default HuggingFace coordinates:
            #   model.pretrained_repo_id  (default: "KdaiP/StableTTS1.1")
            #   model.pretrained_filename (default: "StableTTS/checkpoint_0.pt")
            pretrained_repo_id = model_config.get('pretrained_repo_id', STABLETTS_DEFAULT_REPO_ID)
            pretrained_filename = model_config.get('pretrained_filename', STABLETTS_DEFAULT_FILENAME)

            try:
                pretrained_model = download_pretrained_model(
                    repo_id=pretrained_repo_id,
                    filename=pretrained_filename,
                )
            except RuntimeError as e:
                return {
                    'success': False,
                    'error_message': f"Failed to download pretrained model: {e}"
                }

        # Step 4: Train the model
        print("\nStep 4: Training StableTTS model...")
        callbacks.heartbeat(epochs_completed=0, message="Starting training", stage="training")

        checkpoint_dir = work_dir / "checkpoints"
        log_dir = work_dir / "logs"

        def training_heartbeat(epoch: int):
            """Heartbeat callback for training progress."""
            callbacks.heartbeat(epochs_completed=epoch, message=f"Training epoch {epoch}", stage="training")

        train_result = train_stabletts_api(
            train_dataset_path=str(output_json),
            model_save_path=str(checkpoint_dir),
            log_dir=str(log_dir),
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_interval=16,
            save_interval=save_interval,
            checkpoint=None,
            pretrained_model=pretrained_model,
            val_split=val_split,
            best_metric='total_loss',
            heartbeat_callback=training_heartbeat
        )

        if not train_result['success']:
            return {
                'success': False,
                'error_message': f"Training failed: {train_result['error_message']}"
            }

        print(f"  Training completed: {train_result['epochs_completed']} epochs")
        print(f"  Best model: {train_result['best_model_path']}")

        # Step 5: Upload checkpoint to GitLab
        print("\nStep 5: Uploading checkpoint to GitLab...")
        callbacks.heartbeat(message="Uploading checkpoint", stage="upload")

        upload_result = _upload_checkpoint(
            job_context=job_context,
            callbacks=callbacks,
            checkpoint_path=train_result['best_model_path'],
            job_id=job_context['job_id']
        )

        if not upload_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to upload checkpoint: {upload_result['error_message']}"
            }

        print(f"  Uploaded checkpoint to: {upload_result['remote_path']}")

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'epochs_completed': train_result['epochs_completed'],
            'checkpoint_path': upload_result['remote_path']
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

    This leverages the existing GitLabDatasetDownloader class to handle
    audio file discovery and download, avoiding code duplication.

    Returns:
        Dictionary with success, metadata_csv, sample_count, error_message
    """
    try:
        scanner = callbacks.scanner

        # Create a GitLabDatasetDownloader instance for data operations
        # Configure it with output directories and text source
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=scanner.server_url,
            access_token=scanner.access_token,
            project_id=str(job_context['project_id']),
            config_overrides={
                'dataset.output_dir': str(output_dir),
                'dataset.audio_dir': 'audio',
                'dataset.csv_filename': 'metadata.csv',
                'dataset.text_source': 'value',  # Use cell value for text
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
            # Find all .codex files in the repository
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
            # and correctly resolves audio file paths with proper extensions
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

            # Download audio files in parallel (5 at a time) and add to samples
            def _download_one_pair(pair):
                """Download a single audio file. Returns (pair, success)."""
                transcription = pair['transcription']
                verse_id = pair['verse_id']

                # Skip pairs without transcription text
                if not transcription or not transcription.strip():
                    return (pair, False)

                audio_filename = Path(pair['audio_url']).name
                local_audio_path = audio_dir / audio_filename
                tmp_path = Path(str(local_audio_path) + '.tmp')

                # Clean up any stale .tmp file from a previous interrupted download
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

                # Already downloaded (only trust the final file, not .tmp)
                if local_audio_path.exists():
                    return (pair, True)

                if downloader.download_file(pair['audio_url'], local_audio_path):
                    return (pair, True)
                else:
                    print(f"    Warning: Could not download audio for {verse_id}")
                    return (pair, False)

            max_workers = 5
            completed_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_one_pair, p): p for p in pairs}

                for future in as_completed(futures):
                    pair, success = future.result()
                    completed_count += 1

                    if completed_count % 100 == 0 or completed_count == len(pairs):
                        print(f"    Downloaded {completed_count} of {len(pairs)}")

                    if success:
                        audio_filename = Path(pair['audio_url']).name
                        samples.append({
                            'file_name': f"audio/{audio_filename}",
                            'transcription': pair['transcription'],
                            'verse_id': pair['verse_id']
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

    Returns:
        Dictionary with success, local_path, error_message
    """
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        # Download checkpoint file
        checkpoint_filename = os.path.basename(checkpoint_path)
        local_path = output_dir / checkpoint_filename

        checkpoint_content = scanner.get_file_content(project_id, checkpoint_path, binary=True)
        if not checkpoint_content:
            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download checkpoint from {checkpoint_path}"
            }

        with open(local_path, 'wb') as f:
            f.write(checkpoint_content)

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


def _upload_checkpoint(
    job_context: Dict[str, Any],
    callbacks,
    checkpoint_path: str,
    job_id: str
) -> Dict[str, Any]:
    """
    Upload a checkpoint to GitLab using LFS.

    Returns:
        Dictionary with success, remote_path, error_message
    """
    try:
        # Determine remote path
        checkpoint_filename = os.path.basename(checkpoint_path)
        remote_path = f"gpu_jobs/job_{job_id}/checkpoint/{checkpoint_filename}"

        # Create uploader instance using callbacks credentials
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        # Upload checkpoint using LFS (model files should always use LFS)
        result = uploader.upload_batch(
            files=[{
                'local_path': checkpoint_path,
                'remote_path': remote_path,
                'lfs': True,  # Model checkpoints always use LFS
            }],
            commit_message=f"Upload checkpoint for job {job_id}"
        )

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
