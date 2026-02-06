"""
StableTTS Training and Inference Handler

Handles combined TTS training and inference jobs for the StableTTS model.
This mode saves bandwidth by:
1. Training the model locally
2. Using the trained model directly for inference (no upload/download cycle)
3. Uploading only the final checkpoint and generated audio

The training data and inference data can overlap or be different based on configuration.
"""

import os
import sys
import csv
import json
import traceback
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from handlers.base import (
    get_cells_needing_audio,
    get_cell_reference,
    get_cell_id,
    download_pretrained_model,
    resolve_use_uroman,
    STABLETTS_DEFAULT_REPO_ID,
    STABLETTS_DEFAULT_FILENAME,
)
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Import training and inference APIs
from preprocess_stabletts import preprocess_stabletts_api
from train_stabletts import train_stabletts_api
from inference_stabletts import inference_stabletts_api


def run(job_context: Dict[str, Any], callbacks) -> Dict[str, Any]:
    """
    Execute combined TTS training and inference job.

    This mode is optimized for bandwidth savings:
    - Downloads training data once
    - Trains the model locally
    - Uses the trained model directly for inference (no upload/download)
    - Uploads only the final checkpoint and generated audio

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
    print("StableTTS Training + Inference Handler")
    print("=" * 60)
    print("(Combined mode for bandwidth optimization)")
    print()

    try:
        # Extract job configuration
        job_config = job_context.get('job_config', {})
        model_config = job_context.get('model', {})
        training_config = job_config.get('training', {})
        inference_config = job_config.get('inference', {})

        # Get training parameters with defaults
        num_epochs = training_config.get('epochs', 100)
        batch_size = training_config.get('batch_size', 32)
        learning_rate = training_config.get('learning_rate', 1e-4)
        val_split = training_config.get('val_split', 0.1)
        save_interval = training_config.get('save_interval', 10)

        # Get model parameters
        model_type = model_config.get('type', 'StableTTS')
        base_checkpoint = model_config.get('base_checkpoint')
        reference_audio_path = model_config.get('reference_audio')
        language = model_config.get('language', 'english')
        # use_uroman is resolved after data download (may need auto-detection)
        uroman_lang = model_config.get('uroman_lang', None)

        # Get inference parameters
        diffusion_steps = inference_config.get('diffusion_steps', 10)
        temperature = inference_config.get('temperature', 1.0)
        length_scale = inference_config.get('length_scale', 1.0)
        cfg_scale = inference_config.get('cfg_scale', 3.0)
        audio_format = inference_config.get('audio_format', 'webm')

        # Get filter configuration
        training_include = training_config.get('include_verses', [])
        training_exclude = training_config.get('exclude_verses', [])
        inference_include = inference_config.get('include_verses', [])
        inference_exclude = inference_config.get('exclude_verses', [])

        print("Training Configuration:")
        print(f"  Model type: {model_type}")
        print(f"  Language: {language}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Base checkpoint: {base_checkpoint or 'None (will use default pretrained model)'}")
        print()
        print("Inference Configuration:")
        print(f"  Reference audio: {reference_audio_path}")
        print(f"  Diffusion steps: {diffusion_steps}")
        print(f"  Audio format: {audio_format}")
        print()

        # Validate required parameters
        if not reference_audio_path:
            return {
                'success': False,
                'error_message': "No reference_audio specified in model configuration"
            }

        # ============================================================
        # PHASE 1: Download all data (training + inference)
        # ============================================================
        print("=" * 60)
        print("PHASE 1: Downloading Data")
        print("=" * 60)

        data_dir = work_dir / "data"
        audio_dir = data_dir / "audio"
        data_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Download training data
        print("\nStep 1.1: Downloading training data...")
        callbacks.heartbeat(message="Downloading training data", stage="download")

        training_result = _download_training_data(
            job_context=job_context,
            callbacks=callbacks,
            output_dir=data_dir,
            audio_dir=audio_dir,
            include_verses=training_include,
            exclude_verses=training_exclude
        )

        if not training_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download training data: {training_result['error_message']}"
            }

        training_csv = training_result['metadata_csv']
        training_count = training_result['sample_count']
        print(f"  Downloaded {training_count} training samples")

        if training_count == 0:
            return {
                'success': False,
                'error_message': "No training samples found. Check that cells have both audio and text."
            }

        # Resolve use_uroman (auto-detect from text if not specified in manifest)
        use_uroman = resolve_use_uroman(model_config, training_result.get('sample_texts', []))
        print(f"  Use uroman: {use_uroman}")

        # Download inference data (cells needing audio)
        print("\nStep 1.2: Downloading inference data...")
        callbacks.heartbeat(message="Downloading inference data", stage="download")

        inference_result = _download_inference_data(
            job_context=job_context,
            callbacks=callbacks,
            output_dir=data_dir,
            include_verses=inference_include,
            exclude_verses=inference_exclude
        )

        if not inference_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download inference data: {inference_result['error_message']}"
            }

        inference_csv = inference_result['metadata_csv']
        inference_count = inference_result['sample_count']
        codex_data = inference_result['codex_data']
        print(f"  Found {inference_count} cells needing audio")

        # Download reference audio
        print("\nStep 1.3: Downloading reference audio...")
        callbacks.heartbeat(message="Downloading reference audio", stage="download")

        reference_dir = work_dir / "reference"
        reference_result = _download_file(
            job_context=job_context,
            callbacks=callbacks,
            remote_path=reference_audio_path,
            output_dir=reference_dir
        )

        if not reference_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download reference audio: {reference_result['error_message']}"
            }

        local_reference = reference_result['local_path']
        print(f"  Downloaded reference audio to: {local_reference}")

        # Download base checkpoint if specified, otherwise fetch default pretrained model
        pretrained_model = None
        if base_checkpoint:
            print("\nStep 1.4: Downloading base checkpoint from GitLab...")
            callbacks.heartbeat(message="Downloading base checkpoint", stage="download")

            checkpoint_result = _download_file(
                job_context=job_context,
                callbacks=callbacks,
                remote_path=base_checkpoint,
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
            print("\nStep 1.4: Downloading default pretrained model from HuggingFace Hub...")
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

        # ============================================================
        # PHASE 2: Preprocess and Train
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Training")
        print("=" * 60)

        # Preprocess training data
        print("\nStep 2.1: Preprocessing training data...")
        callbacks.heartbeat(message="Preprocessing data", stage="preprocess")

        feature_dir = work_dir / "features"
        output_json = feature_dir / "filelist.json"

        preprocess_result = preprocess_stabletts_api(
            input_csv=str(training_csv),
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

        # Train the model
        print("\nStep 2.2: Training StableTTS model...")
        callbacks.heartbeat(epochs_completed=0, message="Starting training", stage="training")

        checkpoint_dir = work_dir / "checkpoints"
        log_dir = work_dir / "logs"

        def training_heartbeat(epoch: int):
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

        trained_checkpoint = train_result['best_model_path']
        print(f"  Training completed: {train_result['epochs_completed']} epochs")
        print(f"  Best model: {trained_checkpoint}")

        # ============================================================
        # PHASE 3: Inference (using locally trained model)
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 3: Inference")
        print("=" * 60)

        if inference_count == 0:
            print("\nNo cells need audio generation. Skipping inference phase.")
        else:
            print(f"\nStep 3.1: Running TTS inference on {inference_count} cells...")
            print("  (Using locally trained model - no download needed)")
            callbacks.heartbeat(message="Running inference", stage="inference")

            output_dir = work_dir / "output"

            inference_api_result = inference_stabletts_api(
                checkpoint=trained_checkpoint,
                input_csv=str(inference_csv),
                output_dir=str(output_dir),
                reference_audio=local_reference,
                language=language,
                text_column='transcription',
                diffusion_steps=diffusion_steps,
                temperature=temperature,
                length_scale=length_scale,
                cfg_scale=cfg_scale,
                solver=None,
                use_uroman=use_uroman,
                uroman_lang=uroman_lang,
                batch_size=1,
                audio_format=audio_format,
                resume=False,
                overwrite=True,
                heartbeat_callback=lambda: callbacks.heartbeat(message="Running inference", stage="inference")
            )

            if not inference_api_result['success']:
                return {
                    'success': False,
                    'error_message': f"Inference failed: {inference_api_result['error_message']}"
                }

            print(f"  Generated {inference_api_result['processed_count']} audio files")

        # ============================================================
        # PHASE 4: Upload Results
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 4: Uploading Results")
        print("=" * 60)

        # Upload checkpoint
        print("\nStep 4.1: Uploading trained checkpoint...")
        callbacks.heartbeat(message="Uploading checkpoint", stage="upload")

        checkpoint_upload = _upload_checkpoint(
            job_context=job_context,
            callbacks=callbacks,
            checkpoint_path=trained_checkpoint,
            job_id=job_context['job_id']
        )

        if not checkpoint_upload['success']:
            print(f"  Warning: Failed to upload checkpoint: {checkpoint_upload['error_message']}")
        else:
            print(f"  Uploaded checkpoint to: {checkpoint_upload['remote_path']}")

        # Upload audio files and update .codex files
        if inference_count > 0:
            print("\nStep 4.2: Uploading audio files and updating .codex files...")
            callbacks.heartbeat(message="Uploading audio", stage="upload")

            audio_upload = _upload_audio_and_update_codex(
                job_context=job_context,
                callbacks=callbacks,
                output_dir=output_dir,
                metadata_csv=inference_api_result['metadata_csv'],
                codex_data=codex_data,
                audio_format=audio_format
            )

            if not audio_upload['success']:
                return {
                    'success': False,
                    'error_message': f"Failed to upload audio: {audio_upload['error_message']}"
                }

            print(f"  Uploaded {audio_upload['uploaded_count']} audio files")

        print("\n" + "=" * 60)
        print("Training + Inference completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'epochs_completed': train_result['epochs_completed'],
            'checkpoint_path': checkpoint_upload.get('remote_path'),
            'audio_generated': inference_api_result.get('processed_count', 0) if inference_count > 0 else 0
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\nJob failed with error: {error_msg}")
        return {
            'success': False,
            'error_message': error_msg
        }


# ============================================================
# Helper Functions
# ============================================================

def _download_file(
    job_context: Dict[str, Any],
    callbacks,
    remote_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Download a file from GitLab."""
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(remote_path)
        local_path = output_dir / filename

        file_content = scanner.get_file_content(project_id, remote_path, binary=True)
        if not file_content:
            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download file from {remote_path}"
            }

        with open(local_path, 'wb') as f:
            f.write(file_content)

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


def _download_training_data(
    job_context: Dict[str, Any],
    callbacks,
    output_dir: Path,
    audio_dir: Path,
    include_verses: list,
    exclude_verses: list
) -> Dict[str, Any]:
    """Download training data (cells with both audio and text) using GitLabDatasetDownloader."""
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
                'dataset.csv_filename': 'training_metadata.csv',
                'dataset.text_source': 'value',
            }
        )

        # List all files in the repository to build audio map
        print("    Listing repository files...")
        items = downloader.list_repository_tree()

        # Build audio files map using the existing method
        print("    Building audio files map...")
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
            print(f"    Found {len(codex_files)} .codex files in repository")

        metadata_csv = output_dir / "training_metadata.csv"
        samples = []

        # Convert include/exclude lists to sets for faster lookup
        include_set = set(include_verses) if include_verses else None
        exclude_set = set(exclude_verses) if exclude_verses else None

        for codex_path in codex_files:
            print(f"    Processing: {codex_path}")

            # Download and parse .codex file using downloader
            codex_data = downloader.download_json_file(codex_path)
            if not codex_data:
                print(f"      Warning: Could not download {codex_path}")
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

            print(f"      Found {len(pairs)} audio-transcription pairs")

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
                    print(f"      Warning: Could not download audio for {verse_id}")
                    return (pair, False)

            max_workers = 15
            completed_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_one_pair, p): p for p in pairs}

                for future in as_completed(futures):
                    pair, success = future.result()
                    completed_count += 1

                    if completed_count % 100 == 0 or completed_count == len(pairs):
                        print(f"      Downloaded {completed_count} of {len(pairs)}")

                    if success:
                        audio_filename = Path(pair['audio_url']).name
                        samples.append({
                            'file_name': f"audio/{audio_filename}",
                            'transcription': pair['transcription'],
                            'verse_id': pair['verse_id']
                        })

            callbacks.heartbeat(message=f"Downloaded {len(samples)} training samples", stage="download")

        with open(metadata_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'transcription', 'verse_id'])
            writer.writeheader()
            writer.writerows(samples)

        return {
            'success': True,
            'metadata_csv': metadata_csv,
            'sample_count': len(samples),
            'sample_texts': [s['transcription'] for s in samples],
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'metadata_csv': None,
            'sample_count': 0,
            'sample_texts': [],
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def _download_inference_data(
    job_context: Dict[str, Any],
    callbacks,
    output_dir: Path,
    include_verses: list,
    exclude_verses: list
) -> Dict[str, Any]:
    """Download inference data (cells needing audio)."""
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        job_config = job_context.get('job_config', {})
        codex_files = job_config.get('codex_files', [])

        if not codex_files:
            return {
                'success': False,
                'metadata_csv': None,
                'sample_count': 0,
                'codex_data': {},
                'error_message': "No codex_files specified in job configuration"
            }

        metadata_csv = output_dir / "inference_metadata.csv"
        samples = []
        codex_data = {}

        for codex_path in codex_files:
            print(f"    Processing: {codex_path}")

            codex_content = scanner.get_file_content(project_id, codex_path)
            if not codex_content:
                print(f"      Warning: Could not download {codex_path}")
                continue

            codex_json = json.loads(codex_content)
            codex_data[codex_path] = codex_json
            cells = codex_json.get('cells', [])

            inference_cells = get_cells_needing_audio(
                cells,
                include_verses=include_verses if include_verses else None,
                exclude_verses=exclude_verses if exclude_verses else None
            )

            print(f"      Found {len(inference_cells)} cells needing audio")

            for cell in inference_cells:
                cell_id = get_cell_id(cell)
                cell_ref = get_cell_reference(cell) or cell_id
                text = cell.get('value', '').strip()

                if not text:
                    continue

                samples.append({
                    'verse_id': cell_ref,
                    'cell_id': cell_id,
                    'transcription': text,
                    'codex_path': codex_path
                })

            callbacks.heartbeat(message=f"Found {len(samples)} cells needing audio", stage="download")

        with open(metadata_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['verse_id', 'cell_id', 'transcription', 'codex_path'])
            writer.writeheader()
            writer.writerows(samples)

        return {
            'success': True,
            'metadata_csv': metadata_csv,
            'sample_count': len(samples),
            'codex_data': codex_data,
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'metadata_csv': None,
            'sample_count': 0,
            'codex_data': {},
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def _upload_checkpoint(
    job_context: Dict[str, Any],
    callbacks,
    checkpoint_path: str,
    job_id: str
) -> Dict[str, Any]:
    """Upload checkpoint to GitLab using LFS."""
    try:
        checkpoint_filename = os.path.basename(checkpoint_path)
        remote_path = f"gpu_jobs/job_{job_id}/checkpoint/{checkpoint_filename}"

        # Create uploader instance using callbacks credentials
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        # Upload checkpoint using LFS
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


def _upload_audio_and_update_codex(
    job_context: Dict[str, Any],
    callbacks,
    output_dir: Path,
    metadata_csv: str,
    codex_data: Dict[str, Any],
    audio_format: str
) -> Dict[str, Any]:
    """Upload audio files (LFS) and update .codex files (regular git) in a single commit."""
    try:
        job_id = job_context['job_id']

        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Collect all files to upload
        files_to_upload = []
        codex_updates = {}

        for row in rows:
            cell_id = row.get('cell_id')
            codex_path = row.get('codex_path')
            audio_filename = row.get('file_name', '')

            if not audio_filename:
                continue

            local_audio_path = Path(output_dir) / audio_filename
            if not local_audio_path.exists():
                print(f"      Warning: Audio file not found: {local_audio_path}")
                continue

            audio_id = f"audio-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            remote_audio_path = f".project/audio/{audio_id}.{audio_format}"

            # Add audio file to upload list (LFS)
            files_to_upload.append({
                'local_path': str(local_audio_path),
                'remote_path': remote_audio_path,
                'lfs': True,  # Audio files always use LFS
            })

            if codex_path not in codex_updates:
                codex_updates[codex_path] = []

            codex_updates[codex_path].append({
                'cell_id': cell_id,
                'audio_id': audio_id,
                'audio_format': audio_format
            })

        # Update .codex files in memory and add to upload list
        for codex_path, updates in codex_updates.items():
            if codex_path not in codex_data:
                continue

            codex_json = codex_data[codex_path]
            cells = codex_json.get('cells', [])

            for update in updates:
                cell_id = update['cell_id']
                audio_id = update['audio_id']
                fmt = update['audio_format']

                for cell in cells:
                    if get_cell_id(cell) == cell_id:
                        metadata = cell.setdefault('metadata', {})
                        attachments = metadata.setdefault('attachments', {})

                        attachments[audio_id] = {
                            'type': 'audio',
                            'format': fmt,
                            'isGenerated': True
                        }

                        metadata['selectedAudioId'] = audio_id
                        break

            # Add updated codex file to upload list (regular git)
            updated_content = json.dumps(codex_json, ensure_ascii=False, indent=2)
            files_to_upload.append({
                'content': updated_content,
                'remote_path': codex_path,
                'lfs': False,  # Codex files use regular git
            })

        if not files_to_upload:
            return {
                'success': True,
                'uploaded_count': 0,
                'error_message': None
            }

        # Create uploader instance
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        # Upload all files in a single commit
        callbacks.heartbeat(message=f"Uploading {len(files_to_upload)} files", stage="upload")
        result = uploader.upload_batch(
            files=files_to_upload,
            commit_message=f"TTS inference results for job {job_id}"
        )

        # Count audio files uploaded
        audio_count = sum(1 for f in result.get('files_uploaded', []) if f.get('lfs', False))

        if result['success']:
            return {
                'success': True,
                'uploaded_count': audio_count,
                'error_message': None
            }
        else:
            return {
                'success': len(result.get('files_uploaded', [])) > 0,
                'uploaded_count': audio_count,
                'error_message': result.get('error_message'),
                'files_failed': result.get('files_failed', [])
            }

    except Exception as e:
        return {
            'success': False,
            'uploaded_count': 0,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }
