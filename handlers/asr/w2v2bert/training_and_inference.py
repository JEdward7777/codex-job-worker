"""
W2V2-BERT ASR Training and Inference Handler

Combined handler that trains an ASR model and then uses it for inference
in a single job. This saves bandwidth by not uploading/downloading the
trained model between steps.

Supports both:
- facebook/w2v-bert-2.0 (more accurate, requires more VRAM)
- facebook/wav2vec2-base (less accurate, requires less VRAM)
"""
#pylint: disable=broad-exception-caught

import os
import sys
import csv
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Sequence
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from handlers.base import (
    get_cells_needing_text,
    get_cell_reference,
    get_cell_id,
    create_tar_archive,
)
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Import training and inference APIs
from train_w2v2bert_asr import train_w2v2bert_asr_api
from inference_w2v2bert_asr import (
    load_model_and_processor,
    transcribe_audio,
)

# Optional SentenceTransmorgrifier import
try:
    from sentence_transmorgrifier.transmorgrify import Transmorgrifier
    TRANSMORGRIFIER_AVAILABLE = True
except ImportError:
    TRANSMORGRIFIER_AVAILABLE = False
    Transmorgrifier = None


def run(job_context: Dict[str, Any], callbacks) -> Dict[str, Any]:
    """
    Execute combined ASR training and inference job.

    This handler:
    1. Downloads training data from GitLab
    2. Trains the ASR model
    3. Optionally trains a SentenceTransmorgrifier for post-processing
    4. Uses the trained model directly for inference (no upload/download)
    5. Updates .codex files with transcriptions
    6. Uploads the trained model to GitLab

    Args:
        job_context: Job configuration from GitLab manifest
        callbacks: JobCallbacks object for heartbeat, file operations, etc.

    Returns:
        Dictionary with:
            - success: bool
            - error_message: str (if success is False)
            - epochs_completed: int
            - cells_transcribed: int
    """
    work_dir = callbacks.get_work_dir()

    print("=" * 60)
    print("W2V2-BERT ASR Training + Inference Handler")
    print("=" * 60)

    try:
        # Extract job configuration
        # The manifest places most fields at the top level of the job dict.
        model_config = job_context.get('model', {})
        training_config = job_context.get('training', {})
        inference_config = job_context.get('inference', {})

        # Get model parameters
        base_model = model_config.get('base_model', 'facebook/w2v-bert-2.0')
        use_wav2vec2_base = model_config.get('use_wav2vec2_base')
        if use_wav2vec2_base is None:
            use_wav2vec2_base = 'bert' not in base_model.lower()

        # Set memory-aware and convergence-aware defaults based on model size.
        # W2V2-BERT 2.0 (~600M params) needs smaller batch size, memory
        # optimizations, lower learning rate, and longer warmup to fit on a
        # 24GB GPU and preserve pretrained representations.
        # The smaller wav2vec2-base (~95M params) can use larger batches and
        # higher learning rates comfortably.
        if use_wav2vec2_base:
            default_batch_size = 8
            default_gradient_accumulation = 2
            default_use_8bit_optimizer = False
            default_learning_rate = 3e-4
            default_warmup_steps = 500
            default_epochs = 5
        else:
            default_batch_size = 2
            default_gradient_accumulation = 8
            default_use_8bit_optimizer = True
            default_learning_rate = 3e-5   # Lower LR preserves pretrained weights
            default_warmup_steps = 2000    # Longer warmup avoids gradient instability
            default_epochs = 50            # Fewer epochs needed with proper LR

        # Get training parameters with defaults
        # 'epochs' is a top-level manifest field per the spec.
        num_epochs = job_context.get('epochs', default_epochs)
        batch_size = job_context.get('batch_size', default_batch_size)
        learning_rate = job_context.get('learning_rate', default_learning_rate)
        gradient_accumulation_steps = job_context.get('gradient_accumulation_steps', default_gradient_accumulation)
        warmup_steps = job_context.get('warmup_steps', default_warmup_steps)
        save_steps = job_context.get('save_steps', 500)
        eval_steps = job_context.get('eval_steps', 500)
        eval_strategy = job_context.get('eval_strategy', 'epoch')
        val_split = job_context.get('val_split', 0.1)
        test_split = job_context.get('test_split', 0.1)
        max_duration_seconds = job_context.get('max_duration_seconds')
        use_8bit_optimizer = job_context.get('use_8bit_optimizer', default_use_8bit_optimizer)

        # Text normalization options (handler-specific, not in manifest spec)
        text_config = job_context.get('text_normalization', {})
        lowercase = text_config.get('lowercase', True)
        remove_punctuation = text_config.get('remove_punctuation', True)
        remove_numbers = text_config.get('remove_numbers', False)

        # Get SentenceTransmorgrifier config
        # The manifest only has { enabled: boolean }.  When enabled, we train
        # a new transmorgrifier using the ASR model's own predictions and
        # bundle the .tm file inside the model archive.
        tm_config = job_context.get('transmorgrifier', {})
        train_transmorgrifier = tm_config.get('enabled', False)

        # Get filter configuration
        # Per spec, training filters under 'training', inference under 'inference'
        include_verses = training_config.get('include_verses', [])
        exclude_verses = training_config.get('exclude_verses', [])

        # UNK token handling
        suppress_unk = inference_config.get('suppress_unk', True)
        unk_replacement = inference_config.get('unk_replacement', '')

        # Eval batch size can be larger than train batch size since evaluation
        # doesn't store gradients/activations for backprop.
        eval_batch_size = max(batch_size, 8)

        model_arch = "Wav2Vec2" if use_wav2vec2_base else "Wav2Vec2-BERT"
        print("\nConfiguration:")
        print(f"  Model architecture: {model_arch}")
        print(f"  Base model: {base_model}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Batch size (train): {batch_size}")
        print(f"  Batch size (eval): {eval_batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  SentenceTransmorgrifier training: {'Enabled' if train_transmorgrifier else 'Disabled'}")
        print()

        # Setup directories
        data_dir = work_dir / "data"
        audio_dir = data_dir / "audio"
        output_dir = work_dir / "model_output"
        data_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        project_id = job_context['project_id']

        # ============================================================
        # PHASE 1: Download and prepare training data
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 1: Preparing Training Data")
        print("=" * 60)
        callbacks.heartbeat(message="Downloading training data", stage="download")

        # Create downloader for audio and codex file operations
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(project_id),
            config_overrides={
                'dataset.output_dir': str(data_dir),
            }
        )

        # List repository files and build audio files map to get correct file extensions
        items = downloader.list_repository_tree()
        audio_files_map = downloader.build_audio_files_map(items)
        print(f"  Found {len(audio_files_map)} audio files in repository")

        # Auto-discover .codex files from the repository
        codex_files = [item['path'] for item in items if item['name'].endswith('.codex')]
        if not codex_files:
            return {
                'success': False,
                'error_message': "No .codex files found in repository"
            }
        print(f"  Found {len(codex_files)} .codex files in repository")

        # Download training data (cells with both audio and text)
        training_samples = []
        all_codex_data = {}  # Store for later modification

        for codex_path in codex_files:
            print(f"\n  Processing: {codex_path}")

            codex_data = downloader.download_json_file(codex_path)
            if not codex_data:
                print(f"    Warning: Could not download {codex_path}")
                continue
            cells = codex_data.get('cells', [])
            all_codex_data[codex_path] = codex_data

            # Extract audio-transcription pairs using downloader method
            # This properly handles isDeleted/isMissing checks and file extensions
            audio_transcriptions = downloader.extract_audio_transcriptions(
                codex_data,
                audio_files_map,
            )

            # Apply include/exclude filtering for training data
            if include_verses or exclude_verses:
                def matches_filter(item, filter_list):
                    verse_id = item.get('verse_id', '')
                    return any(
                        verse_id == f  # Exact match only
                        for f in filter_list
                    )

                if exclude_verses:
                    audio_transcriptions = [
                        item for item in audio_transcriptions
                        if not matches_filter(item, exclude_verses)
                    ]
                if include_verses:
                    audio_transcriptions = [
                        item for item in audio_transcriptions
                        if matches_filter(item, include_verses)
                    ]

            print(f"    Found {len(audio_transcriptions)} cells for training")

            for item in audio_transcriptions:
                audio_id = item['audio_id']
                audio_path = item['audio_url']  # extract_audio_transcriptions returns 'audio_url'
                text = item['transcription']
                cell_ref = item.get('verse_id', audio_id)

                # Download audio file
                audio_filename = os.path.basename(audio_path)
                local_audio_path = audio_dir / audio_filename

                if not local_audio_path.exists():
                    success = downloader.download_file(audio_path, local_audio_path)
                    if not success:
                        print(f"    Warning: Could not download audio for {cell_ref}")
                        continue

                training_samples.append({
                    'file_name': audio_filename,
                    'transcription': text,  # Original un-normalized text from codex
                    'verse_id': cell_ref
                })

            callbacks.heartbeat(message=f"Downloaded {len(training_samples)} training samples", stage="download")

        print(f"\n  Total training samples: {len(training_samples)}")

        if len(training_samples) == 0:
            return {
                'success': False,
                'error_message': "No training samples found. Check that cells have both audio and text."
            }

        # Write training metadata CSV
        metadata_csv = data_dir / "metadata.csv"
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'transcription', 'verse_id'])
            writer.writeheader()
            writer.writerows(training_samples)

        # ============================================================
        # PHASE 2: Train the model
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Training ASR Model")
        print("=" * 60)
        callbacks.heartbeat(epochs_completed=0, message="Starting training", stage="training")

        def training_heartbeat(epoch: int):
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
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy=eval_strategy,
            use_8bit_optimizer=use_8bit_optimizer,
            heartbeat_callback=training_heartbeat,
            disable_progress_bars=True,
        )

        if not train_result['success']:
            return {
                'success': False,
                'error_message': f"Training failed: {train_result['error_message']}"
            }

        print(f"\n  Training completed: {train_result['epochs_completed']} epochs")
        print(f"  Final model: {train_result['final_model_path']}")

        # Write training metrics CSV
        metrics_csv_path = None
        log_history = train_result.get('log_history')
        if log_history:
            metrics_csv_path = _write_training_metrics_csv(
                log_history, work_dir / "training_metrics.csv"
            )
            if metrics_csv_path:
                print(f"  Training metrics CSV written: {metrics_csv_path}")

        # ============================================================
        # PHASE 3: Load trained model for inference
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 3: Loading Trained Model for Inference")
        print("=" * 60)
        callbacks.heartbeat(message="Loading trained model", stage="inference")


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor, _ = load_model_and_processor(
            train_result['final_model_path'],
            device,
            use_wav2vec2_base
        )

        print(f"  Loaded model on {device}")

        # ============================================================
        # PHASE 3b: Train SentenceTransmorgrifier (if enabled)
        # ============================================================
        transmorgrifier = None
        if train_transmorgrifier:
            print("\n" + "=" * 60)
            print("PHASE 3b: Training SentenceTransmorgrifier")
            print("=" * 60)

            if not TRANSMORGRIFIER_AVAILABLE:
                print("  Warning: SentenceTransmorgrifier not installed, skipping")
                print("  Install with: pip install sentence-transmorgrifier")
            else:
                callbacks.heartbeat(message="Training SentenceTransmorgrifier", stage="transmorgrifier")

                transmorgrifier = _train_transmorgrifier(
                    training_samples=training_samples,
                    audio_dir=audio_dir,
                    model=model,
                    processor=processor,
                    device=device,
                    model_output_dir=Path(train_result['final_model_path']),
                    callbacks=callbacks,
                )

                if transmorgrifier:
                    print("  SentenceTransmorgrifier trained and saved into model directory")
                else:
                    print("  Warning: SentenceTransmorgrifier training failed, continuing without it")

        # ============================================================
        # PHASE 4: Transcribe cells needing text
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 4: Transcribing Cells")
        print("=" * 60)
        callbacks.heartbeat(message="Transcribing cells", stage="inference")

        # Determine UNK replacement strategy:
        # If transmorgrifier is active, use "*" so it can learn to handle UNK tokens.
        # Otherwise, fall back to the configured suppress_unk / unk_replacement.
        if transmorgrifier:
            effective_unk_replacement = "*"
        elif suppress_unk:
            effective_unk_replacement = unk_replacement
        else:
            effective_unk_replacement = None

        total_transcribed = 0
        total_errors = 0
        modified_codex_files = []

        for codex_path, codex_data in all_codex_data.items():
            print(f"\n  Processing: {codex_path}")

            cells = codex_data.get('cells', [])

            # Find cells needing transcription
            # For inference phase, use inference-specific filters
            inference_include = inference_config.get('include_verses', [])
            inference_exclude = inference_config.get('exclude_verses', [])
            cells_to_process = get_cells_needing_text(
                cells,
                include_verses=inference_include if inference_include else None,
                exclude_verses=inference_exclude if inference_exclude else None
            )

            print(f"    Found {len(cells_to_process)} cells needing transcription")

            if not cells_to_process:
                continue

            codex_modified = False

            for idx, cell in enumerate(cells_to_process):
                cell_id = get_cell_id(cell)
                cell_ref = get_cell_reference(cell) or cell_id

                metadata = cell.get('metadata', {})
                selected_audio_id = metadata.get('selectedAudioId')
                attachments = metadata.get('attachments', {})

                if not selected_audio_id or selected_audio_id not in attachments:
                    continue

                # Check if audio exists locally (may have been downloaded for training)
                # Look up the actual file path from audio_files_map
                if selected_audio_id not in audio_files_map:
                    print(f"    Warning: Audio file not found in repository for {cell_ref}")
                    total_errors += 1
                    continue

                audio_repo_path = audio_files_map[selected_audio_id]
                audio_filename = os.path.basename(audio_repo_path)
                local_audio_path = audio_dir / audio_filename

                if not local_audio_path.exists():
                    # Download audio using downloader
                    success = downloader.download_file(audio_repo_path, local_audio_path)
                    if not success:
                        print(f"    Warning: Could not download audio for {cell_ref}")
                        total_errors += 1
                        continue

                # Transcribe
                result = transcribe_audio(
                    str(local_audio_path),
                    model,
                    processor,
                    device,
                    return_confidence=True,
                    unk_token_replacement=effective_unk_replacement,
                )

                if result.get('error'):
                    print(f"    Error transcribing {cell_ref}: {result['error']}")
                    total_errors += 1
                    continue

                transcription = result['transcription']

                # Post-process with SentenceTransmorgrifier
                if transmorgrifier and transcription:
                    try:
                        processed = list(transmorgrifier.execute([transcription]))[0]
                        transcription = processed
                    except Exception as e:
                        print(f"    Warning: Transmorgrifier failed for {cell_ref}: {e}")

                # Update cell
                for original_cell in cells:
                    if get_cell_id(original_cell) == cell_id:
                        original_cell['value'] = transcription
                        codex_modified = True
                        total_transcribed += 1
                        break

                if (idx + 1) % 10 == 0:
                    callbacks.heartbeat(
                        message=f"Transcribed {total_transcribed} cells",
                        stage="inference"
                    )

            if codex_modified:
                codex_data['cells'] = cells
                modified_codex_files.append({
                    'path': codex_path,
                    'content': json.dumps(codex_data, ensure_ascii=False, indent=2)
                })

        print(f"\n  Total cells transcribed: {total_transcribed}")

        # ============================================================
        # PHASE 5: Upload results
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 5: Uploading Results")
        print("=" * 60)
        callbacks.heartbeat(message="Uploading results", stage="upload")

        # Upload modified .codex files in a single commit
        if modified_codex_files:
            print("\n  Uploading modified .codex files...")

            # Create uploader instance
            uploader = GitLabDatasetDownloader(
                config_path=None,
                gitlab_url=callbacks.scanner.server_url,
                access_token=callbacks.scanner.access_token,
                project_id=str(project_id),
            )

            # Prepare files for batch upload (codex files use regular git, not LFS)
            files_to_upload = [
                {
                    'content': codex_info['content'],
                    'remote_path': codex_info['path'],
                    'lfs': False,  # Codex files use regular git
                }
                for codex_info in modified_codex_files
            ]

            result = uploader.upload_batch(
                files=files_to_upload,
                commit_message=f"ASR transcription for job {job_context['job_id']}"
            )

            if result['success']:
                for f in result.get('files_uploaded', []):
                    print(f"    Updated: {f['remote_path']}")
            else:
                print(f"    Error uploading codex files: {result.get('error_message')}")
                for f in result.get('files_failed', []):
                    print(f"      Failed: {f['remote_path']}: {f['error']}")
                    total_errors += 1

        # Upload trained model (the .tm file is already inside the model directory
        # if transmorgrifier training succeeded, so it gets bundled automatically)
        print("\n  Uploading trained model...")
        upload_result = _upload_model(
            job_context=job_context,
            callbacks=callbacks,
            model_path=train_result['final_model_path'],
            job_id=job_context['job_id'],
            metrics_csv_path=str(metrics_csv_path) if metrics_csv_path else None,
        )

        if upload_result['success']:
            print(f"    Uploaded model to: {upload_result['remote_path']}")
        else:
            error_details = upload_result.get('error_message', 'Unknown error')
            print(f"    Warning: Failed to upload model: {error_details}")
            files_failed = upload_result.get('files_failed', [])
            if files_failed:
                print(f"\n    Upload failure details ({len(files_failed)} file(s) failed):")
                for ff in files_failed:
                    print(f"      - {ff.get('remote_path', '?')}: {ff.get('error', '?')}")

        print("\n" + "=" * 60)
        print("Training and Inference completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'epochs_completed': train_result['epochs_completed'],
            'cells_transcribed': total_transcribed,
            'errors_count': total_errors,
            'checkpoint_path': upload_result.get('remote_path'),
            'test_results': train_result.get('test_results'),
            'train_metrics': train_result.get('train_metrics')
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\nJob failed with error: {error_msg}")
        return {
            'success': False,
            'error_message': error_msg
        }


def _train_transmorgrifier(
    training_samples: List[Dict[str, str]],
    audio_dir: Path,
    model,
    processor,
    device: str,
    model_output_dir: Path,
    callbacks,
) -> Optional[Any]:
    """
    Train a SentenceTransmorgrifier using the ASR model's own predictions.

    1. Run inference on all training audio to get the model's predictions
    2. Replace [UNK] with * in predictions (from_sentences)
    3. Use original un-normalized text as to_sentences
    4. Train the transmorgrifier
    5. Save the .tm file into the model output directory

    Returns the trained Transmorgrifier instance, or None on failure.
    """
    try:
        from_sentences = []
        to_sentences = []

        total = len(training_samples)
        print(f"  Running inference on {total} training samples...")

        for i, sample in enumerate(training_samples):
            audio_path = str(audio_dir / sample['file_name'])
            original_text = sample['transcription']  # Un-normalized ground truth

            if not os.path.exists(audio_path):
                continue

            # Run ASR inference with [UNK] → *
            result = transcribe_audio(
                audio_path,
                model,
                processor,
                device,
                return_confidence=False,
                unk_token_replacement="*",
            )

            if result.get('error'):
                continue

            prediction = result['transcription']
            from_sentences.append(prediction)
            to_sentences.append(original_text)

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{total} samples")
                callbacks.heartbeat(
                    message=f"Transmorgrifier: processed {i + 1}/{total} samples",
                    stage="transmorgrifier"
                )

        print(f"  Collected {len(from_sentences)} sentence pairs for transmorgrifier training")

        if len(from_sentences) < 2:
            print("  Warning: Not enough sentence pairs to train transmorgrifier")
            return None

        # Train the transmorgrifier
        print("  Training SentenceTransmorgrifier...")
        callbacks.heartbeat(message="Training SentenceTransmorgrifier model", stage="transmorgrifier")

        tm = Transmorgrifier()
        tm_device = 'GPU' if device == 'cuda' else 'CPU'
        tm.train(
            from_sentences=from_sentences,
            to_sentences=to_sentences,
            iterations=2000,
            device=tm_device,
            verbose=True,
        )

        # Save the .tm file into the model output directory so it gets
        # bundled into the asr_model.pth.tar archive automatically
        tm_path = str(model_output_dir / "transmorgrifier.tm")
        tm.save(tm_path)
        print(f"  Saved transmorgrifier to: {tm_path}")

        return tm

    except Exception as e:
        print(f"  Error training transmorgrifier: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def _write_training_metrics_csv(
    log_history: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    """
    Extract training metrics from HuggingFace Trainer log_history and write
    a CSV suitable for the frontend chart.

    The log_history contains interleaved training-loss and eval entries.
    We write *both* kinds of entries so the CSV has dense data:

    - Training-loss entries (logged every ``logging_steps``) produce rows
      with ``train_total_loss`` filled and eval columns empty.
    - Eval entries produce rows with ``val_total_loss``, ``val_wer``, and
      ``val_cer`` filled, plus the most recent ``train_total_loss``.

    Column names use the ``train_total_loss`` / ``val_total_loss`` convention
    expected by the frontend chart (see ``wizard.js`` and
    ``TrainingMetricsData.hasPrimaryColumns``).

    This ensures the frontend always has enough data points to render a
    meaningful chart, regardless of how infrequently evaluation runs.

    Returns the path to the written CSV, or None if no entries exist.
    """
    if not log_history:
        return None

    last_train_loss = None
    rows = []

    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            # Training-loss log entry (emitted every logging_steps)
            last_train_loss = entry.get('loss')
            rows.append({
                'epoch': entry.get('epoch', ''),
                'train_total_loss': last_train_loss,
                'val_total_loss': '',
                'val_wer': '',
                'val_cer': '',
            })
        elif 'eval_loss' in entry:
            # Evaluation entry (emitted every eval_steps or every epoch)
            rows.append({
                'epoch': entry.get('epoch', ''),
                'train_total_loss': last_train_loss if last_train_loss is not None else '',
                'val_total_loss': entry.get('eval_loss', ''),
                'val_wer': entry.get('eval_wer', ''),
                'val_cer': entry.get('eval_cer', ''),
            })

    if not rows:
        return None

    columns = ['epoch', 'train_total_loss', 'val_total_loss', 'val_wer', 'val_cer']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _upload_model(
    job_context: Dict[str, Any],
    callbacks,
    model_path: str,
    job_id: str,
    metrics_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload trained model (and optional artifacts) to GitLab using LFS as a .pth.tar archive.

    The model directory may contain a transmorgrifier.tm file which will be
    included in the archive automatically.

    Returns:
        Dictionary with success, remote_path, error_message, files_failed
    """
    try:
        model_dir = Path(model_path)

        # Create tar archive of the model directory (streams directly to disk)
        archive_filename = "asr_model.pth.tar"
        archive_path = model_dir.parent / archive_filename

        print(f"    Creating model archive: {archive_filename}")
        create_tar_archive(model_dir, archive_path)

        # Log archive size so we know what we're trying to upload
        if archive_path.exists():
            archive_size = archive_path.stat().st_size
            archive_mb = archive_size / (1024 * 1024)
            print(f"    Archive size: {archive_mb:.1f} MB ({archive_size:,} bytes)")
        else:
            return {
                'success': False,
                'remote_path': None,
                'error_message': f"Archive creation failed: {archive_path} does not exist",
                'files_failed': [],
            }

        # Remote path for the archive
        remote_path = f"gpu_jobs/job_{job_id}/model/{archive_filename}"

        # Build file list for batch upload
        files: List[Dict[str, Any]] = [
            {
                'local_path': str(archive_path),
                'remote_path': remote_path,
                'lfs': True,  # Model archives always use LFS
            }
        ]

        if metrics_csv_path and os.path.exists(metrics_csv_path):
            csv_remote = f"gpu_jobs/job_{job_id}/checkpoint/training_metrics.csv"
            files.append({
                'local_path': metrics_csv_path,
                'remote_path': csv_remote,
                'lfs': False,  # Small CSV, no need for LFS
            })

        # Create uploader instance
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        print(f"    Uploading {len(files)} file(s) to GitLab...")

        # Upload all files in a single batch commit
        result = uploader.upload_batch(
            files=files,
            commit_message=f"Upload ASR model and artifacts for job {job_id}"
        )

        # Clean up local archive
        if archive_path.exists():
            archive_path.unlink()

        if result['success']:
            return {
                'success': True,
                'remote_path': remote_path,
                'error_message': None,
                'files_failed': [],
            }
        else:
            return {
                'success': False,
                'remote_path': None,
                'error_message': result.get('error_message', 'Upload failed'),
                'files_failed': result.get('files_failed', []),
            }

    except Exception as e:
        return {
            'success': False,
            'remote_path': None,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            'files_failed': [],
        }
