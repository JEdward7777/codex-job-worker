"""
W2V2-BERT ASR Inference Handler

Handles ASR inference jobs for the W2V2-BERT model (and traditional Wav2Vec2).
Downloads audio from GitLab, transcribes it, optionally post-processes with
SentenceTransmorgrifier, and updates the .codex files with transcriptions.

Supports both:
- facebook/w2v-bert-2.0 (more accurate, requires more VRAM)
- facebook/wav2vec2-base (less accurate, requires less VRAM)
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from handlers.base import (
    get_cells_needing_text,
    get_cell_reference,
    get_cell_id,
    extract_tar_archive,
)
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Import inference API
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
    Execute ASR inference job.

    Args:
        job_context: Job configuration from GitLab manifest
        callbacks: JobCallbacks object for heartbeat, file operations, etc.

    Returns:
        Dictionary with:
            - success: bool
            - error_message: str (if success is False)
            - cells_transcribed: int
    """
    work_dir = callbacks.get_work_dir()

    print("=" * 60)
    print("W2V2-BERT ASR Inference Handler")
    print("=" * 60)

    try:
        # Extract job configuration
        # The manifest places most fields at the top level of the job dict.
        model_config = job_context.get('model', {})
        inference_config = job_context.get('inference', {})

        # Get model path - either from job output or specified path
        model_path = model_config.get('checkpoint_path')
        if not model_path:
            # Check if there's a trained model from a previous training job
            model_path = job_context.get('trained_model_path')

        if not model_path:
            return {
                'success': False,
                'error_message': "No model path specified. Set model.checkpoint_path in job config."
            }

        # Get model parameters
        use_wav2vec2_base = model_config.get('use_wav2vec2_base')

        # Get SentenceTransmorgrifier config (handler-specific)
        # Defaults to enabled if model_path is provided, can be explicitly disabled
        tm_config = job_context.get('transmorgrifier', {})
        tm_model_path = tm_config.get('model_path')
        # Default: enabled=True if model_path exists, but can be explicitly set to False
        use_transmorgrifier = tm_config.get('enabled', True) and tm_model_path is not None

        # Get filter configuration (per spec, inference filters under 'inference')
        include_verses = inference_config.get('include_verses', [])
        exclude_verses = inference_config.get('exclude_verses', [])

        # UNK token handling
        suppress_unk = inference_config.get('suppress_unk', True)
        unk_replacement = inference_config.get('unk_replacement', '')

        print("\nInference Configuration:")
        print(f"  Model path: {model_path}")
        print(f"  Use Wav2Vec2 base: {use_wav2vec2_base or 'Auto-detect'}")
        print(f"  SentenceTransmorgrifier: {tm_model_path or 'Disabled'}")
        print(f"  Suppress UNK tokens: {suppress_unk}")
        if include_verses:
            print(f"  Include verses: {len(include_verses)} specified")
        if exclude_verses:
            print(f"  Exclude verses: {len(exclude_verses)} specified")
        print()

        # Step 1: Download model from GitLab
        print("Step 1: Downloading ASR model from GitLab...")
        callbacks.heartbeat(message="Downloading ASR model", stage="download")

        model_dir = work_dir / "model"
        model_result = _download_model(
            job_context=job_context,
            callbacks=callbacks,
            model_path=model_path,
            output_dir=model_dir
        )

        if not model_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download model: {model_result['error_message']}"
            }

        local_model_path = model_result['local_path']
        print(f"  Downloaded model to: {local_model_path}")

        # Step 2: Load SentenceTransmorgrifier if configured
        transmorgrifier = None
        if use_transmorgrifier and tm_model_path:
            print("\nStep 2: Loading SentenceTransmorgrifier...")
            callbacks.heartbeat(message="Loading SentenceTransmorgrifier", stage="download")

            if not TRANSMORGRIFIER_AVAILABLE:
                print("  Warning: SentenceTransmorgrifier not installed, skipping post-processing")
                print("  Install with: pip install sentence-transmorgrifier")
            else:
                tm_result = _download_transmorgrifier(
                    job_context=job_context,
                    callbacks=callbacks,
                    tm_path=tm_model_path,
                    output_dir=work_dir / "transmorgrifier"
                )

                if tm_result['success']:
                    transmorgrifier = Transmorgrifier()
                    transmorgrifier.load(tm_result['local_path'])
                    print(f"  Loaded SentenceTransmorgrifier from: {tm_result['local_path']}")
                else:
                    print(f"  Warning: Failed to load SentenceTransmorgrifier: {tm_result['error_message']}")
                    print("  Continuing without post-processing")
        else:
            print("\nStep 2: SentenceTransmorgrifier not configured, skipping")

        # Step 3: Load ASR model
        print("\nStep 3: Loading ASR model...")
        callbacks.heartbeat(message="Loading ASR model", stage="inference")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor, detected_wav2vec2_base = load_model_and_processor(
            local_model_path,
            device,
            use_wav2vec2_base
        )

        model_arch = "Wav2Vec2" if detected_wav2vec2_base else "Wav2Vec2-BERT"
        print(f"  Loaded {model_arch} model on {device}")

        # Step 4: Download audio and process cells
        print("\nStep 4: Processing cells needing transcription...")
        callbacks.heartbeat(message="Processing cells", stage="inference")

        data_dir = work_dir / "data"
        audio_dir = data_dir / "audio"
        data_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Process each .codex file
        project_id = job_context['project_id']
        scanner = callbacks.scanner
        codex_files = job_context.get('codex_files', [])

        if not codex_files:
            return {
                'success': False,
                'error_message': "No codex_files specified in job configuration"
            }

        # Create downloader for audio file operations
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(project_id),
        )

        # List repository files and build audio files map to get correct file extensions
        items = downloader.list_repository_tree()
        audio_files_map = downloader.build_audio_files_map(items)
        print(f"  Found {len(audio_files_map)} audio files in repository")

        total_transcribed = 0
        total_errors = 0
        modified_codex_files = []

        for codex_path in codex_files:
            print(f"\n  Processing: {codex_path}")

            # Download .codex file
            codex_content = scanner.get_file_content(project_id, codex_path)
            if not codex_content:
                print(f"    Warning: Could not download {codex_path}")
                continue

            codex_data = json.loads(codex_content)
            cells = codex_data.get('cells', [])

            # Find cells needing transcription (have audio but no text)
            cells_to_process = get_cells_needing_text(
                cells,
                include_verses=include_verses if include_verses else None,
                exclude_verses=exclude_verses if exclude_verses else None
            )

            print(f"    Found {len(cells_to_process)} cells needing transcription")

            if not cells_to_process:
                continue

            codex_modified = False

            for idx, cell in enumerate(cells_to_process):
                cell_id = get_cell_id(cell)
                cell_ref = get_cell_reference(cell) or cell_id

                # Get audio info
                metadata = cell.get('metadata', {})
                selected_audio_id = metadata.get('selectedAudioId')
                attachments = metadata.get('attachments', {})

                if not selected_audio_id or selected_audio_id not in attachments:
                    continue

                # Check isDeleted/isMissing flags
                audio_info = attachments[selected_audio_id]
                if audio_info.get('isDeleted') or audio_info.get('isMissing'):
                    print(f"    Skipping {cell_ref}: audio is deleted or missing")
                    continue

                # Look up the actual file path from audio_files_map
                if selected_audio_id not in audio_files_map:
                    print(f"    Warning: Audio file not found in repository for {cell_ref}")
                    total_errors += 1
                    continue

                audio_repo_path = audio_files_map[selected_audio_id]
                audio_filename = os.path.basename(audio_repo_path)
                local_audio_path = audio_dir / audio_filename

                if not local_audio_path.exists():
                    if not downloader.download_file(audio_repo_path, local_audio_path):
                        print(f"    Warning: Could not download audio for {cell_ref}")
                        total_errors += 1
                        continue

                # Transcribe audio
                result = transcribe_audio(
                    str(local_audio_path),
                    model,
                    processor,
                    detected_wav2vec2_base,
                    device,
                    return_confidence=True,
                    unk_token_replacement=unk_replacement if suppress_unk else None,
                )

                if result.get('error'):
                    print(f"    Error transcribing {cell_ref}: {result['error']}")
                    total_errors += 1
                    continue

                transcription = result['transcription']

                # Post-process with SentenceTransmorgrifier if available
                if transmorgrifier and transcription:
                    try:
                        processed = list(transmorgrifier.execute([transcription]))[0]
                        print(f"    {cell_ref}: '{transcription}' -> '{processed}'")
                        transcription = processed
                    except Exception as e:
                        print(f"    Warning: Transmorgrifier failed for {cell_ref}: {e}")

                # Update cell value in the original cells list
                # Find the cell in the original list and update it
                for original_cell in cells:
                    if get_cell_id(original_cell) == cell_id:
                        original_cell['value'] = transcription
                        codex_modified = True
                        total_transcribed += 1
                        break

                # Heartbeat every 10 cells
                if (idx + 1) % 10 == 0:
                    callbacks.heartbeat(
                        message=f"Transcribed {total_transcribed} cells",
                        stage="inference"
                    )

            # Save modified .codex file
            if codex_modified:
                codex_data['cells'] = cells
                modified_codex_files.append({
                    'path': codex_path,
                    'content': json.dumps(codex_data, ensure_ascii=False, indent=2)
                })

        print(f"\n  Total cells transcribed: {total_transcribed}")
        print(f"  Total errors: {total_errors}")

        # Step 5: Upload modified .codex files in a single commit
        if modified_codex_files:
            print("\nStep 5: Uploading modified .codex files...")
            callbacks.heartbeat(message="Uploading results", stage="upload")

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
                    print(f"  Updated: {f['remote_path']}")
            else:
                print(f"  Error uploading codex files: {result.get('error_message')}")
                for f in result.get('files_failed', []):
                    print(f"    Failed: {f['remote_path']}: {f['error']}")
                    total_errors += 1
        else:
            print("\nStep 5: No files to upload (no cells were transcribed)")

        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'cells_transcribed': total_transcribed,
            'errors_count': total_errors,
            'files_modified': len(modified_codex_files)
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\nInference failed with error: {error_msg}")
        return {
            'success': False,
            'error_message': error_msg
        }


def _download_model(
    job_context: Dict[str, Any],
    callbacks,
    model_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download ASR model from GitLab.

    Supports both:
    - .pth.tar archives (new format): Downloads and extracts the archive
    - Directory format (legacy): Downloads individual model files

    Returns:
        Dictionary with success, local_path, error_message
    """
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if model_path is a .pth.tar archive
        if model_path.endswith('.pth.tar') or model_path.endswith('.tar.gz') or model_path.endswith('.tar'):
            print(f"    Downloading model archive: {model_path}")
            content = scanner.get_file_content(project_id, model_path, binary=True)
            if content:
                # Extract the archive
                print("    Extracting model archive...")
                extract_tar_archive(content, output_dir)
                return {
                    'success': True,
                    'local_path': str(output_dir),
                    'error_message': None
                }
            else:
                return {
                    'success': False,
                    'local_path': None,
                    'error_message': f"Could not download model archive from {model_path}"
                }

        # Legacy: Try to download as a directory of individual files
        model_files = [
            'config.json',
            'vocab.json',
            'preprocessor_config.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'model.safetensors',
            'pytorch_model.bin',
        ]

        downloaded_any = False
        for filename in model_files:
            remote_path = f"{model_path}/{filename}"
            local_path = output_dir / filename

            content = scanner.get_file_content(project_id, remote_path, binary=True)
            if content:
                with open(local_path, 'wb') as f:
                    f.write(content)
                downloaded_any = True

        if not downloaded_any:
            # Try downloading as a single file (e.g., a .pth.tar without the extension in path)
            content = scanner.get_file_content(project_id, model_path, binary=True)
            if content:
                # Check if it's actually a tar archive
                if content[:2] == b'\x1f\x8b':  # gzip magic number
                    print("    Detected gzip archive, extracting...")
                    extract_tar_archive(content, output_dir)
                    return {
                        'success': True,
                        'local_path': str(output_dir),
                        'error_message': None
                    }
                else:
                    local_path = output_dir / os.path.basename(model_path)
                    with open(local_path, 'wb') as f:
                        f.write(content)
                    return {
                        'success': True,
                        'local_path': str(local_path),
                        'error_message': None
                    }

            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download model from {model_path}"
            }

        return {
            'success': True,
            'local_path': str(output_dir),
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'local_path': None,
            'error_message': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


def _download_transmorgrifier(
    job_context: Dict[str, Any],
    callbacks,
    tm_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download SentenceTransmorgrifier model from GitLab.

    Returns:
        Dictionary with success, local_path, error_message
    """
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the transmorgrifier model file
        content = scanner.get_file_content(project_id, tm_path, binary=True)
        if not content:
            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download transmorgrifier from {tm_path}"
            }

        local_path = output_dir / os.path.basename(tm_path)
        with open(local_path, 'wb') as f:
            f.write(content)

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
