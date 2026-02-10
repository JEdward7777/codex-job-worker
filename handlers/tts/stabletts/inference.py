"""
StableTTS Inference Handler

Handles TTS inference jobs for the StableTTS model.
Downloads text data from GitLab, generates audio using the model,
and uploads the audio files back to GitLab.
"""

import os
import sys
import csv
import json
import traceback
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import all dependencies at module load time to fail early
from handlers.base import (
    get_cells_needing_audio,
    get_cell_reference,
    get_cell_id,
    cell_has_text,
    should_use_lfs,
    resolve_use_uroman,
)
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Import inference API
from inference_stabletts import inference_stabletts_api, StableTTSInference


def run(job_context: Dict[str, Any], callbacks) -> Dict[str, Any]:
    """
    Execute TTS inference job.

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
    print("StableTTS Inference Handler")
    print("=" * 60)

    try:
        # Extract job configuration
        # The manifest places most fields at the top level of the job dict.
        model_config = job_context.get('model', {})
        inference_config = job_context.get('inference', {})

        # Get model parameters
        model_type = model_config.get('type', 'StableTTS')
        checkpoint_path = model_config.get('checkpoint')
        reference_audio_path = model_config.get('reference_audio') or job_context.get('voice_reference')
        language = model_config.get('language', 'english')
        # use_uroman is resolved after data download (may need auto-detection)
        uroman_lang = model_config.get('uroman_lang', None)

        # Get inference parameters
        diffusion_steps = inference_config.get('diffusion_steps', 10)
        temperature = inference_config.get('temperature', 1.0)
        length_scale = inference_config.get('length_scale', 1.0)
        cfg_scale = inference_config.get('cfg_scale', 3.0)
        audio_format = inference_config.get('audio_format', 'webm')

        # Get filter configuration (per spec, inference filters under 'inference')
        include_verses = inference_config.get('include_verses', [])
        exclude_verses = inference_config.get('exclude_verses', [])

        print("\nInference Configuration:")
        print(f"  Model type: {model_type}")
        print(f"  Language: {language}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Reference audio: {reference_audio_path}")
        print(f"  Diffusion steps: {diffusion_steps}")
        print(f"  Temperature: {temperature}")
        print(f"  Length scale: {length_scale}")
        print(f"  CFG scale: {cfg_scale}")
        print(f"  Audio format: {audio_format}")
        print(f"  Use uroman: {model_config.get('use_uroman', '<auto-detect>')}")
        if include_verses:
            print(f"  Include verses: {len(include_verses)} specified")
        if exclude_verses:
            print(f"  Exclude verses: {len(exclude_verses)} specified")
        print()

        # Validate required parameters
        if not checkpoint_path:
            return {
                'success': False,
                'error_message': "No checkpoint specified in model configuration"
            }

        if not reference_audio_path:
            return {
                'success': False,
                'error_message': "No reference_audio specified in model configuration"
            }

        # Step 1: Download checkpoint
        print("Step 1: Downloading model checkpoint...")
        callbacks.heartbeat(message="Downloading checkpoint")

        checkpoint_dir = work_dir / "checkpoint"
        checkpoint_result = _download_file(
            job_context=job_context,
            callbacks=callbacks,
            remote_path=checkpoint_path,
            output_dir=checkpoint_dir
        )

        if not checkpoint_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download checkpoint: {checkpoint_result['error_message']}"
            }

        local_checkpoint = checkpoint_result['local_path']
        print(f"  Downloaded checkpoint to: {local_checkpoint}")

        # Step 2: Download reference audio
        print("\nStep 2: Downloading reference audio...")
        callbacks.heartbeat(message="Downloading reference audio")

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

        # Step 3: Download text data from GitLab
        print("\nStep 3: Downloading text data from GitLab...")
        callbacks.heartbeat(message="Downloading text data")

        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        download_result = _download_inference_data(
            job_context=job_context,
            callbacks=callbacks,
            output_dir=data_dir,
            include_verses=include_verses,
            exclude_verses=exclude_verses
        )

        if not download_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to download text data: {download_result['error_message']}"
            }

        metadata_csv = download_result['metadata_csv']
        sample_count = download_result['sample_count']
        codex_data = download_result['codex_data']

        print(f"  Found {sample_count} cells needing audio")

        if sample_count == 0:
            print("  No cells need audio generation. Job complete.")
            return {
                'success': True,
                'error_message': None,
                'processed_count': 0
            }

        # Resolve use_uroman (auto-detect from text if not specified in manifest)
        use_uroman = resolve_use_uroman(model_config, download_result.get('sample_texts', []))
        print(f"  Use uroman: {use_uroman}")

        # Step 4: Run inference
        print("\nStep 4: Running TTS inference...")
        callbacks.heartbeat(message="Running inference")

        output_dir = work_dir / "output"

        inference_result = inference_stabletts_api(
            checkpoint=local_checkpoint,
            input_csv=str(metadata_csv),
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
            heartbeat_callback=lambda: callbacks.heartbeat(message="Running inference")
        )

        if not inference_result['success']:
            return {
                'success': False,
                'error_message': f"Inference failed: {inference_result['error_message']}"
            }

        print(f"  Generated {inference_result['processed_count']} audio files")

        # Step 5: Upload audio files to GitLab and update .codex files
        print("\nStep 5: Uploading audio files and updating .codex files...")
        callbacks.heartbeat(message="Uploading results")

        upload_result = _upload_audio_and_update_codex(
            job_context=job_context,
            callbacks=callbacks,
            output_dir=output_dir,
            metadata_csv=inference_result['metadata_csv'],
            codex_data=codex_data,
            audio_format=audio_format
        )

        if not upload_result['success']:
            return {
                'success': False,
                'error_message': f"Failed to upload results: {upload_result['error_message']}"
            }

        print(f"  Uploaded {upload_result['uploaded_count']} audio files")

        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print("=" * 60)

        return {
            'success': True,
            'error_message': None,
            'processed_count': inference_result['processed_count'],
            'uploaded_count': upload_result['uploaded_count']
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\nInference failed with error: {error_msg}")
        return {
            'success': False,
            'error_message': error_msg
        }


def _download_file(
    job_context: Dict[str, Any],
    callbacks,
    remote_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Download a file from GitLab.

    Returns:
        Dictionary with success, local_path, error_message
    """
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


def _download_inference_data(
    job_context: Dict[str, Any],
    callbacks,
    output_dir: Path,
    include_verses: list,
    exclude_verses: list
) -> Dict[str, Any]:
    """
    Download text data for inference from GitLab.

    Returns:
        Dictionary with success, metadata_csv, sample_count, codex_data, error_message
    """
    try:
        project_id = job_context['project_id']
        scanner = callbacks.scanner

        codex_files = job_context.get('codex_files', [])

        if not codex_files:
            return {
                'success': False,
                'metadata_csv': None,
                'sample_count': 0,
                'codex_data': {},
                'error_message': "No codex_files specified in job configuration"
            }

        # Create metadata CSV
        metadata_csv = output_dir / "metadata.csv"
        samples = []
        codex_data = {}  # Store codex data for later update

        for codex_path in codex_files:
            print(f"  Processing: {codex_path}")

            # Download .codex file
            codex_content = scanner.get_file_content(project_id, codex_path)
            if not codex_content:
                print(f"    Warning: Could not download {codex_path}")
                continue

            codex_json = json.loads(codex_content)
            codex_data[codex_path] = codex_json
            cells = codex_json.get('cells', [])

            # Filter cells needing audio
            inference_cells = get_cells_needing_audio(
                cells,
                include_verses=include_verses if include_verses else None,
                exclude_verses=exclude_verses if exclude_verses else None
            )

            print(f"    Found {len(inference_cells)} cells needing audio")

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

            callbacks.heartbeat(message=f"Found {len(samples)} cells needing audio")

        # Write metadata CSV
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['verse_id', 'cell_id', 'transcription', 'codex_path'])
            writer.writeheader()
            writer.writerows(samples)

        return {
            'success': True,
            'metadata_csv': metadata_csv,
            'sample_count': len(samples),
            'sample_texts': [s['transcription'] for s in samples],
            'codex_data': codex_data,
            'error_message': None
        }

    except Exception as e:
        return {
            'success': False,
            'metadata_csv': None,
            'sample_count': 0,
            'sample_texts': [],
            'codex_data': {},
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
    """
    Upload generated audio files to GitLab (LFS) and update .codex files (regular git)
    in a single commit.

    Returns:
        Dictionary with success, uploaded_count, error_message
    """
    try:
        job_id = job_context['job_id']

        # Read output metadata
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Collect all files to upload
        files_to_upload = []
        codex_updates = {}  # Track updates per codex file

        for row in rows:
            cell_id = row.get('cell_id')
            codex_path = row.get('codex_path')
            audio_filename = row.get('file_name', '')

            if not audio_filename:
                continue

            # Get local audio file path
            local_audio_path = Path(output_dir) / audio_filename
            if not local_audio_path.exists():
                print(f"    Warning: Audio file not found: {local_audio_path}")
                continue

            # Generate audio ID for GitLab
            audio_id = f"audio-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            remote_audio_path = f".project/audio/{audio_id}.{audio_format}"

            # Add audio file to upload list (LFS for audio)
            files_to_upload.append({
                'local_path': str(local_audio_path),
                'remote_path': remote_audio_path,
                'lfs': True,  # Audio files always use LFS
            })

            # Track codex update
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

                # Find and update the cell
                for cell in cells:
                    if get_cell_id(cell) == cell_id:
                        metadata = cell.setdefault('metadata', {})
                        attachments = metadata.setdefault('attachments', {})

                        # Add audio attachment
                        attachments[audio_id] = {
                            'type': 'audio',
                            'format': fmt,
                            'isGenerated': True
                        }

                        # Set as selected audio
                        metadata['selectedAudioId'] = audio_id
                        break

            # Add updated codex file to upload list (regular git, not LFS)
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

        # Create uploader instance using callbacks credentials
        uploader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=callbacks.scanner.server_url,
            access_token=callbacks.scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        # Upload all files in a single commit
        callbacks.heartbeat(message=f"Uploading {len(files_to_upload)} files")
        result = uploader.upload_batch(
            files=files_to_upload,
            commit_message=f"TTS inference results for job {job_id}"
        )

        # Count audio files uploaded (exclude codex files)
        audio_count = sum(1 for f in result.get('files_uploaded', []) if f.get('lfs', False))

        if result['success']:
            return {
                'success': True,
                'uploaded_count': audio_count,
                'error_message': None
            }
        else:
            # Partial success - report what was uploaded
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
