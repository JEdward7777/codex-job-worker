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
        checkpoint_path = model_config.get('base_checkpoint')
        reference_audio_path = job_context.get('voice_reference')
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

    Uses GitLabDatasetDownloader.download_file() which automatically handles
    Git LFS files (model checkpoints and audio are typically stored via LFS).

    Returns:
        Dictionary with success, local_path, error_message
    """
    try:
        scanner = callbacks.scanner

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(remote_path)
        local_path = output_dir / filename

        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=scanner.server_url,
            access_token=scanner.access_token,
            project_id=str(job_context['project_id']),
        )

        if not downloader.download_file(remote_path, local_path):
            return {
                'success': False,
                'local_path': None,
                'error_message': f"Could not download file from {remote_path}"
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

        # Create downloader for file operations (handles LFS transparently)
        downloader = GitLabDatasetDownloader(
            config_path=None,
            gitlab_url=scanner.server_url,
            access_token=scanner.access_token,
            project_id=str(project_id),
            config_overrides={
                'dataset.output_dir': str(output_dir),
            }
        )

        # Auto-discover .codex files from the repository
        print("  Listing repository files...")
        items = downloader.list_repository_tree()
        codex_files = [item['path'] for item in items if item['name'].endswith('.codex')]

        if not codex_files:
            return {
                'success': False,
                'metadata_csv': None,
                'sample_count': 0,
                'codex_data': {},
                'error_message': "No .codex files found in repository"
            }

        print(f"  Found {len(codex_files)} .codex files in repository")

        # Create metadata CSV
        metadata_csv = output_dir / "metadata.csv"
        samples = []
        codex_data = {}  # Store codex data for later update

        for codex_path in codex_files:
            print(f"  Processing: {codex_path}")

            # Download .codex file using downloader
            codex_json = downloader.download_json_file(codex_path)
            if not codex_json:
                print(f"    Warning: Could not download {codex_path}")
                continue
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


def _get_book_code(codex_path: str) -> str:
    """
    Extract the book code from a codex file path.

    Examples:
        'target/MAT.codex' -> 'MAT'
        'files/target/EXO.codex' -> 'EXO'
        'LUK.codex' -> 'LUK'
    """
    filename = os.path.basename(codex_path)
    return filename.rsplit('.', 1)[0]


def _get_audio_file_metadata(local_path: Path, audio_format: str) -> Dict[str, Any]:
    """
    Compute metadata for an audio file (size, duration, sample rate, etc.).

    Returns a metadata dict matching the codex attachment metadata schema:
        mimeType, sizeBytes, sampleRate, channels, durationSec, bitrateKbps
    """
    metadata = {}

    # File size is always available
    try:
        metadata['sizeBytes'] = local_path.stat().st_size
    except OSError:
        metadata['sizeBytes'] = 0

    # MIME type from format
    mime_map = {
        'webm': 'audio/webm',
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'ogg': 'audio/ogg',
        'flac': 'audio/flac',
    }
    metadata['mimeType'] = mime_map.get(audio_format, f'audio/{audio_format}')

    # Try to get audio properties using mutagen or ffprobe
    try:
        import mutagen
        audio = mutagen.File(str(local_path))
        if audio is not None:
            if hasattr(audio.info, 'sample_rate') and audio.info.sample_rate:
                metadata['sampleRate'] = audio.info.sample_rate
            if hasattr(audio.info, 'channels') and audio.info.channels:
                metadata['channels'] = audio.info.channels
            if hasattr(audio.info, 'length') and audio.info.length:
                metadata['durationSec'] = round(audio.info.length, 2)
            if hasattr(audio.info, 'bitrate') and audio.info.bitrate:
                metadata['bitrateKbps'] = round(audio.info.bitrate / 1000)
    except ImportError:
        pass
    except Exception:
        pass

    # If mutagen didn't get duration, try ffprobe as fallback
    if 'durationSec' not in metadata:
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', '-show_streams', str(local_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json as json_mod
                probe = json_mod.loads(result.stdout)
                fmt_info = probe.get('format', {})
                if 'duration' in fmt_info:
                    metadata['durationSec'] = round(float(fmt_info['duration']), 2)
                if 'bit_rate' in fmt_info and 'bitrateKbps' not in metadata:
                    metadata['bitrateKbps'] = round(int(fmt_info['bit_rate']) / 1000)
                # Check streams for sample rate and channels
                for stream in probe.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        if 'sample_rate' in stream and 'sampleRate' not in metadata:
                            metadata['sampleRate'] = int(stream['sample_rate'])
                        if 'channels' in stream and 'channels' not in metadata:
                            metadata['channels'] = int(stream['channels'])
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    return metadata


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

    Audio files are uploaded to .project/attachments/pointers/{BOOK}/ in git,
    and the codex url field references .project/attachments/files/{BOOK}/ which
    is where the codex environment maps them for local access.

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

            # Derive book code from codex path (e.g., 'target/EXO.codex' -> 'EXO')
            book_code = _get_book_code(codex_path)

            # Generate audio ID for GitLab
            audio_id = f"audio-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

            # Git upload path uses 'pointers' directory
            remote_audio_path = f".project/attachments/pointers/{book_code}/{audio_id}.{audio_format}"
            # Codex url field uses 'files' directory (where codex env maps them)
            codex_audio_url = f".project/attachments/files/{book_code}/{audio_id}.{audio_format}"

            # Compute audio file metadata
            audio_metadata = _get_audio_file_metadata(local_audio_path, audio_format)

            # Add audio file to upload list (LFS for audio)
            files_to_upload.append({
                'local_path': str(local_audio_path),
                'remote_path': remote_audio_path,
                'lfs': True,  # Audio files always use LFS
            })

            # Track codex update
            if codex_path not in codex_updates:
                codex_updates[codex_path] = []

            now_ms = int(time.time() * 1000)
            codex_updates[codex_path].append({
                'cell_id': cell_id,
                'audio_id': audio_id,
                'codex_audio_url': codex_audio_url,
                'audio_metadata': audio_metadata,
                'timestamp_ms': now_ms,
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
                codex_audio_url = update['codex_audio_url']
                audio_meta = update['audio_metadata']
                ts = update['timestamp_ms']

                # Find and update the cell
                for cell in cells:
                    if get_cell_id(cell) == cell_id:
                        metadata = cell.setdefault('metadata', {})
                        attachments = metadata.setdefault('attachments', {})

                        # Build attachment object matching codex schema
                        attachment = {
                            'url': codex_audio_url,
                            'type': 'audio',
                            'createdAt': ts,
                            'updatedAt': ts,
                            'isDeleted': False,
                            'isGenerated': True,
                        }

                        # Add audio metadata sub-object
                        if audio_meta:
                            attachment['metadata'] = audio_meta

                        attachments[audio_id] = attachment

                        # Set as selected audio with timestamp
                        metadata['selectedAudioId'] = audio_id
                        metadata['selectionTimestamp'] = ts
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
