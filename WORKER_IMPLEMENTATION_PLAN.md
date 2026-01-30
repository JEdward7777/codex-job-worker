# GPU Worker Implementation Plan

## Overview

This document describes the architecture and implementation plan for the GPU worker system that processes TTS and ASR jobs from GitLab-hosted Bible translation projects. The worker claims jobs from a manifest file, executes training or inference tasks, and uploads results back to GitLab.

---

## System Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              worker_entry.py                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │ Claim Job    │──▶│ Load Handler │──▶│ Execute Handler (with callbacks)│ │
│  │ (gitlab_jobs)│    │ (dynamic)    │    │                                 │ │
│  └──────────────┘    └──────────────┘    └─────────────────────────────────┘ │
│         │                                           │                        │
│         ▼                                           ▼                        │
│  ┌──────────────┐                          ┌─────────────────────────────┐   │
│  │ Update       │◀────────────────────────│ Upload Results              │   │
│  │ response.yaml│                          │ (models, audio, .codex)     │   │
│  └──────────────┘                          └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
audio_text_tests/
├── worker_entry.py              # Main entry point
├── gitlab_jobs.py               # Job claiming, status updates (existing)
├── gitlab_to_hf_dataset.py      # Download/upload audio/text (existing, to extend)
├── handlers/                    # Dynamic job handlers
│   ├── __init__.py
│   ├── base.py                  # Base handler utilities
│   ├── tts/
│   │   ├── __init__.py
│   │   └── stabletts/
│   │       ├── __init__.py
│   │       ├── training.py      # TTS training handler
│   │       └── inference.py     # TTS inference handler
│   └── asr/
│       ├── __init__.py
│       └── w2v2bert/
│           ├── __init__.py
│           ├── training.py      # ASR training handler
│           └── inference.py     # ASR inference handler
├── train_stabletts.py           # Existing (to refactor for Python API)
├── inference_stabletts.py       # Existing (to refactor for Python API)
├── train_w2v2bert_asr.py        # Existing (to refactor for Python API)
├── inference_w2v2bert_asr.py    # Existing (to refactor for Python API)
├── preprocess_stabletts.py      # Existing (to refactor for Python API)
└── preprocess_audio.py          # Existing (silence trimming)
```

---

## Job Manifest Format

Location: `gpu_jobs/manifest.yaml`

```yaml
version: 1
jobs:
  - job_id: "abc123"
    job_type: tts              # 'tts' or 'asr'
    mode: training             # 'training', 'inference', or 'training_and_inference'
    model:
      type: StableTTS          # 'StableTTS' or 'W2V2BERT' (case-insensitive)
      base_checkpoint: gpu_jobs/job_xyz/models/best_model.pt  # Optional, for inference/fine-tuning
    epochs: 1000               # For training modes
    inference:
      include_verses:          # Optional list of verse IDs/references
        - "MAT 1:1"
        - "5e7dadd4-9879-026e-82d9-80337922faad"
      exclude_verses: []       # Optional list to exclude
    voice_reference: "files/audio/speaker_sample.webm"  # For TTS inference
    audio_format: webm         # Output format: wav, ogg, mp3, webm (default: webm)
    timeout: "2027-01-01T00:00:00Z"  # Optional
    canceled: false
```

### Model Type to Handler Mapping

The handler is loaded dynamically based on `job_type` and `model.type`:

| job_type | model.type | Handler Path |
|----------|------------|--------------|
| tts | StableTTS | handlers/tts/stabletts/{mode}.py |
| asr | W2V2BERT | handlers/asr/w2v2bert/{mode}.py |

---

## Response File Format

Location: `gpu_jobs/job_{job_id}/response.yaml`

```yaml
worker_id: "gpu-worker-1"
state: running              # running, completed, failed, canceled
timestamp: "2026-01-25T22:00:00Z"
epochs_completed: 500       # Optional, for training
expected_completion: "2026-01-26T10:00:00Z"  # Optional estimate
error_message: null         # Set on failure
termination_reason: null    # Set on cancellation (e.g., "user_canceled", "job_removed")
```

---

## Worker Entry Point (worker_entry.py)

### Command-Line Interface

```bash
python worker_entry.py \
  --token YOUR_GITLAB_TOKEN \           # Or use GITLAB_TOKEN env var
  --worker-id gpu-worker-1 \            # Or use WORKER_ID env var
  --gitlab-url https://git.genesisrnd.com \
  --work-dir ./work \                   # Or use WORK_DIR env var
  --loop-interval 300 \                 # Seconds between job checks (-1 = exit when done)
  --keep-jobs 5                         # Number of completed job dirs to keep
```

### Main Loop Logic

```python
def main():
    while True:
        try:
            job = scanner.claim_next_job(worker_id)
            process_job(job)
        except NoJobsAvailableError:
            if loop_interval < 0:
                break  # Exit
            time.sleep(loop_interval)
```

### Job Processing Flow

1. **Claim job** - Creates response.yaml with worker_id
2. **Setup logging** - Redirect stdout/stderr to job log file
3. **Load handler** - Dynamic import based on job_type/model.type/mode
4. **Execute handler** - Pass job_context and callbacks
5. **Handle result** - Update response.yaml (completed/failed/canceled)
6. **Upload logs** - Push log file to GitLab
7. **Cleanup** - Manage work directory retention

---

## Callbacks Object

The callbacks object provides shared functionality to handlers:

```python
class JobCallbacks:
    def __init__(self, job_context, scanner, work_dir):
        self.job_context = job_context
        self.scanner = scanner
        self.work_dir = work_dir
        self._last_heartbeat = time.time()

    def heartbeat(self, epochs_completed=None, message=None):
        """
        Update response.yaml with timestamp and optional progress.
        Checks for cancellation - raises JobCanceledException if canceled.
        Called every 5 minutes automatically, or manually by handler.
        """
        # Check if job still exists and we still own it
        job_status = self.scanner.get_job_status(project_id, job_id)

        if not job_status:
            raise JobCanceledException("Job removed from manifest")

        if job_status.get('canceled'):
            raise JobCanceledException("Job canceled by user")

        response = job_status.get('response', {})
        if response.get('worker_id') != self.worker_id:
            raise JobCanceledException("Job claimed by another worker")

        # Update response.yaml with heartbeat
        self._update_response(epochs_completed=epochs_completed)

    def download_project_data(self, filter_config=None):
        """
        Download audio and text from GitLab project.
        Returns path to metadata.csv and audio directory.
        """
        pass

    def upload_file_lfs(self, local_path, remote_path):
        """Upload a file to GitLab LFS."""
        pass

    def upload_files_batch(self, file_mappings):
        """Upload multiple files in a single commit."""
        pass

    def update_codex_file(self, codex_path, modifications):
        """Update a .codex file with new audio/text."""
        pass

    def get_work_dir(self):
        """Get the job-specific work directory."""
        return self.work_dir / f"job_{self.job_context['job_id']}"
```

### JobCanceledException

```python
class JobCanceledException(Exception):
    """Raised when job is canceled. Handler can catch to cleanup."""
    pass
```

---

## Handler Interface

Each handler is a Python module with a `run` function:

```python
# handlers/tts/stabletts/training.py

def run(job_context: dict, callbacks: JobCallbacks) -> dict:
    """
    Execute TTS training job.

    Args:
        job_context: Job definition from manifest
        callbacks: Callbacks object for shared functionality

    Returns:
        dict with 'success': bool and optional 'error_message': str
    """
    try:
        # 1. Download training data
        data_dir = callbacks.download_project_data()

        # 2. Preprocess (silence trimming, mel spectrograms)
        preprocess_audio(data_dir, callbacks)
        preprocess_stabletts(data_dir, callbacks)

        # 3. Train model (with periodic heartbeat)
        train_config = build_train_config(job_context)
        train_stabletts_api(train_config, heartbeat_callback=callbacks.heartbeat)

        # 4. Upload model
        callbacks.upload_file_lfs(
            local_path=work_dir / "checkpoints/best_model.pt",
            remote_path=f"gpu_jobs/job_{job_id}/models/best_model.pt"
        )

        return {'success': True}

    except JobCanceledException:
        # Upload partial model if available
        if os.path.exists(work_dir / "checkpoints/best_model.pt"):
            callbacks.upload_file_lfs(...)
        raise  # Re-raise for worker to handle

    except Exception as e:
        return {'success': False, 'error_message': str(e)}
```

---

## Job Type Workflows

### TTS Training (handlers/tts/stabletts/training.py)

1. Download audio and text from project (filtered by include/exclude_verses if specified)
2. Run silence trimming (preprocess_audio.py with --trim_silence)
3. Generate mel spectrograms (preprocess_stabletts.py)
4. Train model (train_stabletts.py API)
   - Periodic heartbeat every 5 minutes
   - Save best_model.pt and final_model.pt
5. Upload best_model.pt to `gpu_jobs/job_{job_id}/models/`

### TTS Inference (handlers/tts/stabletts/inference.py)

1. Download text for verses without audio (or filtered verses)
2. Download base checkpoint from manifest
3. Get voice reference (from manifest or first audio file)
4. Run inference (inference_stabletts.py API)
5. Inject audio into .codex files:
   - Create new attachment entry
   - Set as selectedAudioId
   - Only for verses without existing audio (unless in include_verses)
6. Upload audio files to LFS
7. Commit .codex changes

### ASR Training (handlers/asr/w2v2bert/training.py)

1. Download audio and text from project
2. Run silence trimming
3. Train W2V2-BERT model (train_w2v2bert_asr.py API)
4. Run ASR inference on training data to get "from" text
5. Train SentenceTransmorgrifier with (ASR output → original text)
6. Upload both models to `gpu_jobs/job_{job_id}/models/`

### ASR Inference (handlers/asr/w2v2bert/inference.py)

1. Download audio for verses without text (or filtered verses)
2. Run silence trimming
3. Run W2V2-BERT inference (inference_w2v2bert_asr.py API)
4. Run SentenceTransmorgrifier on ASR output
5. Inject text into .codex files:
   - Set cell `value` field
   - Only for verses without existing text (unless in include_verses)
6. Commit .codex changes

### Training and Inference Combined

For `mode: training_and_inference`:
1. Run training workflow
2. Use newly trained model for inference workflow
3. Share downloaded data between steps (don't re-download)

---

## Refactoring Existing Scripts

### Pattern for Adding Python API

Each existing script needs a refactored entry point that can be called from Python:

```python
# train_stabletts.py

def train_stabletts_api(config: dict, heartbeat_callback=None) -> dict:
    """
    Train StableTTS model with Python API.

    Args:
        config: Dictionary with training parameters
        heartbeat_callback: Optional callback(epochs_completed) for progress

    Returns:
        dict with training results
    """
    # Extract config with defaults
    train_dataset_path = config.get('train_dataset_path')
    model_save_path = config.get('model_save_path')
    log_dir = config.get('log_dir')
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    num_epochs = config.get('num_epochs', 10000)
    # ... etc

    # Run training (existing logic)
    # Call heartbeat_callback periodically

    return {'best_model_path': ..., 'final_model_path': ...}


def main():
    """CLI entry point - unchanged."""
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()

    # Convert args to config dict
    config = vars(args)

    # Call API
    train_stabletts_api(config)


if __name__ == "__main__":
    main()
```

### Scripts to Refactor

1. **train_stabletts.py**
   - Add `train_stabletts_api(config, heartbeat_callback)`
   - Add final_model.pt saving at end of training
   - Keep CLI interface unchanged

2. **inference_stabletts.py**
   - Add `inference_stabletts_api(config)`
   - Return list of generated audio paths

3. **train_w2v2bert_asr.py**
   - Add `train_w2v2bert_api(config, heartbeat_callback)`
   - Keep CLI interface unchanged

4. **inference_w2v2bert_asr.py**
   - Add `inference_w2v2bert_api(config)`
   - Return transcription results

5. **preprocess_stabletts.py**
   - Add `preprocess_stabletts_api(config)`
   - Return path to generated JSON filelist

6. **preprocess_audio.py**
   - Add `preprocess_audio_api(config)`
   - Support silence trimming via config

---

## GitLab Integration Extensions

### gitlab_jobs.py Additions

```python
class GitLabJobScanner:
    # Existing methods...

    def update_file(self, project_id, file_path, content, commit_message):
        """Update an existing file in the repository."""
        pass

    def get_file_content(self, project_id, file_path):
        """Get file content (public wrapper for _get_file_content)."""
        pass
```

### gitlab_to_hf_dataset.py Additions

```python
class GitLabDatasetDownloader:
    # Existing methods...

    def upload_file_lfs(self, local_path, remote_path):
        """Upload a file to GitLab LFS storage."""
        # 1. Calculate SHA256 hash and size
        # 2. Call LFS batch API with operation="upload"
        # 3. Upload file content to LFS storage
        # 4. Create pointer file via GitLab API
        pass

    def upload_files_batch(self, file_mappings, commit_message):
        """Upload multiple files in a single commit."""
        # Use GitLab Commits API for text files
        # Use LFS batch API for large files
        pass

    def update_codex_file(self, codex_path, cell_id, modifications):
        """Update a specific cell in a .codex file."""
        # 1. Download current .codex content
        # 2. Parse JSON
        # 3. Find cell by ID
        # 4. Apply modifications
        # 5. Upload updated file
        pass
```

---

## Cell Format Compatibility

### Reading Cell ID

```python
def get_cell_id(cell: dict) -> str:
    """Get cell identifier, handling both old and new formats."""
    metadata = cell.get('metadata', {})

    # New format: globalReferences
    data = metadata.get('data', {})
    global_refs = data.get('globalReferences', [])
    if global_refs:
        return global_refs[0]

    # Old format: id field contains reference
    return metadata.get('id', '')
```

### Matching Filter Items

```python
def is_reference_format(item: str) -> bool:
    """Check if filter item is a Bible reference (contains colon)."""
    return ':' in item

def cell_matches_filter(cell: dict, filter_item: str) -> bool:
    """Check if cell matches a filter item."""
    if is_reference_format(filter_item):
        # Match against globalReferences or legacy id
        cell_ref = get_cell_reference(cell)
        return cell_ref == filter_item
    else:
        # Match against UUID
        return cell.get('metadata', {}).get('id') == filter_item
```

---

## Logging Setup

```python
class TeeLogger:
    """Write to both file and stdout."""

    def __init__(self, log_path):
        self.log_file = open(log_path, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.log_file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.log_file.flush()
        self.stdout.flush()


def setup_job_logging(job_id, work_dir):
    """Setup logging for a job."""
    log_path = work_dir / f"job_{job_id}" / "logs.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    return log_path
```

---

## Security Considerations

### Path Validation

```python
def validate_checkpoint_path(path: str) -> bool:
    """Validate that checkpoint path is safe."""
    # Must start with gpu_jobs/
    if not path.startswith('gpu_jobs/'):
        return False

    # Must not contain path traversal
    if '..' in path:
        return False

    return True
```

---

## Work Directory Management

```python
def cleanup_old_jobs(work_dir: Path, keep_count: int):
    """Keep only the N most recent job directories."""
    job_dirs = sorted(
        work_dir.glob("job_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for old_dir in job_dirs[keep_count:]:
        shutil.rmtree(old_dir)
```

Job directories are named with datetime for uniqueness:
```
work/
├── job_abc123_2026-01-25T22-30-00/
├── job_def456_2026-01-25T23-00-00/
└── job_ghi789_2026-01-26T00-30-00/
```

---

## Error Handling

### Job Failure

```python
try:
    result = handler.run(job_context, callbacks)
    if result.get('success'):
        update_response(state='completed')
    else:
        update_response(state='failed', error_message=result.get('error_message'))
except JobCanceledException as e:
    update_response(state='canceled', termination_reason=str(e))
except Exception as e:
    update_response(state='failed', error_message=str(e))
finally:
    upload_logs()
```

---

## Configuration Defaults

### Training Defaults

| Parameter | StableTTS Default | W2V2BERT Default |
|-----------|-------------------|------------------|
| batch_size | 8 | 4 |
| learning_rate | 1e-4 | 1e-4 |
| val_split | 0.1 | 0.1 |
| save_interval | 100 | 500 |

### Inference Defaults

| Parameter | Default |
|-----------|---------|
| audio_format | webm |
| language | english |
| diffusion_steps | 10 |
| temperature | 1.0 |

---

## Implementation Status

**Last Updated:** 2026-01-26

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Create worker_entry.py with CLI
- [x] Create callbacks object (JobCallbacks class)
- [x] Add update_file to gitlab_jobs.py
- [x] Setup logging infrastructure (TeeLogger)
- [x] Create handler directory structure

### Phase 2: TTS Training ✅ COMPLETE
- [x] Refactor train_stabletts.py for Python API (`train_stabletts_api()`)
- [x] Refactor preprocess_stabletts.py for Python API (`preprocess_stabletts_api()`)
- [x] Create handlers/tts/stabletts/training.py
- [x] Add LFS upload to gitlab_to_hf_dataset.py (`upload_batch()` method)

### Phase 3: TTS Inference ✅ COMPLETE
- [x] Refactor inference_stabletts.py for Python API (`inference_stabletts_api()`)
- [x] Create handlers/tts/stabletts/inference.py
- [x] Create handlers/tts/stabletts/training_and_inference.py
- [x] Handle audio injection into .codex files

### Phase 4: ASR Training ✅ COMPLETE
- [x] Refactor train_w2v2bert_asr.py for Python API (`train_w2v2bert_asr_api()`)
- [x] Support both Wav2Vec2-BERT and traditional Wav2Vec2 architectures
- [x] Create handlers/asr/w2v2bert/training.py

### Phase 5: ASR Inference ✅ COMPLETE
- [x] Refactor inference_w2v2bert_asr.py for Python API (`inference_w2v2bert_asr_api()`)
- [x] Create handlers/asr/w2v2bert/inference.py
- [x] Integrate SentenceTransmorgrifier for post-processing
- [x] Handle text injection into .codex files

### Phase 6: Combined Mode ✅ COMPLETE
- [x] Create handlers/asr/w2v2bert/training_and_inference.py
- [x] Implement bandwidth-saving local model usage
- [x] Add work directory cleanup in worker_entry.py

### Remaining TODO
- [x] Add GitLab LFS batch API support for large file uploads (implemented in `gitlab_to_hf_dataset.py`)
- [ ] Add GPU memory monitoring
- [ ] End-to-end testing with real GitLab project

---

## Testing Strategy

1. **Unit Tests**: Test individual components (path validation, cell matching, etc.)
2. **Integration Tests**: Test handler workflows with mock GitLab
3. **End-to-End Tests**: Test full job cycle with real GitLab project

---

## Open Questions / Future Considerations

1. Should we support resuming interrupted training jobs?
2. Should there be a way to prioritize certain jobs?
3. How should we handle very large projects with thousands of verses?
4. Should we add GPU memory monitoring to prevent OOM crashes?
