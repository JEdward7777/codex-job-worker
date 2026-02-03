# GPU Worker for TTS and ASR Jobs

A GPU worker system that processes Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) jobs from GitLab-hosted Bible translation projects. The worker claims jobs from a manifest file, executes training or inference tasks using dynamically loaded handlers, and uploads results back to GitLab.

## Overview

This project provides a **unified GPU worker** ([`worker_entry.py`](worker_entry.py)) that:

1. **Claims jobs** from GitLab repositories via a YAML manifest system
2. **Dynamically loads handlers** based on job type, model type, and mode
3. **Executes training/inference** using the appropriate handler
4. **Uploads results** back to GitLab (models, audio files, `.codex` updates)

### Supported Job Types

- **TTS Training**: Train StableTTS models on audio-text pairs from Bible translation projects
- **TTS Inference**: Generate speech audio for verses that have text but no recordings
- **ASR Training**: Train Wav2Vec2-BERT models for speech recognition
- **ASR Inference**: Transcribe audio recordings to text

The system is designed to run on GPU workers launched via [SkyPilot](https://skypilot.readthedocs.io/), with jobs coordinated through a GitLab-based manifest system.

### Codex Editor Integration

Jobs are created through a **Codex Editor plugin** that provides a user-friendly interface for:
- Creating TTS/ASR training and inference jobs
- Selecting voice references and model checkpoints
- Monitoring job progress and results

> **Download Codex Editor**: [codexeditor.app](https://codexeditor.app/)

The plugin generates the `gpu_jobs/manifest.yaml` file that this worker consumes. See [Job Manifest System](#job-manifest-system) for details on the manifest format.

## Architecture

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

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training/inference)
- GitLab access token with `read_api`, `read_repository`, and `write_repository` scopes

### Installation

```bash
# Clone the repository with submodules
git clone --recurse-submodules <repository-url>
cd audio_text_tests

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies using uv
uv sync

# Apply patches to submodules (required!)
./apply_submodule_patches.sh
```

### Submodule Patches

This project uses git submodules for StableTTS and finetune-hf-vits. Local patches are required to fix compatibility issues with newer library versions. The [`apply_submodule_patches.sh`](apply_submodule_patches.sh) script applies these patches automatically.

**Patches applied:**
- **finetune-hf-vits**: Removes deprecated `send_example_telemetry` import (removed in newer transformers versions)
- **finetune-hf-vits**: Fixes matplotlib `canvas.tostring_rgb()` → `canvas.buffer_rgba()` API change
- **StableTTS**: Fixes relative path resolution for mel spectrograms in dataset loading

If you need to reset and reapply patches:

```bash
# Reset submodules to clean state
cd finetune-hf-vits && git reset --hard HEAD && cd ..
cd StableTTS && git reset --hard HEAD && cd ..

# Reapply patches
./apply_submodule_patches.sh
```

For more information on working with git submodules and patches, see [gist.github.com](https://gist.github.com/marc-hanheide/77d2685ceb2aaa4b90324c520dd4c34c).

## Deployment with SkyPilot

The recommended way to deploy the worker is using [SkyPilot](https://skypilot.readthedocs.io/), which provides a unified interface for launching GPU instances across multiple cloud providers.

### Supported Cloud Providers

- AWS, GCP, Azure
- Lambda Labs
- RunPod
- Vast.ai ([announced August 2025](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot))
- Kubernetes

### Quick Start with SkyPilot

```bash
# Install SkyPilot
pip install "skypilot[aws,gcp,azure,lambda,runpod,vast]"

# Configure your cloud credentials (see SkyPilot docs)
sky check

# Set your secrets as environment variables
export GITLAB_TOKEN=your_gitlab_token
export WORKER_ID=gpu-worker-1

# Launch the worker on Vast.ai (or change cloud in skypilot_worker.yaml)
sky launch skypilot_worker.yaml --env GITLAB_TOKEN --env WORKER_ID
```

### Secret Management

**Important**: Never commit secrets to the repository. Use one of these methods:

1. **Environment variables** (recommended):
   ```bash
   export GITLAB_TOKEN=your_token
   export WORKER_ID=your_worker_id
   sky launch skypilot_worker.yaml --env GITLAB_TOKEN --env WORKER_ID
   ```

2. **Env file** (not committed to repo):
   ```bash
   # Create .env.secrets file (already in .gitignore)
   echo "GITLAB_TOKEN=your_token" > .env.secrets
   echo "WORKER_ID=your_worker_id" >> .env.secrets

   sky launch skypilot_worker.yaml --env-file .env.secrets
   ```

3. **Cloud secret managers** (for production):
   - AWS Secrets Manager
   - GCP Secret Manager
   - Azure Key Vault

For more details on SkyPilot environment variables and secrets, see [docs.skypilot.co](https://docs.skypilot.co/en/stable/running-jobs/environment-variables.html).

### Running the Worker (Manual)

```bash
# Basic usage
python worker_entry.py --token YOUR_GITLAB_TOKEN --worker-id gpu-worker-1

# With continuous polling (every 5 minutes)
python worker_entry.py --token YOUR_TOKEN --worker-id gpu-worker-1 --loop-interval 300

# Using environment variables
export GITLAB_TOKEN=your_token
export WORKER_ID=gpu-worker-1
python worker_entry.py
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--token` | `$GITLAB_TOKEN` | GitLab access token |
| `--worker-id` | `$WORKER_ID` | Unique worker identifier |
| `--gitlab-url` | `https://git.genesisrnd.com` | GitLab server URL |
| `--work-dir` | `./work` | Base directory for job work files |
| `--loop-interval` | `-1` | Seconds between job checks (-1 = exit when done) |
| `--keep-jobs` | `5` | Number of completed job directories to keep |
| `--verbose` | `false` | Enable verbose output |

## Job Manifest System

Jobs are defined in a YAML manifest file at `gpu_jobs/manifest.yaml` in each GitLab project. The manifest is created by a **Codex Editor plugin** that provides a UI for creating and managing GPU jobs.

> **Note**: The Codex Editor plugin that creates these manifest files is available separately. Download Codex Editor from [codexeditor.app](https://codexeditor.app/).

### Manifest Format

```yaml
version: 1
jobs:
  - job_id: "abc123"
    job_type: tts              # 'tts' or 'asr'
    mode: training             # 'training', 'inference', or 'training_and_inference'
    model:
      type: StableTTS          # 'StableTTS' or 'W2V2BERT'
      base_checkpoint: gpu_jobs/job_xyz/models/best_model.pt  # Optional
    epochs: 1000               # For training modes
    inference:
      include_verses:          # Optional list of verse IDs/references
        - "MAT 1:1"
      exclude_verses: []       # Optional list to exclude
    voice_reference: "files/audio/speaker_sample.webm"  # For TTS inference
    audio_format: webm         # Output format: wav, ogg, mp3, webm
    timeout: "2027-01-01T00:00:00Z"
    canceled: false
```

### Job States

| State | Condition |
|-------|-----------|
| pending | No job folder exists |
| running | Job folder exists with worker ID |
| completed | `response.yaml` says completed |
| failed | `response.yaml` says failed |
| canceled | Manifest canceled + response says canceled |

## Dynamic Handler System

Handlers are loaded dynamically based on `job_type`, `model.type`, and `mode`:

```
handlers/
├── __init__.py
├── base.py                  # Shared utilities
├── tts/
│   ├── __init__.py
│   └── stabletts/
│       ├── __init__.py
│       ├── training.py
│       ├── inference.py
│       └── training_and_inference.py
└── asr/
    ├── __init__.py
    └── w2v2bert/
        ├── __init__.py
        ├── training.py
        ├── inference.py
        └── training_and_inference.py
```

### Handler Interface

Each handler module must implement a `run()` function:

```python
def run(job_context: dict, callbacks: JobCallbacks) -> dict:
    """
    Execute the job.

    Args:
        job_context: Job definition from manifest
        callbacks: Callbacks object for heartbeat, file operations, etc.

    Returns:
        dict with 'success': bool and optional 'error_message': str
    """
```

### Adding New Handlers

To add support for a new model type:

1. Create a new directory under `handlers/{job_type}/{model_name}/`
2. Implement `training.py`, `inference.py`, and/or `training_and_inference.py`
3. Each module must have a `run(job_context, callbacks)` function

The worker will automatically discover and load handlers based on the job configuration.

## Callbacks API

The `JobCallbacks` object provides handlers with:

- **`heartbeat(epochs_completed, message)`**: Update progress and check for cancellation
- **`get_work_dir()`**: Get the job-specific work directory
- **`get_job_config(key, default)`**: Get configuration values
- **`mark_complete(result_data)`**: Mark job as completed
- **`mark_failed(error_message)`**: Mark job as failed
- **`scanner`**: Access to GitLab API for file operations

## Project Structure

```
audio_text_tests/
├── worker_entry.py              # Main entry point for GPU worker
├── gitlab_jobs.py               # Job claiming and status updates
├── gitlab_to_hf_dataset.py      # Download/upload audio/text from GitLab
├── handlers/                    # Dynamic job handlers
│   ├── base.py                  # Shared utilities
│   ├── tts/stabletts/           # StableTTS handlers
│   └── asr/w2v2bert/            # Wav2Vec2-BERT handlers
├── train_stabletts.py           # StableTTS training script
├── inference_stabletts.py       # StableTTS inference script
├── train_w2v2bert_asr.py        # ASR training script
├── inference_w2v2bert_asr.py    # ASR inference script
├── preprocess_stabletts.py      # Data preprocessing for TTS
├── preprocess_audio.py          # Audio preprocessing (silence trimming)
├── StableTTS/                   # StableTTS model (submodule)
└── finetune-hf-vits/            # VITS fine-tuning (submodule)
```

## Standalone Tools

In addition to the worker system, this project includes standalone tools that can be used independently:

### GitLab to HuggingFace Dataset Converter

Download audio files and transcriptions from GitLab and prepare them in HuggingFace AudioFolder format:

```bash
# Explore repository structure
uv run python gitlab_to_hf_dataset.py explore --config_path=config.yaml

# Process and download dataset
uv run python gitlab_to_hf_dataset.py process --config_path=config.yaml
```

### Gradio Demos

Interactive demos for testing models:

```bash
# StableTTS demo
uv run python gradio_demo_stabletts.py

# ASR demo
uv run python gradio_demo_w2v2bert_asr.py
```

### Batch TTS Inference

Generate audio for multiple texts:

```bash
uv run python batch_tts_inference.py --input texts.csv --output audio/ --checkpoint model.pt
```

## Configuration

### config.yaml (for standalone tools)

```yaml
gitlab:
  server_url: "https://git.genesisrnd.com"
  access_token: "your-token-here"
  project_id: "eten/config/config-fxrmlox2kyrrr3fwupgsq"

dataset:
  output_dir: "huggingface_dataset"
  audio_dir: "audio"
  csv_filename: "metadata.csv"
  max_records: 10  # Set to 0 or null for all records
```

> **Note**: All `.yaml` files are ignored by git. Copy `config.yaml.example` to `config.yaml` and fill in your credentials.

## Related Documentation

- [WORKER_IMPLEMENTATION_PLAN.md](WORKER_IMPLEMENTATION_PLAN.md) - Detailed architecture and implementation plan
- [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md) - Manual training instructions
- [TTS_FINETUNING_README.md](TTS_FINETUNING_README.md) - TTS fine-tuning guide
- [W2V2BERT_ASR_README.md](W2V2BERT_ASR_README.md) - ASR training guide
- [GITLAB_JOBS_README.md](GITLAB_JOBS_README.md) - GitLab job scanner documentation

## How to Create a GitLab Access Token

1. Go to your GitLab instance (e.g., https://git.genesisrnd.com)
2. Click on your profile picture (top right) → **Edit Profile**
3. In the left sidebar, click **Access Tokens**
4. Click **Add new token**
5. Fill in the details:
   - **Token name**: Something descriptive like "GPU Worker"
   - **Expiration date**: Set as needed
   - **Select scopes**: Check `read_api`, `read_repository`, and `write_repository`
6. Click **Create personal access token**
7. **Important**: Copy the token immediately (you won't be able to see it again)

## License

This project uses components from:
- [StableTTS](https://github.com/KdaiP/StableTTS) - Apache 2.0 License
- [Parler-TTS](https://github.com/huggingface/parler-tts) - Apache 2.0 License ([github.com](https://github.com/huggingface/parler-tts))
- [Coqui TTS](https://github.com/coqui-ai/TTS) - MPL 2.0 License ([github.com](https://github.com/coqui-ai/TTS/tree/main))
- [Inworld TTS](https://github.com/inworld-ai/tts) - MIT License ([github.com](https://github.com/inworld-ai/tts))
