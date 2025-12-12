# MMS-TTS Fine-tuning Pipeline

Complete pipeline for fine-tuning MMS-TTS models on custom datasets for low-resource languages.

## Overview

This repository provides a complete, modular pipeline for fine-tuning Meta's Massively Multilingual Speech (MMS) Text-to-Speech models. The pipeline handles everything from data collection to model training, with each step producing immutable outputs that can be re-run independently.

### Pipeline Stages

```
1. Data Collection (gitlab_to_hf_dataset.py)
   ↓
2. Audio Preprocessing (preprocess_audio.py)
   ↓
3. Speaker Diarization (speaker_diarization.py)
   ↓
4. Dataset Preparation (prepare_tts_dataset.py)
   ↓
5. Model Training (train_mms_tts.py → finetune-hf-vits)
```

## Features

- **Modular Design**: Each step is independent and can be re-run without affecting previous steps
- **Immutable Outputs**: Each stage creates new output directories, preserving previous results
- **Config-Driven**: All parameters controlled via YAML configuration files
- **Multi-Format Support**: Handles webm, mp3, wav, flac, ogg, m4a, and more
- **Automatic Speaker Clustering**: Uses Resemblyzer for unsupervised speaker identification
- **Official Training**: Uses the official `finetune-hf-vits` repository for training
- **Generic + Specific**: Generic tools work for any language; language-specific configs in YAML

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd audio_text_tests

# Initialize submodules (for finetune-hf-vits)
git submodule update --init --recursive

# Install dependencies using uv (includes Cython for building monotonic alignment)
uv sync

# Build monotonic alignment for VITS training
# Note: This requires creating the monotonic_align subdirectory first
cd finetune-hf-vits/monotonic_align
mkdir -p monotonic_align
uv run python setup.py build_ext --inplace
cd ../..
```

### 2. Prepare Your Configuration

Copy and modify the example configuration for your language:

```bash
cp config.yaml your_language.yaml
# Edit your_language_tts.yaml with your paths and settings
```

### 3. Run the Pipeline

```bash
# Step 1: Collect data (if using GitLab/local repo)
python gitlab_to_hf_dataset.py process --config_path your_language.yaml

# Step 2: Preprocess audio
python preprocess_audio.py --config your_language_tts.yaml

# Step 3: Perform speaker diarization
python speaker_diarization.py --config your_language_tts.yaml

# Step 4: Prepare HuggingFace dataset
python prepare_tts_dataset.py --config your_language_tts.yaml

# Step 5: Train the model
python train_mms_tts.py --config your_language_tts.yaml
```

## Detailed Documentation

### Configuration Files

#### Language-Specific Config (e.g., `config.yaml`)

Contains all paths and parameters specific to your language/project:

```yaml
language:
  name: "config"
  code: "<three leter language code>"
  base_model: "facebook/mms-tts-cmn"

paths:
  raw_dataset: "/path/to/raw/data"
  preprocessed: "/path/to/preprocessed"
  speaker_analysis: "/path/to/speaker_analysis"
  hf_dataset: "/path/to/hf_dataset"
  checkpoints: "/path/to/checkpoints"

audio:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
  normalize: true

training:
  num_epochs: 250
  learning_rate: 1e-4
  per_device_train_batch_size: 8
  # ... more parameters
```

See [`config.yaml`](config.yaml:1) for a complete example.

### Pipeline Scripts

#### 1. Data Collection: `gitlab_to_hf_dataset.py`

Collects audio files and transcriptions from GitLab repositories or local directories.

**Input**: GitLab repo or local directory with CODEX files
**Output**: Raw audio files + `metadata.csv`

```bash
python gitlab_to_hf_dataset.py process --config_path your_language.yaml
```

**Output Structure**:
```
/path/to/raw/
├── audio/
│   ├── file1.webm
│   ├── file2.webm
│   └── ...
└── metadata.csv
```

#### 2. Audio Preprocessing: `preprocess_audio.py`

Converts audio to standardized format (16kHz, mono, 16-bit WAV) with normalization.

**Input**: Raw audio directory + metadata.csv
**Output**: Preprocessed audio directory + new metadata.csv

```bash
python preprocess_audio.py --config your_language_tts.yaml
```

**Features**:
- Supports multiple input formats (webm, mp3, wav, flac, ogg, m4a, etc.)
- Resamples to 16kHz
- Converts to mono
- Normalizes audio to -20dB
- Creates new output directory (doesn't modify input)

**Output Structure**:
```
/path/to/preprocessed/
├── audio/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── metadata.csv  (NEW file with updated paths)
```

#### 3. Speaker Diarization: `speaker_diarization.py`

Extracts speaker embeddings and clusters them to assign speaker IDs.

**Input**: Preprocessed audio directory
**Output**: Speaker ID mapping + cluster samples

```bash
python speaker_diarization.py --config your_language_tts.yaml
```

**Features**:
- Uses Resemblyzer for speaker embeddings
- HDBSCAN or K-means clustering
- Automatic speaker count detection
- Generates sample audio for each cluster (for verification)
- Caches embeddings for faster re-runs

**Output Structure**:
```
/path/to/speaker_analysis/
├── embeddings.npy           # Cached embeddings
├── speaker_ids.json         # filename → speaker_id mapping
└── cluster_samples/         # Sample audio per speaker
    ├── speaker_0_sample_1.wav
    ├── speaker_0_sample_2.wav
    └── ...
```

#### 4. Dataset Preparation: `prepare_tts_dataset.py`

Merges audio, metadata, and speaker IDs into a HuggingFace dataset with train/val splits.

**Input**: Preprocessed audio + metadata.csv + speaker_ids.json
**Output**: HuggingFace dataset with splits

```bash
python prepare_tts_dataset.py --config your_language_tts.yaml
```

**Features**:
- Creates train/validation/test splits
- Filters by audio duration
- Adds speaker IDs to dataset
- Computes dataset statistics
- Saves in HuggingFace format

**Output Structure**:
```
/path/to/hf_dataset/
├── train/
├── validation/
└── dataset_info.json
```

**Dataset Schema**:
- `audio`: Audio array (16kHz, mono)
- `text`: Transcription text
- `speaker_id`: Speaker identifier (int)
- `duration`: Audio duration in seconds
- `file_name`: Original filename

#### 5. Model Training: `train_mms_tts.py`

Wrapper script that generates config and calls the official `finetune-hf-vits` training script.

**Input**: HuggingFace dataset + config
**Output**: Fine-tuned model checkpoints

```bash
python train_mms_tts.py --config your_language_tts.yaml
```

**Features**:
- Generates training config in `finetune-hf-vits` format
- Automatically builds monotonic alignment if needed
- Calls official training script with `accelerate`
- Supports pushing to HuggingFace Hub

**Options**:
```bash
# Generate config only (don't train)
python train_mms_tts.py --config your_language_tts.yaml --generate_config_only

# Push to HuggingFace Hub after training
python train_mms_tts.py --config your_language_tts.yaml --push_to_hub --hub_model_id your-username/model-name
```

## Missing Verses Generation

After training, you can generate audio for verses that were missing from the original dataset.

### 1. Extract Missing Verses
Identify verses in CODEX files that don't have audio and extract their text:

```bash
python extract_missing_verses.py --config config.yaml
```

This creates a `metadata.csv` in the generated output directory.

### 2. Generate Audio
Use the fine-tuned model to generate audio for the extracted verses:

```bash
python batch_tts_inference.py --config config.yaml
```

This will:
- Automatically find the best checkpoint (lowest validation loss)
- Generate audio for each missing verse
- Save files as `BOOK-CHAPTER-VERSE.webm`
- Update metadata with duration and speaker info

### 3. Interactive Demo
Launch a web interface to browse generated verses and generate custom text:

```bash
python gradio_demo.py --config config.yaml
```
## Advanced Usage

### Re-running Individual Steps

Since each step creates new output directories, you can re-run any step with different parameters:

```bash
# Re-preprocess with different normalization settings
# (edit config first)
python preprocess_audio.py --config your_language_tts.yaml

# Re-run speaker diarization with different clustering
python speaker_diarization.py --config your_language_tts.yaml

# Re-prepare dataset with different train/val split
python prepare_tts_dataset.py --config your_language_tts.yaml
```

### Using Command-Line Arguments

All scripts support both config files and command-line arguments:

```bash
# Using config file (recommended)
python preprocess_audio.py --config config.yaml

# Using command-line arguments
python preprocess_audio.py \
  --input_dir /path/to/raw/audio \
  --input_metadata /path/to/raw/metadata.csv \
  --output_dir /path/to/preprocessed \
  --sample_rate 16000 \
  --normalize
```

### Monitoring Training

Training uses TensorBoard by default:

```bash
# In a separate terminal
tensorboard --logdir /path/to/checkpoints/logs
```

### Using the Fine-tuned Model

After training, use the model with HuggingFace Transformers:

```python
from transformers import pipeline
import scipy

model_id = "/path/to/checkpoints"  # or "your-username/model-name" if pushed to Hub
synthesiser = pipeline("text-to-speech", model_id, device=0)

speech = synthesiser("Your text here in the target language")

scipy.io.wavfile.write("output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])
```

## Hardware Requirements

### Minimum
- **GPU**: 8GB VRAM (for testing)
- **RAM**: 16GB
- **Disk**: 50GB free space

### Recommended
- **GPU**: 12GB+ VRAM (NVIDIA with CUDA support)
- **RAM**: 32GB
- **Disk**: 100GB+ free space

### Training Time
- With 12GB GPU: ~20 minutes for 80-150 samples
- With 8GB GPU: ~30-40 minutes for 80-150 samples
- Scales linearly with dataset size

## Troubleshooting

### Common Issues

**1. "No audio files found"**
- Check that your audio directory path is correct
- Verify audio files have supported extensions (.wav, .webm, .mp3, etc.)

**2. "Speaker diarization failed"**
- Ensure audio is preprocessed first (16kHz, mono)
- Check that resemblyzer is installed: `uv add resemblyzer`

**3. "Training fails with CUDA out of memory"**
- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing: true`

**4. "Monotonic alignment not found" or "ModuleNotFoundError: No module named 'Cython'"**
- Ensure Cython is installed: `uv add cython`
- Build monotonic alignment manually:
  ```bash
  cd finetune-hf-vits/monotonic_align
  mkdir -p monotonic_align
  uv run python setup.py build_ext --inplace
  cd ../..
  ```
- The build creates `core.cpython-313-x86_64-linux-gnu.so` in the `monotonic_align/monotonic_align/` subdirectory

**5. "Dataset has no validation split"**
- Check that your dataset has enough samples for the split ratio
- Adjust `val_split` in config if needed

## Project Structure

```
audio_text_tests/
├── gitlab_to_hf_dataset.py      # Data collection
├── preprocess_audio.py           # Audio preprocessing
├── speaker_diarization.py        # Speaker clustering
├── prepare_tts_dataset.py        # Dataset preparation
├── train_mms_tts.py             # Training wrapper
├── config.yaml               # Example config (Xiang language)
├── finetune-hf-vits/            # Official training code (submodule)
├── TTS_FINETUNING_REQUIREMENTS.md  # Project requirements doc
├── TTS_WORKFLOW_PLANNING.md     # Workflow planning doc
└── TTS_FINETUNING_README.md     # This file
```

## References

- [MMS-TTS Paper](https://arxiv.org/abs/2305.13516) - Scaling Speech Technology to 1,000+ Languages
- [VITS Paper](https://arxiv.org/abs/2106.06103) - Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech
- [finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) - Official fine-tuning repository
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model integration
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Speaker embedding extraction

## License

- **This Pipeline**: MIT License (or your chosen license)
- **MMS Models**: CC BY-NC 4.0 (non-commercial)
- **VITS Models**: MIT License
- **Fine-tuned Models**: Inherit the license of their base model

## Contributing

Contributions are welcome! Please:
1. Keep generic tools language-agnostic
2. Put language-specific settings in YAML configs
3. Maintain immutable output principle (new directories for each step)
4. Update documentation for new features

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the planning documents in the repository
3. Open an issue with detailed error messages and config

---

**Last Updated**: 2025-11-28