# Generate Missing Verses Audio - Implementation Plan

## Overview

This document describes the implementation of a system to generate synthetic audio for Bible verses that don't have paired audio data. The system will:

1. Identify verses that exist in the CODEX files but don't have corresponding audio files (or are marked deleted/missing).
2. Extract the text content for these missing verses using the configured text source.
3. Use the trained TTS model to generate audio.
4. Save files with dynamic verse-reference-based filenames (e.g., `MAT-001-001.webm`).
5. Create a metadata CSV compatible with the existing HuggingFace dataset format.

## Configuration Extensions

Add the following section to `config.yaml`:

```yaml
# Missing verses generation configuration
missing_verses_generation:
  # Output configuration
  output_dir: "/path/to/output/generated_missing"
  audio_dir: "audio"  # subdirectory for generated audio files
  metadata_file: "metadata.csv"

  # TTS model configuration
  # Set to "auto" to find best checkpoint, or specific path
  tts_model_path: "auto" 
  # tts_model_path: "/path/to/output/checkpoints/checkpoint-15000"
  
  # Speaker configuration
  speaker_id: 0  # Default speaker ID to use for generation

  # Audio format (match existing data)
  audio_format: "webm"
  sample_rate: 16000
  channels: 1

  # Generation parameters (VITS standard)
  noise_scale: 0.667
  noise_scale_w: 0.8
  length_scale: 1.0

  # Processing
  resume_from_checkpoint: true  # skip already generated files (checks file existence)
  retry_attempts: 10 # Exponential backoff retries for failures

  # Filtering options
  min_text_length: 5  # minimum characters for generation (skip if shorter)
  max_text_length: 500  # maximum characters for generation (log if longer)
```

## Implementation Approach

Use **Approach 2: Batch Inference with Data Preparation** as it:
- Leverages existing codebase patterns
- Provides efficient bulk processing
- Integrates with the current workflow
- Allows GPU acceleration for faster generation

## Implementation Steps

### Phase 1: Data Extraction and Preparation

1. **Create `extract_missing_verses.py`**
   - Load CODEX files from GitLab or local directory (reusing `GitLabDatasetDownloader` logic).
   - Iterate through CODEX files to identify "missing" verses:
     - `selectedAudioId` is null/empty
     - OR `selectedAudioId` exists but file is missing
     - OR all recordings are marked deleted
   - Extract text using the configured source (e.g., 'value'), reusing `extract_text_from_cell`.
   - Parse verse IDs to extract Book, Chapter, Verse (handling ranges like `MAT 1:1-3`).
   - Create a "missing verses" dataset in CSV format with columns: `verse_id`, `text`, `book`, `chapter`, `verse`, `filename`.
   - **Filename Convention:** `[BOOK]-[CHAPTER]-[VERSE].webm` (e.g., `MAT-001-001.webm`).

### Phase 2: TTS Inference Pipeline

2. **Create `batch_tts_inference.py`**
   - Load trained TTS model.
     - If `tts_model_path` is "auto", use `find_best_checkpoint.py` logic.
   - Process missing verses dataset.
   - **Preprocessing:** Apply `uroman` and normalization to match training pipeline.
   - **Inference:**
     - Use configured `speaker_id`.
     - Use configured VITS parameters (`noise_scale`, etc.).
     - Handle errors with exponential backoff (up to 10 retries), then log and quit.
   - **Filtering:**
     - Skip verses shorter than `min_text_length`.
     - Log verses longer than `max_text_length` (do not generate).
   - **Output:**
     - Save audio as `.webm`.
     - Update `metadata.csv` with columns: `file_name`, `transcription`, `speaker_id`, `duration`.
   - **Quality Check:** Implement basic checks (e.g., non-zero duration) if easy.

## File Structure

```
dataset/config/
├── raw/                          # Original downloaded data
│   ├── audio/
│   └── metadata.csv
├── generated_missing/            # Generated audio output
│   ├── audio/                    # Generated webm files
│   │   ├── GEN-001-001.webm
│   │   ├── MAT-001-001.webm
│   │   └── ...
│   └── metadata.csv              # Generated verses metadata
└── checkpoints/                  # Trained TTS model
```

## Dependencies

Add to `pyproject.toml` using uv:

```toml
[tool.uv]
dev-dependencies = [
    "gradio>=4.0.0",           # For demo interface
    "librosa>=0.10.0",         # Audio processing
    "soundfile>=0.12.0",       # Audio I/O
    "pydub>=0.25.0",           # Audio format conversion
    "transformers>=4.57.3",    # HuggingFace models (try newer version first)
    "datasets>=2.10.0",        # Dataset handling
    "torch>=2.0.0",            # PyTorch for inference
    "torchaudio>=2.0.0",       # Audio processing
]
```
*Note: Potential conflict with `finetune-hf-vits` requirements. Will attempt with newer versions first.*

## Script Interfaces

### extract_missing_verses.py
```bash
# Extract missing verses from CODEX files
python extract_missing_verses.py --config config.yaml
```

### batch_tts_inference.py
```bash
# Generate audio for missing verses
python batch_tts_inference.py --config config.yaml
```

## Error Handling and Recovery

- **Retries:** Implement exponential backoff for model failures (up to 10 times).
- **Logging:** Log failed generations to console/file.
- **Resume:** Check if output file exists to skip already generated verses.
- **Validation:** Log summary of skipped (too short) and ignored (too long) verses.

## Performance Considerations

- Use GPU acceleration for inference.
- Cache TTS model in memory.
- Efficient audio encoding.

---

## Task 2: Gradio Demo Interface (Second Priority)

### Overview
Create a simple web interface to browse and play generated missing verses.

### Implementation Approach

1. **Create `gradio_demo.py`**
   - Reference `config.yaml` for configuration.
   - Serve pre-generated audio files from `generated_missing/audio/`.
   - **Custom Generation:** Allow users to input text and generate speech using the loaded model and uroman settings.
   - **No Translation:** Translation feature removed from scope.
   - **No Real vs Gen:** Comparison removed from scope.

2. **Interface Components**
   - Verse selector (dropdown of generated verses).
   - Audio player.
   - Download button.
   - **Custom Text Input:** Text area for entering new text.
   - **Generate Button:** To trigger inference for custom text.

### Configuration Extensions

Add to `config.yaml`:

```yaml
# Gradio demo interface
demo:
  port: 7860
  host: "0.0.0.0"
  share: false
  enable_download: true
```

### Usage
```bash
# Launch demo interface
python gradio_demo.py --config config.yaml