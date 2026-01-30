# StableTTS Checkpoint Comparison Interface

A Gradio-based web interface for comparing different TTS model checkpoints to help evaluate which checkpoint produces the best audio quality.

## Features

- **Free-form TTS Tab**: Generate speech from custom text using any checkpoint
- **Validation Comparison Tab**: Systematically compare checkpoints on validation samples
- **Smart Caching**: Disk-based caching to avoid regenerating the same audio
- **Reference Audio Support**: Upload custom reference audio or use a default
- **Progressive Comparison**: Generate audio on-demand for each checkpoint
- **Ground Truth Playback**: Compare generated audio against original recordings

## Installation

Ensure you have the required dependencies installed:

```bash
pip install gradio torch torchaudio
```

If using uroman preprocessing:
```bash
pip install uroman
```

## Usage

### Basic Usage

```bash
python gradio_demo_stabletts.py \
  --checkpoints-dir ./StableTTS/checkpoints \
  --default-reference-audio ./reference.wav \
  --g2p-type english
```

### With Validation Dataset

```bash
python gradio_demo_stabletts.py \
  --checkpoints-dir ./StableTTS/checkpoints \
  --validation-csv /path/to/metadata.csv \
  --default-reference-audio ./reference.wav \
  --filename-column file_name \
  --text-column transcription \
  --g2p-type english
```

### With Uroman Preprocessing

```bash
python gradio_demo_stabletts.py \
  --checkpoints-dir ./StableTTS/checkpoints \
  --default-reference-audio ./reference.wav \
  --g2p-type english \
  --use-uroman \
  --uroman-lang eng
```

### Public Sharing

To create a public link for external evaluators:

```bash
python gradio_demo_stabletts.py \
  --checkpoints-dir ./StableTTS/checkpoints \
  --validation-csv ./validation/metadata.csv \
  --default-reference-audio ./reference.wav \
  --g2p-type english \
  --share
```

## Command Line Arguments

### Required Arguments

- `--checkpoints-dir`: Directory containing checkpoint files (*.pt)

### Dataset Arguments

- `--validation-csv`: Path to validation metadata CSV file
- `--filename-column`: Column name for audio filenames (default: `file_name`)
- `--text-column`: Column name for text (default: `transcription`)

### Audio Arguments

- `--default-reference-audio`: Path to default reference audio file
- `--cache-dir`: Directory for caching generated audio (default: `./gradio_cache`)

### Text Processing Arguments

- `--g2p-type`: Text-to-phoneme conversion type (choices: `chinese`, `english`, `japanese`, default: `english`)
- `--use-uroman`: Enable uroman text romanization preprocessing
- `--uroman-lang`: ISO code for uroman romanization (default: `eng`)

### Inference Parameters

- `--diffusion-steps`: Number of diffusion steps (default: 10)
- `--temperature`: Sampling temperature (default: 1.0)
- `--length-scale`: Speech pace control (default: 1.0)
- `--cfg-scale`: Classifier-free guidance scale (default: 3.0)
- `--solver`: ODE solver for diffusion (default: None)

### Server Configuration

- `--share`: Create a public Gradio link
- `--server-port`: Port for Gradio server (default: 7860)
- `--server-name`: Server name for Gradio (default: `0.0.0.0`)

## Interface Guide

### Tab 1: Free-form TTS

1. Enter text in the text box
2. Select a checkpoint from the dropdown
3. (Optional) Upload a reference audio file
4. Click "Generate Audio"
5. Listen to or download the generated audio

### Tab 2: Validation Comparison

1. Select a validation sample from the dropdown
2. Listen to the ground truth (original recording)
3. Select a checkpoint and click "➕ Generate & Add Comparison"
4. The audio is generated immediately and displayed
5. Repeat step 3 to add more checkpoints for comparison
6. The dropdown automatically suggests the next checkpoint

**Tips:**
- The interface automatically sorts checkpoints by iteration number
- Generated audio is cached to disk for faster repeated access
- If cache is deleted, audio will be regenerated automatically
- All checkpoints use the same default reference audio for consistency

## Checkpoint Naming

Checkpoints should be named with a numeric iteration identifier:
- `checkpoint_100.pt`
- `checkpoint_200.pt`
- `checkpoint_300.pt`
- etc.

The interface extracts the number and sorts checkpoints numerically.

## Validation CSV Format

The validation CSV should have the following structure:

```csv
file_name,transcription,speaker_id,duration
audio/sample1.wav,"Text content here",0,5.2
audio/sample2.wav,"More text content",0,7.8
```

Required columns (configurable via arguments):
- Audio filename column (default: `file_name`)
- Text column (default: `transcription`)

Audio file paths should be relative to the CSV file location.

## Caching

Generated audio is cached using a hash of:
- Checkpoint name
- Input text
- Reference audio

Cache files are stored in the cache directory (default: `./gradio_cache`).

**Cache Management:**
- Cache can be safely deleted - audio will regenerate on demand
- Cache persists across sessions
- No automatic cleanup (manage manually if needed)

## Troubleshooting

### "Checkpoint directory not found"
Ensure the path to `--checkpoints-dir` is correct and contains `.pt` files.

### "Vocoder not found"
Ensure StableTTS vocoders are installed at `StableTTS/vocoders/pretrained/vocos.pt`.

### "No validation samples loaded"
Check that:
- `--validation-csv` path is correct
- CSV has the correct column names (use `--filename-column` and `--text-column` if different)
- Audio files exist at the paths specified in the CSV

### "Failed to load reference audio"
Ensure the reference audio file exists and is a valid audio format (WAV, MP3, etc.).

### Out of memory errors
- Reduce the number of cached models (currently limited to 3)
- Use a machine with more GPU memory
- Reduce `--diffusion-steps` for faster, less memory-intensive generation

## Example Workflow

1. Train your TTS model, saving checkpoints every 100 iterations
2. Prepare a validation dataset with ground truth audio and transcriptions
3. Launch the interface with validation dataset:
   ```bash
   python gradio_demo_stabletts.py \
     --checkpoints-dir ./checkpoints \
     --validation-csv ./validation/metadata.csv \
     --default-reference-audio ./reference.wav \
     --g2p-type english \
     --share
   ```
4. Share the public link with evaluators
5. Evaluators compare checkpoints on validation samples
6. Collect feedback to determine the best checkpoint

## Notes

- The interface does not expose g2p type or uroman settings to end users - these are configured at launch
- All inference parameters are set via command line for consistency
- The interface is designed for evaluation, not for production TTS serving