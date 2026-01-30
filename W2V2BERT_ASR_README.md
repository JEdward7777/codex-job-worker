# W2V2-BERT ASR Training and Inference

This directory contains scripts for fine-tuning and using Wav2Vec2-BERT models for Automatic Speech Recognition (ASR), based on the [HuggingFace tutorial](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Overview

The W2V2-BERT (Wav2Vec2-BERT) implementation provides:
- **Training**: Fine-tune pre-trained W2V2-BERT models on custom datasets
- **Inference**: Batch transcription of audio files
- **Evaluation**: WER (Word Error Rate) and CER (Character Error Rate) metrics
- **Flexibility**: Command-line arguments for all parameters (no config file editing needed)

## Files

- [`train_w2v2bert_asr.py`](train_w2v2bert_asr.py) - Main training script
- [`inference_w2v2bert_asr.py`](inference_w2v2bert_asr.py) - Batch inference script

## Installation

Dependencies are managed via `uv`. The required packages are already in `pyproject.toml`:
- `transformers`
- `datasets`
- `evaluate`
- `pandas`
- `torch`
- `torchaudio`
- `jiwer` (for WER/CER metrics)

## Training

### Basic Usage

```bash
python train_w2v2bert_asr.py \
  --csv_path data/my_dataset.csv \
  --output_dir outputs/w2v2bert_finetuned
```

### CSV Format

Your CSV file should have columns for audio file paths and transcriptions. The script auto-detects common column names:
- Audio: `file_name`, `audio_path`, `audio`, `file`, `path`
- Text: `transcription`, `text`, `transcript`, `sentence`

Example CSV:
```csv
file_name,transcription
audio/sample1.wav,This is the first transcription
audio/sample2.wav,This is the second transcription
```

### Common Training Options

```bash
python train_w2v2bert_asr.py \
  --csv_path data/biblical_recordings.csv \
  --output_dir outputs/biblical_w2v2bert \
  --model_name facebook/w2v-bert-2.0 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_steps 500 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 100 \
  --fp16 \
  --group_by_length
```

### Data Preprocessing Options

**Text Normalization:**
```bash
--lowercase                    # Convert all text to lowercase
--remove_punctuation          # Remove punctuation marks
--remove_numbers              # Remove numeric digits
```

**Audio Duration Filtering:**
```bash
--min_duration_sec 1.0        # Minimum audio duration in seconds
--max_duration_sec 30.0       # Maximum audio duration in seconds
--filter_by_std_dev 3.0       # Filter outliers by standard deviations from mean
```

**Data Splitting:**
```bash
--train_split 0.8             # 80% for training
--val_split 0.1               # 10% for validation
--test_split 0.1              # 10% for testing
--seed 42                     # Random seed for reproducibility
```

### Training Parameters

**Model Selection:**
```bash
--model_name facebook/w2v-bert-2.0              # Default model
--model_name facebook/w2v-bert-2.0-v2           # Alternative
```

**Batch Size and Memory:**
```bash
--per_device_train_batch_size 8       # Batch size per GPU (default: 8)
--per_device_eval_batch_size 8        # Eval batch size (default: 8)
--gradient_accumulation_steps 2       # Accumulate gradients (default: 2)
--fp16                                # Use mixed precision (recommended)
```

**Learning Rate and Optimization:**
```bash
--learning_rate 1e-4                  # Learning rate (default: 1e-4)
--warmup_steps 500                    # Warmup steps (default: 500)
--weight_decay 0.01                   # Weight decay (default: 0.01)
```

**Training Duration:**
```bash
--num_train_epochs 10                 # Number of epochs (default: 10)
--max_steps -1                        # Max steps (-1 = use epochs)
```

**Checkpointing:**
```bash
--save_steps 500                      # Save checkpoint every N steps
--save_total_limit 3                  # Keep only N best checkpoints
--eval_steps 500                      # Evaluate every N steps
--logging_steps 100                   # Log every N steps
```

**Data Loading:**
```bash
--dataloader_num_workers 4            # Number of data loading workers
--group_by_length                     # Group samples by length (faster training)
```

### Advanced Features

**Auto-Resume Training:**
The script automatically detects and resumes from the latest checkpoint in the output directory.

**Language Detection:**
The script auto-detects the language from text samples and configures the processor accordingly.

**Custom Vocabulary:**
A vocabulary is automatically created from your dataset's transcriptions.

**Evaluation Metrics:**
- WER (Word Error Rate) - computed every evaluation step
- CER (Character Error Rate) - computed every evaluation step

**TensorBoard Logging:**
```bash
tensorboard --logdir outputs/w2v2bert_finetuned/runs
```

### Example: Training on Biblical Text

```bash
python train_w2v2bert_asr.py \
  --csv_path data/biblical_recordings.csv \
  --output_dir outputs/biblical_w2v2bert \
  --model_name facebook/w2v-bert-2.0 \
  --num_train_epochs 15 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_steps 1000 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 50 \
  --fp16 \
  --group_by_length \
  --lowercase \
  --min_duration_sec 1.0 \
  --max_duration_sec 30.0 \
  --train_split 0.85 \
  --val_split 0.10 \
  --test_split 0.05
```

## Inference

### Basic Usage

```bash
python inference_w2v2bert_asr.py \
  --model_path outputs/w2v2bert_finetuned \
  --audio_dir audio_files/ \
  --output_csv transcriptions.csv
```

### Inference Options

```bash
python inference_w2v2bert_asr.py \
  --model_path outputs/biblical_w2v2bert \
  --audio_dir recordings/ \
  --output_csv results/transcriptions.csv \
  --batch_size 8 \
  --include_confidence \
  --num_alternatives 3
```

**Parameters:**
- `--model_path`: Path to fine-tuned model directory
- `--audio_dir`: Directory containing audio files to transcribe
- `--output_csv`: Output CSV file path
- `--batch_size`: Batch size for inference (default: 8)
- `--include_confidence`: Include confidence scores in output
- `--num_alternatives`: Number of alternative transcriptions (default: 1)

**Supported Audio Formats:**
- `.wav`
- `.mp3`
- `.flac`
- `.ogg`
- `.m4a`

### Output Format

The output CSV contains:
```csv
audio_file,audio_path,transcription,confidence,alternative_1,alternative_2
sample1.wav,/path/to/sample1.wav,This is the transcription,0.95,This is a transcription,This was the transcription
```

## GPU Memory Requirements

**Recommended GPU Memory:**
- Minimum: 8GB (with small batch size and gradient accumulation)
- Recommended: 12GB or more
- Optimal: 16GB+ (allows larger batch sizes)

**Memory Optimization Tips:**
1. Use `--fp16` for mixed precision training
2. Reduce `--per_device_train_batch_size`
3. Increase `--gradient_accumulation_steps` to maintain effective batch size
4. Use `--gradient_checkpointing` if available (saves memory at cost of speed)

Example for 8GB GPU:
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--fp16
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/w2v2bert_finetuned/runs

# View in browser
# http://localhost:6006
```

### Training Logs

The script logs:
- Training loss
- Validation loss
- WER (Word Error Rate)
- CER (Character Error Rate)
- Learning rate
- Steps per second

### Checkpoints

Checkpoints are saved in `{output_dir}/checkpoint-{step}/`:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `preprocessor_config.json` - Processor configuration
- `training_args.bin` - Training arguments
- `vocab.json` - Vocabulary

The best checkpoint (lowest WER) is saved separately in `{output_dir}/best_checkpoint/`.

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size:
   ```bash
   --per_device_train_batch_size 2
   ```

2. Increase gradient accumulation:
   ```bash
   --gradient_accumulation_steps 8
   ```

3. Enable mixed precision:
   ```bash
   --fp16
   ```

4. Filter long audio files:
   ```bash
   --max_duration_sec 20.0
   ```

### Poor WER/CER Scores

1. Train for more epochs:
   ```bash
   --num_train_epochs 20
   ```

2. Adjust learning rate:
   ```bash
   --learning_rate 5e-5
   ```

3. Increase warmup steps:
   ```bash
   --warmup_steps 1000
   ```

4. Check data quality:
   - Ensure transcriptions are accurate
   - Verify audio quality
   - Check for mismatched audio/text pairs

### CSV Loading Errors

1. Verify CSV format (comma-delimited)
2. Check column names (use `file_name` and `transcription`)
3. Ensure file paths are correct (relative or absolute)
4. Check for special characters in transcriptions

### Audio Loading Errors

1. Verify audio file format is supported
2. Check file paths exist
3. Ensure audio files are not corrupted
4. Verify sample rate (will auto-resample to 16kHz)

## Performance Tips

1. **Use `--group_by_length`**: Groups similar-length samples for faster training
2. **Enable `--fp16`**: Reduces memory usage and speeds up training
3. **Optimize batch size**: Find the largest batch size that fits in GPU memory
4. **Use multiple workers**: `--dataloader_num_workers 4` for faster data loading
5. **Filter outliers**: Use `--filter_by_std_dev` to remove extremely long/short samples

## Comparison with Other Models

This implementation uses W2V2-BERT, which offers:
- **Better performance** than original Wav2Vec2 on many benchmarks
- **Multilingual support** with pre-trained models
- **Efficient fine-tuning** with relatively small datasets
- **State-of-the-art results** on various ASR tasks

Alternative models in this repo:
- **StableTTS**: Text-to-Speech (TTS) model
- **VITS**: End-to-end TTS model
- **MMS-TTS**: Multilingual TTS model

## References

- [HuggingFace W2V2-BERT Tutorial](https://huggingface.co/blog/fine-tune-w2v2-bert)
- [W2V2-BERT Paper](https://arxiv.org/abs/2108.06209)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)

## License

This implementation follows the same license as the original W2V2-BERT model and HuggingFace Transformers library.