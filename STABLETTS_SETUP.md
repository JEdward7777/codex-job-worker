# StableTTS Fine-tuning Setup Guide

This guide explains what models you need to download for fine-tuning StableTTS and how to set them up.

## Required Models

### 1. Text-to-Mel Model (Required for Fine-tuning)

This is the pretrained StableTTS model that you'll fine-tune on your data.

**Model:** StableTTS v1.1
- **Download Link:** https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/StableTTS/checkpoint_0.pt
- **Size:** ~31M parameters
- **Trained on:** 600 hours of multilingual data (Chinese, English, Japanese)
- **Location:** Place in `./StableTTS/checkpoints/checkpoint_0.pt`

**Download command:**
```bash
cd StableTTS/checkpoints
wget https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/StableTTS/checkpoint_0.pt
```

### 2. Mel-to-Wav Model (Vocoder - Required for Inference)

You need a vocoder to convert mel spectrograms to audio waveforms. Choose ONE of the following:

#### Option A: Vocos (Recommended)
- **Download Link:** https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt
- **Trained on:** 2k hours
- **Location:** Place in `./StableTTS/vocoders/pretrained/vocos.pt`

**Download command:**
```bash
cd StableTTS/vocoders/pretrained
wget https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt
```

#### Option B: FireflyGAN
- **Download Link:** https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt
- **Trained on:** HiFi-16kh dataset
- **Location:** Place in `./StableTTS/vocoders/pretrained/firefly-gan-base-generator.ckpt`

**Download command:**
```bash
cd StableTTS/vocoders/pretrained
wget https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt
```

## Directory Structure

After downloading, your directory structure should look like:

```
StableTTS/
├── checkpoints/
│   └── checkpoint_0.pt          # Text-to-Mel model (for fine-tuning)
└── vocoders/
    └── pretrained/
        └── vocos.pt              # Vocoder (for inference)
        # OR
        └── firefly-gan-base-generator.ckpt
```

## Usage with train_stabletts.py

### Fine-tuning from Pretrained Model

To fine-tune from the pretrained StableTTS model:

```bash
python train_stabletts.py \
  --train_dataset_path ./output/filelist.json \
  --model_save_path ./my_finetuned_model \
  --log_dir ./logs \
  --pretrained_model ./StableTTS/checkpoints/checkpoint_0.pt \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --val_split 0.1
```

### Training from Scratch

If you want to train from scratch (not recommended unless you have a large dataset):

```bash
python train_stabletts.py \
  --train_dataset_path ./output/filelist.json \
  --model_save_path ./my_model \
  --log_dir ./logs \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 1000
```

### Resuming Training

The script will automatically detect and resume from the latest checkpoint in `--model_save_path`. To resume from a specific checkpoint:

```bash
python train_stabletts.py \
  --train_dataset_path ./output/filelist.json \
  --model_save_path ./my_finetuned_model \
  --log_dir ./logs \
  --checkpoint ./my_finetuned_model/checkpoint_50.pt \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 100
```

## Complete Workflow

1. **Download pretrained models** (see commands above)
2. **Prepare your dataset** using `preprocess_stabletts.py`:
   ```bash
   python preprocess_stabletts.py \
     --input_csv ./metadata.csv \
     --audio_base_dir ./audio \
     --output_json ./output/filelist.json \
     --output_feature_dir ./output \
     --language english
   ```

3. **Fine-tune the model** using `train_stabletts.py`:
   ```bash
   python train_stabletts.py \
     --train_dataset_path ./output/filelist.json \
     --model_save_path ./checkpoints \
     --log_dir ./logs \
     --pretrained_model ./StableTTS/checkpoints/checkpoint_0.pt \
     --batch_size 32 \
     --learning_rate 1e-4 \
     --num_epochs 100 \
     --val_split 0.1 \
     --save_interval 5
   ```

4. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir ./logs
   ```

5. **Use your fine-tuned model** for inference (see `StableTTS/inference.ipynb` or `StableTTS/webui.py`)

## Recommended Training Parameters

### For Fine-tuning (Recommended)
Based on StableTTS defaults and fine-tuning best practices:

- **batch_size:** 32 (reduce to 16 or 8 if you have GPU memory issues)
- **learning_rate:** 1e-4 (0.0001) - same as original training
  - For smaller datasets (<10 hours): consider 5e-5 to avoid overfitting
  - For larger datasets (>50 hours): 1e-4 works well
- **num_epochs:**
  - Small dataset (<5 hours): 50-100 epochs
  - Medium dataset (5-20 hours): 100-200 epochs
  - Large dataset (>20 hours): 200-500 epochs
  - Monitor validation loss to avoid overfitting
- **save_interval:** 5-10 epochs (to avoid filling disk with too many checkpoints)

### For Training from Scratch (Not Recommended)
- **batch_size:** 32
- **learning_rate:** 1e-4
- **num_epochs:** 10000 (as per StableTTS default)
- Requires a large dataset (100+ hours recommended)

## Notes

- **Vocoder is NOT needed for training** - it's only required for inference (converting mel spectrograms to audio)
- **Fine-tuning is recommended** over training from scratch, especially for smaller datasets
- The pretrained model supports **Chinese, English, and Japanese** out of the box
- The original StableTTS was trained on **600 hours** of data with batch_size=32 and learning_rate=1e-4
- The `--best_metric` argument lets you choose which loss to optimize for (default: `total_loss`)
- Use `--val_split 0.1` to reserve 10% of data for validation monitoring

## Troubleshooting

### CUDA Out of Memory Error

**Problem:** Training fails with `torch.OutOfMemoryError: CUDA out of memory`

**Solution:** Reduce the batch size based on your GPU memory:

| GPU Memory | Recommended batch_size | Notes |
|------------|----------------------|-------|
| 24GB (RTX 3090/4090) | 32 | Default, works well |
| 12GB (RTX 3060/4070) | 8-16 | Start with 8, increase if stable |
| 8GB (RTX 3070) | 4-8 | Start with 4 |
| 6GB or less | 2-4 | Very slow, consider cloud GPU |

**Example for 12GB GPU:**
```bash
python train_stabletts.py \
  --train_dataset_path ./output/filelist.json \
  --model_save_path ./checkpoints \
  --log_dir ./logs \
  --pretrained_model ./StableTTS/checkpoints/checkpoint_0.pt \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_epochs 100
```

**Additional memory-saving tips:**
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable
- Reduce `num_workers` in the script (currently fixed at 4)
- Close other GPU-using applications

### Model not found error
Make sure you've downloaded the models to the correct locations as specified above.

### Poor quality after fine-tuning
- Make sure your audio data is high quality (44.1kHz, clean recordings)
- Try a lower learning rate (5e-5 instead of 1e-4)
- Train for more epochs
- Check validation loss to ensure the model is learning