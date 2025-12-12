# Training Instructions for TTS Model

## Current Status

✅ **Completed Steps:**
1. ✅ Data collection (downloaded 2892 audio files from GitLab)
2. ✅ Audio preprocessing (normalized to 16kHz mono WAV)
3. ✅ Speaker diarization (identified 2 speakers)
4. ✅ Dataset preparation (created HuggingFace dataset with train/val/test splits)
5. ✅ Dataset validation (verified with inspect_dataset.py)

## Dataset Summary

- **Total examples**: 2892
- **Train set**: 2602 examples (6.01 hours)
- **Validation set**: 289 examples (0.68 hours)
- **Test set**: 1 example
- **Speakers**: 2 unique speakers identified
- **Audio format**: 16kHz, mono, WAV
- **Text format**: Romanized Language
- **Dataset location**: `/path/to/output/hf_dataset`
- **Dataset size**: 736MB (uncompressed, optimized for training)

## Dataset Validation

Before training, you can inspect and validate the prepared dataset:

```bash
uv run python inspect_dataset.py
```

Or specify a custom dataset path:

```bash
uv run python inspect_dataset.py /path/to/your/hf_dataset
```

### What This Does

The [`inspect_dataset.py`](inspect_dataset.py:1) script provides comprehensive validation:

1. **Dataset Structure** - Shows splits (train/val/test) and feature types
2. **Data Loading Tests** - Verifies audio and text can be loaded correctly
3. **Sample Inspection** - Displays first 3 examples from each split with:
   - File names and transcriptions
   - Speaker IDs
   - Audio properties (sample rate, shape, signal strength)
   - Duration verification
4. **Summary Statistics** - Reports:
   - Total examples and unique speakers per split
   - Duration ranges and averages
   - Text length statistics
5. **Validation Checks** - Ensures:
   - All examples have audio data
   - All examples have text
   - All examples have speaker IDs
   - Audio has signal (not silent)
   - Sample rates are consistent

### Example Output

```
================================================================================
📊 DATASET STRUCTURE
--------------------------------------------------------------------------------
Splits: ['train', 'validation']

TRAIN split:
  - Number of examples: 2602
  - Features: ['audio', 'text', 'speaker_id', 'duration', 'file_name']
...

================================================================================
✅ VALIDATION CHECKS
--------------------------------------------------------------------------------
TRAIN:
  ✓ All examples have audio data
  ✓ All examples have text
  ✓ All examples have speaker IDs
  ✓ Audio has signal (10/10 samples checked)
  ✓ Consistent sample rate: {16000}

================================================================================
🎉 DATASET IS VALID AND READY FOR TRAINING!
================================================================================
```

## Next Step: Model Training

### Initial Setup for Fresh Clone

If you've just cloned this repository, you need to initialize the submodules and apply local patches:

```bash
# Initialize and update submodules
git submodule update --init --recursive

# Apply patches to fix compatibility issues
./apply_submodule_patches.sh
```

**What this does:**
- The [`finetune-hf-vits`](finetune-hf-vits/) submodule contains the training code
- The patch fixes compatibility issues with newer library versions:
  - Removes deprecated `send_example_telemetry` from transformers
  - Fixes matplotlib's deprecated `canvas.tostring_rgb()` method
- This is necessary because we don't control the upstream repository

**Note:** You only need to run this once after cloning. The patch is stored in [`finetune-hf-vits.patch`](finetune-hf-vits.patch:1) and automatically applied by the script.

### Prerequisites Check

Before training, verify the monotonic alignment module is built:

```bash
cd finetune-hf-vits/monotonic_align
ls -la monotonic_align/core.cpython*.so
```

If the `.so` file doesn't exist, build it:

```bash
cd finetune-hf-vits/monotonic_align
mkdir -p monotonic_align
uv run python setup.py build_ext --inplace
cd ../..
```

### Training Command

Run the training script with the configuration:

```bash
uv run train_mms_tts.py --config config.yaml
```

**Note:** Training will automatically resume from the latest checkpoint if one exists. You don't need to do anything special - just run the same command again after an interruption.

### What This Will Do

1. **Generate training configuration** in `finetune-hf-vits` format
2. **Verify monotonic alignment** is built (builds if needed)
3. **Check for existing checkpoints** and resume automatically if found
4. **Launch training** using the official `finetune-hf-vits/run_vits_finetuning.py` script
5. **Save checkpoints** to `/path/to/output/checkpoints`

### Training Configuration (from config.yaml)

- **Base model**: `facebook/mms-tts-cmn` (Mandarin Chinese MMS-TTS)
- **Target language**: Language (three letter language code)
- **Training epochs**: 250
- **Learning rate**: 1e-4
- **Batch size**: 8 per device
- **Gradient accumulation**: 4 steps
- **Mixed precision**: fp16
- **Save frequency**: Every 25 steps
- **Logging frequency**: Every 10 steps

### Expected Training Time

- **With 12GB GPU**: ~20-30 minutes for 250 epochs
- **With 8GB GPU**: ~40-60 minutes for 250 epochs
- Scales with dataset size (6 hours of audio)

### Monitoring Training

Training uses TensorBoard for monitoring. In a separate terminal:

```bash
tensorboard --logdir /home/lansford/vast_ai_tts/checkpoints/config-mms-tts/logs
```

Then open http://localhost:6006 in your browser.

### Optional: Push to HuggingFace Hub

To push the trained model to HuggingFace Hub after training:

```bash
uv run train_mms_tts.py --config config.yaml --push_to_hub --hub_model_id your-username/config-mms-tts
```

### Optional: Generate Config Only (No Training)

To just generate the training config without starting training:

```bash
uv run train_mms_tts.py --config config.yaml --generate_config_only
```

This creates the config file at:
`/home/lansford/vast_ai_tts/checkpoints/config-mms-tts/training_config.json`

### Troubleshooting

**If you get "CUDA out of memory":**
- Edit `config.yaml` and reduce `per_device_train_batch_size` (try 4 or 2)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing: true`

**If monotonic alignment fails:**
```bash
uv add cython
cd finetune-hf-vits/monotonic_align
mkdir -p monotonic_align
uv run python setup.py build_ext --inplace
cd ../..
```

**If training crashes:**
- Simply run the same training command again - it will automatically resume from the last checkpoint
- Check GPU memory: `nvidia-smi`
- Check logs in checkpoint directory
- Verify dataset is accessible: `uv run python inspect_dataset.py`

**To start training from scratch (ignoring checkpoints):**
- Delete or rename the checkpoint directory
- Or manually edit the generated `training_config.json` and set `"overwrite_output_dir": true`

### After Training

Once training completes, test the model:

```python
from transformers import pipeline
import scipy

model_id = "/home/lansford/vast_ai_tts/checkpoints/config-mms-tts"
synthesiser = pipeline("text-to-speech", model_id, device=0)

# Use romanized config text
speech = synthesiser("text example")

scipy.io.wavfile.write("test_output.wav", 
                       rate=speech["sampling_rate"], 
                       data=speech["audio"][0])
```

### Generating Missing Verses

You can also use the trained model to generate audio for verses that were missing from the original dataset:

1. **Extract missing verses** from CODEX files:
   ```bash
   python extract_missing_verses.py --config config.yaml
   ```

2. **Generate audio** (automatically uses best checkpoint):
   ```bash
   python batch_tts_inference.py --config config.yaml
   ```

3. **Browse results** with the web interface:
   ```bash
   python gradio_demo.py --config config.yaml
   ```
### Files and Directories

```
/path/to/output/
├── raw/                    # Original downloaded files (757MB)
├── preprocessed/           # Normalized audio (743MB)
├── speaker_analysis/       # Speaker diarization results
└── hf_dataset/            # Training dataset (736MB) ← READY FOR TRAINING

/home/lansford/vast_ai_tts/checkpoints/
└── config-mms-tts/         # Training checkpoints (will be created)
    ├── checkpoint-25/
    ├── checkpoint-50/
    ├── ...
    ├── logs/              # TensorBoard logs
    └── training_config.json
```

### Configuration File

The training uses: `config.yaml`

Key settings:
- Dataset path: `/path/to/output/hf_dataset`
- Checkpoint path: `/home/lansford/vast_ai_tts/checkpoints/config-mms-tts`
- Base model: `facebook/mms-tts-cmn`

---

## Ready to Train!

Your dataset has been validated and is ready. Simply run:

```bash
uv run train_mms_tts.py --config config.yaml
```

The training will start automatically and save checkpoints every 25 steps.