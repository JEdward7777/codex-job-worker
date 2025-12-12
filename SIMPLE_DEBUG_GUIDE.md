# Simple Guide: Debugging run_vits_finetuning.py

## Key Insight: You Can Run It Directly Without Accelerate!

Even though `run_vits_finetuning.py` is designed to work with accelerate, **you can debug it directly** by running it as a normal Python script. Accelerate will automatically detect single-process mode.

## Quick Start: Direct Debugging (Recommended)

### Option 1: VSCode Debugger (Easiest)

1. Open `finetune-hf-vits/run_vits_finetuning.py`
2. Set breakpoints (click left of line numbers)
3. Press F5 and add this simple configuration to your existing launch.json:

```json
{
    "name": "Debug VITS Direct",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/finetune-hf-vits/run_vits_finetuning.py",
    "args": [
        "${workspaceFolder}/finetune-hf-vits/training_config_examples/finetune_english.json"
    ],
    "console": "integratedTerminal",
    "justMyCode": false
}
```

4. Select "Debug VITS Direct" and press F5
5. Use F10 (step over), F11 (step into) to debug

### Option 2: Command Line

Simply run it directly:

```bash
python finetune-hf-vits/run_vits_finetuning.py \
    finetune-hf-vits/training_config_examples/finetune_english.json
```

The script will work fine! Accelerate detects it's running in single-process mode.

### Option 3: With Python Debugger (pdb)

```bash
python -m pdb finetune-hf-vits/run_vits_finetuning.py \
    finetune-hf-vits/training_config_examples/finetune_english.json
```

## Why This Works

Looking at the code:
- Line 874: `Accelerator()` automatically detects single-process mode
- Line 19: `from accelerate import Accelerator` - it's just a library import
- The script adapts to whether you use `accelerate launch` or run directly

## Tips for Better Debugging

### 1. Use a Small Debug Config

Create `finetune-hf-vits/training_config_examples/finetune_debug.json`:

```json
{
    "model_name_or_path": "facebook/mms-tts-eng",
    "dataset_name": "your-dataset-name",
    "output_dir": "./debug_output",
    "overwrite_output_dir": true,
    "do_train": true,
    "do_eval": false,
    "per_device_train_batch_size": 1,
    "num_train_epochs": 1,
    "max_steps": 5,
    "save_steps": 10,
    "logging_steps": 1,
    "max_train_samples": 10,
    "preprocessing_num_workers": 1,
    "dataloader_num_workers": 0
}
```

### 2. Common Breakpoint Locations

- **Line 524**: Start of `main()` function
- **Line 587**: Dataset loading begins
- **Line 803**: Model loading
- **Line 1090**: Training loop starts
- **Line 1094**: Forward pass through model

### 3. Limit GPU Memory

```bash
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
# Or for CPU debugging:
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

## When You Actually Need Accelerate Launch

You only need `accelerate launch` for:
- Multi-GPU training
- Distributed training across machines
- Mixed precision training (though single GPU works too)

For debugging, **direct execution is simpler and works perfectly!**

## Quick Comparison

| Method | Command | Best For |
|--------|---------|----------|
| **Direct (Recommended for debug)** | `python run_vits_finetuning.py config.json` | Single-stepping, breakpoints |
| With accelerate | `accelerate launch run_vits_finetuning.py config.json` | Production, multi-GPU |
| With pdb | `python -m pdb run_vits_finetuning.py config.json` | Command-line debugging |

## Summary

**You don't need accelerate to debug!** Just run the script directly with Python and use your normal debugging workflow. The script is smart enough to handle both modes.