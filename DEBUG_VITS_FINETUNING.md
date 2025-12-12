# Debugging run_vits_finetuning.py with Accelerate

This guide explains how to single-step debug `run_vits_finetuning.py` which is designed to run with accelerate.

## Method 1: Using VSCode Debugger with Launch Configuration

### Step 1: Create a Launch Configuration

Add this to `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug VITS Finetuning (Single GPU)",
            "type": "debugpy",
            "request": "launch",
            "module": "accelerate.commands.launch",
            "args": [
                "--config_file", "finetune-hf-vits/accelerate_config_debug.yaml",
                "finetune-hf-vits/run_vits_finetuning.py",
                "finetune-hf-vits/training_config_examples/finetune_english.json"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "Debug VITS Finetuning (Direct - No Accelerate)",
            "type": "debugpy",
            "request": "launch",
            "program": "finetune-hf-vits/run_vits_finetuning.py",
            "args": [
                "finetune-hf-vits/training_config_examples/finetune_english.json"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}
```

### Step 2: Create Accelerate Debug Configuration

Create `finetune-hf-vits/accelerate_config_debug.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Step 3: Set Breakpoints and Debug

1. Open `finetune-hf-vits/run_vits_finetuning.py` in VSCode
2. Set breakpoints by clicking left of line numbers (e.g., line 524 in `main()`)
3. Press F5 or go to Run → Start Debugging
4. Select "Debug VITS Finetuning (Single GPU)"
5. Use F10 (step over), F11 (step into), Shift+F11 (step out) to navigate

## Method 2: Using Python Debugger (pdb)

### Add breakpoint in code:

```python
# Add this line where you want to break
import pdb; pdb.set_trace()
```

Then run:
```bash
accelerate launch --config_file finetune-hf-vits/accelerate_config_debug.yaml \
    finetune-hf-vits/run_vits_finetuning.py \
    finetune-hf-vits/training_config_examples/finetune_english.json
```

### Common pdb commands:
- `n` (next): Execute current line
- `s` (step): Step into function
- `c` (continue): Continue execution
- `l` (list): Show current code
- `p variable`: Print variable value
- `q` (quit): Exit debugger

## Method 3: Using debugpy for Remote Debugging

### Add to your script (at line 524, start of main()):

```python
def main():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached!")
    
    # Rest of main() function...
```

### Then:
1. Run the script normally with accelerate
2. Attach VSCode debugger using "Python: Attach" configuration

## Method 4: Single Process Debug Mode

### Modify training config to use minimal resources:

Create `finetune-hf-vits/training_config_examples/finetune_debug.json`:

```json
{
    "model_name_or_path": "facebook/mms-tts-eng",
    "dataset_name": "your-dataset",
    "output_dir": "./debug_output",
    "overwrite_output_dir": true,
    "do_train": true,
    "do_eval": false,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "max_steps": 10,
    "save_steps": 5,
    "logging_steps": 1,
    "max_train_samples": 10,
    "preprocessing_num_workers": 1,
    "dataloader_num_workers": 0,
    "seed": 42
}
```

## Method 5: Using Python's Built-in Debugger with Accelerate

Run with Python's debugger module:

```bash
python -m pdb -m accelerate.commands.launch \
    --config_file finetune-hf-vits/accelerate_config_debug.yaml \
    finetune-hf-vits/run_vits_finetuning.py \
    finetune-hf-vits/training_config_examples/finetune_english.json
```

## Tips for Effective Debugging

1. **Start Simple**: Use single GPU, small batch size, few samples
2. **Disable Multi-processing**: Set `num_processes: 1` in accelerate config
3. **Use CPU if needed**: Set `use_cpu: true` in accelerate config for easier debugging
4. **Add Logging**: Insert print statements or use logging module
5. **Check Environment Variables**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export ACCELERATE_DEBUG_MODE=1
   ```

## Common Breakpoint Locations

- Line 524: `main()` function start
- Line 587: Dataset loading
- Line 803: Model loading
- Line 874: Accelerator initialization
- Line 1090: Training loop start
- Line 1094: Forward pass through model

## Troubleshooting

### Issue: Debugger doesn't stop at breakpoints
- Ensure `justMyCode: false` in launch.json
- Check that the correct Python interpreter is selected
- Verify breakpoints are in executed code paths

### Issue: Multi-process debugging is complex
- Use `num_processes: 1` in accelerate config
- Or debug without accelerate using "Direct - No Accelerate" configuration

### Issue: CUDA out of memory during debugging
- Reduce batch size to 1
- Use fewer training samples
- Consider CPU debugging: `use_cpu: true`