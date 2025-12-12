#!/usr/bin/env python3
"""
Script to analyze TensorBoard logs and find the checkpoint with the best validation loss.
"""

import os
import yaml
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def find_best_checkpoint(log_dir, val_loss_metric='val_loss_mel_kl'):
    """Find the checkpoint with the lowest validation loss from TensorBoard logs."""

    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        print("No TensorBoard event files found!")
        return None

    print(f"Found {len(event_files)} event files")

    best_loss = float('inf')
    best_step = None
    best_checkpoint = None

    for event_file in event_files:
        print(f"\nProcessing: {event_file}")

        try:
            # Load the event accumulator
            ea = EventAccumulator(event_file)
            ea.Reload()

            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']
            print(f"Available metrics: {scalar_tags}")

            # Look for validation loss (try multiple possible names)
            val_loss_tag = None
            for tag in [val_loss_metric, 'val_loss_mel_kl', 'eval/loss', 'val_loss', 'eval_loss']:
                if tag in scalar_tags:
                    val_loss_tag = tag
                    break

            if not val_loss_tag:
                print(f"No validation loss metric found. Available: {scalar_tags}")
                continue

            print(f"Using validation loss metric: {val_loss_tag}")

            # Get validation loss values
            val_loss_events = ea.Scalars(val_loss_tag)

            for event in val_loss_events:
                loss = event.value
                step = event.step

                print(f"Step {step}: validation loss = {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_step = step
                    # Calculate checkpoint number (assuming save_steps matches eval_steps from config)
                    checkpoint_num = step  # Step numbers match checkpoint numbers
                    best_checkpoint = f"checkpoint-{checkpoint_num}"

        except Exception as e:
            print(f"Error processing {event_file}: {e}")
            continue

    return {
        'best_checkpoint': best_checkpoint,
        'best_loss': best_loss,
        'best_step': best_step
    }

def main():
    parser = argparse.ArgumentParser(description="Find the best checkpoint from TensorBoard logs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config.yaml)"
    )
    parser.add_argument(
        "--val_loss_metric",
        type=str,
        default="val_loss_mel_kl",
        help="Name of the validation loss metric to use (default: val_loss_mel_kl)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get TensorBoard logs path from config
    paths = config.get('paths', {})
    checkpoints_dir = paths.get('checkpoints', '')
    if not checkpoints_dir:
        print("Error: 'paths.checkpoints' not found in config file")
        return 1

    log_dir = os.path.join(checkpoints_dir, 'runs')

    if not os.path.exists(log_dir):
        print(f"Error: TensorBoard logs directory not found: {log_dir}")
        return 1

    print("Analyzing TensorBoard logs to find best checkpoint...")
    print(f"Log directory: {log_dir}")
    print(f"Validation loss metric: {args.val_loss_metric}")

    result = find_best_checkpoint(log_dir, args.val_loss_metric)

    if result and result['best_checkpoint']:
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Best checkpoint: {result['best_checkpoint']}")
        print(f"Validation loss: {result['best_loss']:.4f}")
        print(f"Training step: {result['best_step']}")
        print(f"Full path: {os.path.join(checkpoints_dir, result['best_checkpoint'])}")
        print("="*60)

        # Update the config file with the best checkpoint path
        if 'missing_verses_generation' not in config:
            config['missing_verses_generation'] = {}

        config['missing_verses_generation']['tts_model_path'] = os.path.join(checkpoints_dir, result['best_checkpoint'])

        # Save updated config
        with open(args.config, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        print(f"\nUpdated {args.config} with best checkpoint path")

    else:
        print("Could not determine best checkpoint from logs")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())