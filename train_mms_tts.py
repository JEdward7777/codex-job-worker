#!/usr/bin/env python3
"""
MMS-TTS Fine-tuning Wrapper Script
Generates training config and calls the official finetune-hf-vits training script.
This wrapper bridges our dataset preparation pipeline with the official training code.
"""

import os
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSTrainingConfigGenerator:
    """Generates training config for finetune-hf-vits from our config format."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the config generator.
        
        Args:
            config_path: Path to our YAML config file (e.g., xiang_tts.yaml)
        """
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.language = self.config.get('language', {})
        self.paths = self.config.get('paths', {})
        self.training_config = self.config.get('training', {})
        self.dataset_config = self.config.get('dataset', {})
    
    def generate_training_config(
        self,
        output_path: Path,
        dataset_path: str,
        project_name: str = None,
        push_to_hub: bool = False,
        hub_model_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate training config in finetune-hf-vits format.
        
        Args:
            output_path: Path to save the generated config
            dataset_path: Path to the HuggingFace dataset
            project_name: Name for the training project
            push_to_hub: Whether to push model to HuggingFace Hub
            hub_model_id: Hub model ID if pushing
            
        Returns:
            Generated config dictionary
        """
        # Get language info
        language_name = self.language.get('name', 'unknown')
        language_code = self.language.get('code', 'unk')
        base_model = self.language.get('base_model', 'facebook/mms-tts-cmn')
        
        # Get training parameters
        num_epochs = self.training_config.get('num_epochs', 250)
        learning_rate = self.training_config.get('learning_rate', 1e-4)
        per_device_train_batch_size = self.training_config.get('per_device_train_batch_size', 8)
        per_device_eval_batch_size = self.training_config.get('per_device_eval_batch_size', 8)
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 4)
        
        # Get output directory
        output_dir = self.paths.get('checkpoints', './checkpoints')
        
        # Get dataset parameters
        min_duration = self.dataset_config.get('min_duration', 0.5)
        max_duration = self.dataset_config.get('max_duration', 30.0)
        
        # Generate project name if not provided
        if not project_name:
            project_name = f"mms_{language_code}_finetuning"
        
        # Build the config
        training_config = {
            # Project info
            "project_name": project_name,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id if hub_model_id else f"mms-tts-{language_code}-finetuned",
            "report_to": ["tensorboard"],  # Can add "wandb" if configured
            "overwrite_output_dir": True,
            "output_dir": output_dir,
            
            # Dataset info - using local dataset
            "dataset_name": str(dataset_path),
            "audio_column_name": "audio",
            "text_column_name": "text",
            "train_split_name": "train",
            "eval_split_name": "validation",
            "speaker_id_column_name": "speaker_id",
            "override_speaker_embeddings": True,  # Important for multi-speaker
            
            # Duration filters
            "max_duration_in_seconds": max_duration,
            "min_duration_in_seconds": min_duration,
            "max_tokens_length": 500,
            
            # Model
            "model_name_or_path": base_model,
            
            # Preprocessing
            "preprocessing_num_workers": 4,
            
            # Training
            "do_train": True,
            "num_train_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": self.training_config.get('gradient_checkpointing', True),
            "per_device_train_batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "adam_beta1": 0.8,
            "adam_beta2": 0.99,
            "warmup_ratio": self.training_config.get('warmup_ratio', 0.01) if self.training_config.get('warmup_ratio') else None,
            "warmup_steps": self.training_config.get('warmup_steps', 500) if not self.training_config.get('warmup_ratio') else None,
            "group_by_length": False,
            
            # Evaluation
            "do_eval": True,
            "eval_steps": self.training_config.get('eval_steps', 100),
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "max_eval_samples": 25,
            "do_step_schedule_per_epoch": True,
            
            # Loss weights (VITS-specific)
            "weight_disc": 3,
            "weight_fmaps": 1,
            "weight_gen": 1,
            "weight_kl": 1.5,
            "weight_duration": 1,
            "weight_mel": 35,
            
            # Mixed precision
            "fp16": self.training_config.get('mixed_precision', 'fp16') == 'fp16',
            "seed": self.dataset_config.get('seed', 42)
        }
        
        # Remove None values
        training_config = {k: v for k, v in training_config.items() if v is not None}
        
        # Save config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(training_config, f, indent=4)
        
        logger.info(f"Generated training config saved to: {output_path}")
        logger.info(f"  Project: {project_name}")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {per_device_train_batch_size}")
        logger.info(f"  Output: {output_dir}")
        
        return training_config


def setup_finetune_vits():
    """Setup the finetune-hf-vits repository."""
    finetune_dir = Path("finetune-hf-vits")
    
    if not finetune_dir.exists():
        logger.error(f"finetune-hf-vits directory not found at {finetune_dir}")
        logger.error("Please run: git submodule update --init --recursive")
        return False
    
    # Check if monotonic_align is built
    monotonic_align_dir = finetune_dir / "monotonic_align" / "monotonic_align"
    if not monotonic_align_dir.exists():
        logger.info("Building monotonic alignment search...")
        try:
            subprocess.run(
                ["python", "setup.py", "build_ext", "--inplace"],
                cwd=finetune_dir / "monotonic_align",
                check=True
            )
            logger.info("Monotonic alignment built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build monotonic alignment: {e}")
            return False
    
    return True


def run_training(config_path: Path, finetune_dir: Path):
    """
    Run the official finetune-hf-vits training script.
    
    Args:
        config_path: Path to the generated training config JSON
        finetune_dir: Path to finetune-hf-vits directory
    """
    training_script = finetune_dir / "run_vits_finetuning.py"
    
    if not training_script.exists():
        logger.error(f"Training script not found: {training_script}")
        return False
    
    logger.info("="*60)
    logger.info("Starting MMS-TTS Fine-tuning")
    logger.info("="*60)
    logger.info(f"Using config: {config_path}")
    logger.info(f"Training script: {training_script}")
    logger.info("")
    
    # Run training with accelerate
    cmd = ["accelerate", "launch", str(training_script), str(config_path)]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("")
    
    try:
        subprocess.run(cmd, check=True, cwd=finetune_dir)
        logger.info("\nTraining completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\nTraining failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MMS-TTS model using official finetune-hf-vits script"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., xiang_tts.yaml)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to prepared HuggingFace dataset (overrides config)"
    )
    parser.add_argument(
        "--output_config",
        type=str,
        default="./training_config.json",
        help="Path to save generated training config (default: ./training_config.json)"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="Project name for tracking"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace Hub model ID (required if --push_to_hub)"
    )
    parser.add_argument(
        "--generate_config_only",
        action="store_true",
        help="Only generate config without running training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load our config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    # Generate training config
    generator = TTSTrainingConfigGenerator(config_path)
    
    # Get dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = generator.paths.get('hf_dataset', '')
    
    if not dataset_path or not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        logger.error("Please run prepare_tts_dataset.py first or specify --dataset_path")
        return 1
    
    # Generate config
    output_config_path = Path(args.output_config)
    training_config = generator.generate_training_config(
        output_path=output_config_path,
        dataset_path=dataset_path,
        project_name=args.project_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    if args.generate_config_only:
        logger.info("\nConfig generation complete. Exiting without training.")
        logger.info(f"To train, run: accelerate launch finetune-hf-vits/run_vits_finetuning.py {output_config_path}")
        return 0
    
    # Setup finetune-hf-vits
    if not setup_finetune_vits():
        logger.error("Failed to setup finetune-hf-vits")
        return 1
    
    # Run training
    finetune_dir = Path("finetune-hf-vits")
    success = run_training(output_config_path, finetune_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())