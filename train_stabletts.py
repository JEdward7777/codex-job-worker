#!/usr/bin/env python3
"""
Train StableTTS model with configurable parameters
Supports distributed multi-GPU training, validation, and checkpoint management
"""
import os
import sys
import json
import argparse
import warnings
from typing import Tuple, List

# Suppress warnings from third-party libraries
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
warnings.filterwarnings('ignore', category=SyntaxWarning)

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dataclasses import asdict

# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from datas.dataset import StableDataset, collate_fn
from datas.sampler import DistributedBucketSampler
from text import symbols
from config import MelConfig, ModelConfig, TrainConfig
from models.model import StableTTS
from utils.scheduler import get_cosine_schedule_with_warmup

torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

class StableDatasetWrapper(StableDataset):
    """Wrapper for StableDataset that resolves relative paths relative to JSON file location"""
    def _load_filelist(self, filelist_path):
        # Get the directory containing the JSON file for resolving relative paths
        json_dir = os.path.dirname(os.path.abspath(filelist_path))
        
        filelist, lengths = [], []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                mel_path = line['mel_path']
                # Resolve relative paths relative to the JSON file location
                if not os.path.isabs(mel_path):
                    mel_path = os.path.join(json_dir, mel_path)
                filelist.append((mel_path, line['phone']))
                lengths.append(line['mel_length'])
            
        self.filelist = filelist
        self.lengths = lengths  # length is used for DistributedBucketSampler

class SubsetWithLengths(Subset):
    """Subset that preserves the lengths attribute for DistributedBucketSampler"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Preserve lengths for the subset indices
        self.lengths = [dataset.lengths[i] for i in indices]

def split_dataset(dataset: StableDataset, val_split: float) -> Tuple[SubsetWithLengths, SubsetWithLengths]:
    """Split dataset into train and validation sets while preserving lengths attribute"""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Create indices for splitting
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = SubsetWithLengths(dataset, train_indices)
    val_dataset = SubsetWithLengths(dataset, val_indices)
    
    return train_dataset, val_dataset

def load_checkpoint(checkpoint_path: str, model: DDP, optimizer: optim.Optimizer, rank: int) -> int:
    """Load checkpoint from specific path"""
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return 0
    
    # Try to find corresponding optimizer checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_file = os.path.basename(checkpoint_path)
    
    # Extract epoch from checkpoint filename
    if '_' in checkpoint_file:
        name, epoch_str = checkpoint_file.rsplit('_', 1)
        epoch = int(epoch_str.split('.')[0])
        optimizer_path = os.path.join(checkpoint_dir, f'optimizer_{epoch}.pt')
        
        # Load model
        model.module.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # Load optimizer if exists
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
            if rank == 0:
                print(f'Loaded model and optimizer from epoch {epoch}')
            return epoch + 1
        else:
            if rank == 0:
                print(f'Loaded model from epoch {epoch} (optimizer not found)')
            return epoch + 1
    else:
        # Load model without epoch info
        model.module.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        if rank == 0:
            print(f'Loaded model from {checkpoint_path}')
        return 0

def auto_detect_checkpoint(checkpoint_dir: str, model: DDP, optimizer: optim.Optimizer, rank: int) -> int:
    """Auto-detect and load latest checkpoint from directory"""
    if not os.path.exists(checkpoint_dir):
        return 0
    
    model_dict = {}
    optimizer_dict = {}
    
    # Find all checkpoints in the directory
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pt") and '_' in file:
            name, epoch_str = file.rsplit('_', 1)
            try:
                epoch = int(epoch_str.split('.')[0])
                
                if name.startswith("checkpoint"):
                    model_dict[epoch] = file
                elif name.startswith("optimizer"):
                    optimizer_dict[epoch] = file
            except ValueError:
                continue
    
    # Get the largest epoch with both model and optimizer
    common_epochs = set(model_dict.keys()) & set(optimizer_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        model_path = os.path.join(checkpoint_dir, model_dict[max_epoch])
        optimizer_path = os.path.join(checkpoint_dir, optimizer_dict[max_epoch])
        
        # Load model and optimizer
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'))
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        
        if rank == 0:
            print(f'Auto-detected and loaded checkpoint from epoch {max_epoch}')
        return max_epoch + 1
    
    # If no matching pairs, try to load just the model
    elif model_dict:
        max_epoch = max(model_dict.keys())
        model_path = os.path.join(checkpoint_dir, model_dict[max_epoch])
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'))
        if rank == 0:
            print(f'Auto-detected and loaded model from epoch {max_epoch} (no optimizer)')
        return 0
    
    return 0

def validate(model: DDP, val_dataloader: DataLoader, rank: int, device: int) -> Tuple[float, float, float, float]:
    """Run validation and return average losses"""
    model.eval()
    
    # Use no_grad instead of inference_mode to avoid tensor tracking issues
    with torch.no_grad():
    
        total_dur_loss = 0.0
        total_diff_loss = 0.0
        total_prior_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for datas in val_dataloader:
            datas = [data.to(device, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas
            
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            
            total_dur_loss += dur_loss.item()
            total_diff_loss += diff_loss.item()
            total_prior_loss += prior_loss.item()
            total_loss += loss.item()
            num_batches += 1
    
        model.train()
    
        if num_batches > 0:
            return (total_dur_loss / num_batches,
                    total_diff_loss / num_batches,
                    total_prior_loss / num_batches,
                    total_loss / num_batches)
        else:
            return 0.0, 0.0, 0.0, 0.0

def train(rank, world_size, args):
    """Main training function"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Initialize configs
    model_config = ModelConfig()
    mel_config = MelConfig()
    
    # Create model save directory
    if rank == 0 and not os.path.exists(args.model_save_path):
        print(f'Creating checkpoint directory: {args.model_save_path}')
        os.makedirs(args.model_save_path, exist_ok=True)
    
    # Create log directory
    if rank == 0 and not os.path.exists(args.log_dir):
        print(f'Creating log directory: {args.log_dir}')
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize model
    model = StableTTS(len(symbols), mel_config.n_mels, **asdict(model_config)).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Load full dataset with path resolution
    full_dataset = StableDatasetWrapper(args.train_dataset_path, mel_config.hop_length)
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(full_dataset, args.val_split)
    
    if rank == 0:
        print(f"\nDataset split:")
        print(f"  Total samples: {len(full_dataset)}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
    
    # Create train dataloader with distributed sampler
    train_sampler = DistributedBucketSampler(
        train_dataset, 
        args.batch_size, 
        [32, 300, 400, 500, 600, 700, 800, 900, 1000], 
        num_replicas=world_size, 
        rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn, 
        persistent_workers=True
    )
    
    # Create validation dataloader (no distributed sampler needed for validation)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    # Initialize TensorBoard writer
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    warmup_steps = 200  # Fixed from TrainConfig
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_epochs * len(train_dataloader)
    )
    
    # Load pretrained model, checkpoint, or auto-detect
    current_epoch = 0
    if args.pretrained_model:
        # Load pretrained model for fine-tuning
        if rank == 0:
            print(f"Loading pretrained model from: {args.pretrained_model}")
        model.module.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
        if rank == 0:
            print("Pretrained model loaded successfully. Starting fine-tuning from epoch 0.")
        current_epoch = 0
    elif args.checkpoint:
        # Load specific checkpoint
        current_epoch = load_checkpoint(args.checkpoint, model, optimizer, rank)
    else:
        # Auto-detect latest checkpoint
        current_epoch = auto_detect_checkpoint(args.model_save_path, model, optimizer, rank)
    
    # Print training configuration
    if rank == 0:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Model Architecture:")
        print(f"  Hidden channels: {model_config.hidden_channels}")
        print(f"  Filter channels: {model_config.filter_channels}")
        print(f"  Attention heads: {model_config.n_heads}")
        print(f"  Encoder layers: {model_config.n_enc_layers}")
        print(f"  Decoder layers: {model_config.n_dec_layers}")
        print(f"\nTraining Parameters:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Number of epochs: {args.num_epochs}")
        print(f"  Starting epoch: {current_epoch}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Log interval: {args.log_interval}")
        print(f"  Save interval: {args.save_interval}")
        print(f"  Validation split: {args.val_split}")
        print(f"  Best metric: {args.best_metric}")
        print(f"\nPaths:")
        print(f"  Dataset: {args.train_dataset_path}")
        print(f"  Checkpoints: {args.model_save_path}")
        print(f"  Logs: {args.log_dir}")
        print(f"\nDistributed Training:")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Device: cuda:{rank}")
        print("="*60 + "\n")
    
    # Track best validation loss
    best_val_metric = float('inf')
    
    # Training loop
    model.train()
    for epoch in range(current_epoch, args.num_epochs):
        train_dataloader.batch_sampler.set_epoch(epoch)
        
        if rank == 0:
            dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
        else:
            dataloader = train_dataloader
        
        # Training phase
        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas
            
            optimizer.zero_grad()
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log training metrics
            if rank == 0 and batch_idx % args.log_interval == 0:
                steps = epoch * len(dataloader) + batch_idx
                writer.add_scalar("training/diff_loss", diff_loss.item(), steps)
                writer.add_scalar("training/dur_loss", dur_loss.item(), steps)
                writer.add_scalar("training/prior_loss", prior_loss.item(), steps)
                writer.add_scalar("training/total_loss", loss.item(), steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)
        
        # Validation phase
        if rank == 0 and len(val_dataset) > 0:
            val_dur_loss, val_diff_loss, val_prior_loss, val_total_loss = validate(
                model, val_dataloader, rank, rank
            )
            
            # Log validation metrics
            writer.add_scalar("validation/diff_loss", val_diff_loss, epoch)
            writer.add_scalar("validation/dur_loss", val_dur_loss, epoch)
            writer.add_scalar("validation/prior_loss", val_prior_loss, epoch)
            writer.add_scalar("validation/total_loss", val_total_loss, epoch)
            
            print(f"\nEpoch {epoch} Validation - "
                  f"Total: {val_total_loss:.4f}, "
                  f"Diff: {val_diff_loss:.4f}, "
                  f"Dur: {val_dur_loss:.4f}, "
                  f"Prior: {val_prior_loss:.4f}")
            
            # Check if this is the best model based on selected metric
            metric_map = {
                'total_loss': val_total_loss,
                'diff_loss': val_diff_loss,
                'dur_loss': val_dur_loss,
                'prior_loss': val_prior_loss
            }
            current_metric = metric_map[args.best_metric]
            
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                best_model_path = os.path.join(args.model_save_path, 'best_model.pt')
                best_optimizer_path = os.path.join(args.model_save_path, 'best_optimizer.pt')
                torch.save(model.module.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)
                print(f"  → New best model saved! ({args.best_metric}: {current_metric:.4f})")
        
        # Save regular checkpoints
        if rank == 0 and epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_{epoch}.pt')
            optimizer_path = os.path.join(args.model_save_path, f'optimizer_{epoch}.pt')
            torch.save(model.module.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"Checkpoint saved at epoch {epoch}")
    
    if rank == 0:
        writer.close()
        print("\nTraining completed!")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(
        description='Train StableTTS model with distributed multi-GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--train_dataset_path', type=str, required=True,
                        help='Path to preprocessed JSON filelist from preprocess_stabletts.py')
    parser.add_argument('--model_save_path', type=str, required=True,
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory for TensorBoard logs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--num_epochs', type=int, default=10000,
                        help='Total number of training epochs')
    
    # Monitoring parameters
    parser.add_argument('--log_interval', type=int, default=16,
                        help='Log training metrics every N batches')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    # Checkpoint management
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint to resume from (auto-detects latest if not provided)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model to fine-tune from (e.g., downloaded StableTTS checkpoint)')
    
    # Validation parameters
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of dataset to use for validation (0.0-1.0)')
    parser.add_argument('--best_metric', type=str, default='total_loss',
                        choices=['total_loss', 'diff_loss', 'dur_loss', 'prior_loss'],
                        help='Metric to track for saving best model')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.train_dataset_path):
        print(f"Error: Training dataset not found at: {args.train_dataset_path}")
        print("Please run preprocess_stabletts.py first to generate the dataset.")
        sys.exit(1)
    
    if args.val_split < 0.0 or args.val_split >= 1.0:
        print(f"Error: val_split must be between 0.0 and 1.0 (got {args.val_split})")
        sys.exit(1)
    
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"Error: Specified checkpoint not found at: {args.checkpoint}")
        sys.exit(1)
    
    if args.pretrained_model and not os.path.exists(args.pretrained_model):
        print(f"Error: Specified pretrained model not found at: {args.pretrained_model}")
        sys.exit(1)
    
    if args.checkpoint and args.pretrained_model:
        print("Error: Cannot specify both --checkpoint and --pretrained_model. Use one or the other.")
        sys.exit(1)
    
    # Start distributed training
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("Error: No CUDA devices available. This script requires GPU(s).")
        sys.exit(1)
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    main()