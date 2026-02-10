#!/usr/bin/env python3
"""
Train StableTTS model with configurable parameters
Supports distributed multi-GPU training, validation, and checkpoint management

This module provides both a command-line interface and a Python API.

Python API Usage:
    from train_stabletts import train_stabletts_api

    result = train_stabletts_api(
        train_dataset_path='/path/to/dataset.json',
        model_save_path='/path/to/checkpoints',
        log_dir='/path/to/logs',
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        pretrained_model='/path/to/pretrained.pt',  # Optional
        heartbeat_callback=lambda epoch: None  # Optional callback
    )
"""
import os
import sys
import json
import argparse
from typing import Tuple, Optional, Callable, Dict, Any

from dataclasses import asdict

import traceback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

# # Suppress warnings from third-party libraries
# warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
# warnings.filterwarnings('ignore', category=SyntaxWarning)



# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from datas.dataset import StableDataset, collate_fn          #pylint: disable=import-error, wrong-import-position
from datas.sampler import DistributedBucketSampler           #pylint: disable=import-error, wrong-import-position
from text import symbols                                     #pylint: disable=import-error, wrong-import-position
from config import MelConfig, ModelConfig                    #pylint: disable=import-error, wrong-import-position
from models.model import StableTTS                           #pylint: disable=import-error, wrong-import-position
from utils.scheduler import get_cosine_schedule_with_warmup  #pylint: disable=import-error, wrong-import-position

torch.backends.cudnn.benchmark = True


class _TeeWriter:
    """Write to both a file and the original stream. Used by child processes
    spawned via ``torch.multiprocessing.spawn`` to tee their stdout/stderr
    into the same log file that the parent process writes to."""

    def __init__(self, log_file, original_stream):
        self._log_file = log_file
        self._original = original_stream

    def write(self, message):
        self._log_file.write(message)
        self._log_file.flush()
        self._original.write(message)
        self._original.flush()

    def flush(self):
        self._log_file.flush()
        self._original.flush()


def periodic_progress(iterable, total=None, desc="", interval=100):
    """Drop-in replacement for ``tqdm`` that prints a status line every
    *interval* items instead of a continuous progress bar.  Useful when
    stdout is captured to a log file where progress-bar control characters
    create noise."""
    last = 0
    for i, item in enumerate(iterable):
        if i % interval == 0:
            total_str = f"/{total}" if total else ""
            print(f"  {desc}: {i}{total_str}")
        last = i
        yield item
    total_str = f"/{total}" if total else ""
    print(f"  {desc}: done ({last + 1}{total_str})")


class TrainingArgs:
    """Simple namespace for training arguments. Defined at module level so it
    can be pickled by ``torch.multiprocessing.spawn`` (spawn start method)."""
    pass


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
        _name, epoch_str = checkpoint_file.rsplit('_', 1)
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

def validate(model: DDP, val_dataloader: DataLoader, _rank: int, device: int) -> Tuple[float, float, float, float]:
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

# Global variables for multiprocessing (set before spawn, read by child)
_heartbeat_callback: Optional[Callable[[int], None]] = None
_use_tqdm: bool = True


def train(rank, world_size, args):
    """Main training function"""
    # Note: _heartbeat_callback and _use_tqdm are read-only here,
    # set by train_stabletts_api() before spawn.

    # --- Tee stdout/stderr to the log file (if provided) ---
    # torch.multiprocessing.spawn creates a new process, so the parent's
    # TeeLogger does not capture output here.  Re-open the same log file
    # in append mode so training prints appear in the log.
    _log_file_handle = None
    if rank == 0 and getattr(args, 'log_file_path', None):
        try:
            _log_file_handle = open(args.log_file_path, 'a', encoding='utf-8')
            sys.stdout = _TeeWriter(_log_file_handle, sys.__stdout__)
            sys.stderr = _TeeWriter(_log_file_handle, sys.__stderr__)
        except Exception as e:
            print(f"Warning: could not tee to log file: {e}", file=sys.__stderr__)

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
        print("\nDataset split:")
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
    writer = None  # Initialize for all ranks to avoid pylint warning
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
        print("Model Architecture:")
        print(f"  Hidden channels: {model_config.hidden_channels}")
        print(f"  Filter channels: {model_config.filter_channels}")
        print(f"  Attention heads: {model_config.n_heads}")
        print(f"  Encoder layers: {model_config.n_enc_layers}")
        print(f"  Decoder layers: {model_config.n_dec_layers}")
        print("\nTraining Parameters:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Number of epochs: {args.num_epochs}")
        print(f"  Starting epoch: {current_epoch}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Log interval: {args.log_interval}")
        print(f"  Save interval: {args.save_interval}")
        print(f"  Validation split: {args.val_split}")
        print(f"  Best metric: {args.best_metric}")
        print("\nPaths:")
        print(f"  Dataset: {args.train_dataset_path}")
        print(f"  Checkpoints: {args.model_save_path}")
        print(f"  Logs: {args.log_dir}")
        print("\nDistributed Training:")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Device: cuda:{rank}")
        print("="*60 + "\n")

    # Track best validation loss
    best_val_metric = float('inf')
    best_epoch = -1

    # Per-epoch metrics (written to a JSON file so the parent process can read them)
    metrics_path = os.path.join(args.model_save_path, '_epoch_metrics.json') if rank == 0 else None
    epoch_metrics_list = []

    # Training loop
    model.train()
    for epoch in range(current_epoch, args.num_epochs):
        train_dataloader.batch_sampler.set_epoch(epoch)

        # Choose progress wrapper based on _use_tqdm flag
        if rank == 0:
            if _use_tqdm:
                dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
            else:
                num_batches = len(train_dataloader)
                dataloader = periodic_progress(
                    train_dataloader, total=num_batches,
                    desc=f"Epoch {epoch}/{args.num_epochs}", interval=max(1, num_batches // 4)
                )
        else:
            dataloader = train_dataloader

        # Training phase — accumulate losses for per-epoch summary
        epoch_train_total = 0.0
        epoch_train_diff = 0.0
        epoch_train_dur = 0.0
        epoch_train_prior = 0.0
        epoch_num_batches = 0

        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas

            optimizer.zero_grad()
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accumulate for epoch average
            if rank == 0:
                epoch_train_total += loss.item()
                epoch_train_diff += diff_loss.item()
                epoch_train_dur += dur_loss.item()
                epoch_train_prior += prior_loss.item()
                epoch_num_batches += 1

            # Log training metrics to TensorBoard
            if writer is not None:
                if rank == 0 and batch_idx % args.log_interval == 0:
                    steps = epoch * len(train_dataloader) + batch_idx
                    writer.add_scalar("training/diff_loss", diff_loss.item(), steps)
                    writer.add_scalar("training/dur_loss", dur_loss.item(), steps)
                    writer.add_scalar("training/prior_loss", prior_loss.item(), steps)
                    writer.add_scalar("training/total_loss", loss.item(), steps)
                    writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)

        # Compute epoch-average training losses
        if rank == 0 and epoch_num_batches > 0:
            avg_train_total = epoch_train_total / epoch_num_batches
            avg_train_diff = epoch_train_diff / epoch_num_batches
            avg_train_dur = epoch_train_dur / epoch_num_batches
            avg_train_prior = epoch_train_prior / epoch_num_batches
        else:
            avg_train_total = avg_train_diff = avg_train_dur = avg_train_prior = 0.0

        # Validation phase
        val_total_loss = val_diff_loss = val_dur_loss = val_prior_loss = 0.0
        if rank == 0 and len(val_dataset) > 0:
            val_dur_loss, val_diff_loss, val_prior_loss, val_total_loss = validate(
                model, val_dataloader, rank, rank
            )

            # Log validation metrics
            if writer is not None:
                writer.add_scalar("validation/diff_loss", val_diff_loss, epoch)
                writer.add_scalar("validation/dur_loss", val_dur_loss, epoch)
                writer.add_scalar("validation/prior_loss", val_prior_loss, epoch)
                writer.add_scalar("validation/total_loss", val_total_loss, epoch)

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
                best_epoch = epoch
                best_model_path = os.path.join(args.model_save_path, 'best_model.pt')
                best_optimizer_path = os.path.join(args.model_save_path, 'best_optimizer.pt')
                torch.save(model.module.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)

        # Per-epoch summary line (always printed, regardless of tqdm setting)
        if rank == 0:
            lr = scheduler.get_last_lr()[0]
            best_marker = " ★" if best_epoch == epoch else ""
            print(
                f"Epoch {epoch}/{args.num_epochs} - "
                f"Train: {avg_train_total:.4f} (diff={avg_train_diff:.4f}, dur={avg_train_dur:.4f}, prior={avg_train_prior:.4f}) | "
                f"Val: {val_total_loss:.4f} (diff={val_diff_loss:.4f}, dur={val_dur_loss:.4f}, prior={val_prior_loss:.4f}) | "
                f"Best: epoch {best_epoch} ({best_val_metric:.4f}) | "
                f"LR: {lr:.2e}{best_marker}"
            )

            # Collect metrics for later retrieval
            epoch_metrics_list.append({
                'epoch': epoch,
                'train_total_loss': round(avg_train_total, 6),
                'train_diff_loss': round(avg_train_diff, 6),
                'train_dur_loss': round(avg_train_dur, 6),
                'train_prior_loss': round(avg_train_prior, 6),
                'val_total_loss': round(val_total_loss, 6),
                'val_diff_loss': round(val_diff_loss, 6),
                'val_dur_loss': round(val_dur_loss, 6),
                'val_prior_loss': round(val_prior_loss, 6),
            })

        # Save regular checkpoints
        if rank == 0 and epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_{epoch}.pt')
            optimizer_path = os.path.join(args.model_save_path, f'optimizer_{epoch}.pt')
            torch.save(model.module.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"  Checkpoint saved at epoch {epoch}")

        # Call heartbeat callback after each epoch (only on rank 0)
        if rank == 0:
            try:
                if _heartbeat_callback is not None:
                    _heartbeat_callback(epoch)
            except Exception as e:
                # If heartbeat fails (e.g., job canceled), stop training
                print(f"Heartbeat callback failed: {e}")
                cleanup()
                raise

    if rank == 0:
        # Write epoch metrics to JSON so the parent process can read them
        if metrics_path and epoch_metrics_list:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch_metrics': epoch_metrics_list,
                    'best_epoch': best_epoch,
                    'best_val_metric': round(best_val_metric, 6),
                    'best_metric_name': args.best_metric,
                }, f, indent=2)

        if writer is not None:
            writer.close()

        print(f"\nTraining completed! Best model from epoch {best_epoch} "
              f"({args.best_metric}: {best_val_metric:.4f})")

    # Close log file tee if we opened one
    if _log_file_handle is not None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file_handle.close()

    cleanup()


def train_stabletts_api(
    train_dataset_path: str,
    model_save_path: str,
    log_dir: str,
    num_epochs: int = 10000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    log_interval: int = 16,
    save_interval: int = 1,
    checkpoint: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    val_split: float = 0.1,
    best_metric: str = 'total_loss',
    heartbeat_callback: Optional[Callable[[int], None]] = None,
    log_file_path: Optional[str] = None,
    use_tqdm: bool = True,
) -> Dict[str, Any]:
    """
    Python API for training StableTTS model.

    Args:
        train_dataset_path: Path to preprocessed JSON filelist from preprocess_stabletts.py
        model_save_path: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        num_epochs: Total number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate for AdamW optimizer
        log_interval: Log training metrics every N batches
        save_interval: Save checkpoint every N epochs
        checkpoint: Path to specific checkpoint to resume from
        pretrained_model: Path to pretrained model to fine-tune from
        val_split: Fraction of dataset to use for validation (0.0-1.0)
        best_metric: Metric to track for saving best model
        heartbeat_callback: Optional callback function called after each epoch with epoch number
        log_file_path: Optional path to a log file.  The spawned child process
            will open this file in append mode and tee stdout/stderr into it,
            ensuring training output is captured even though
            ``torch.multiprocessing.spawn`` creates a separate process.
        use_tqdm: If True (default), use tqdm progress bars.  If False, use
            periodic text-based progress suitable for log files.

    Returns:
        Dictionary with:
            - success: bool
            - epochs_completed: int
            - best_model_path: str (path to best model checkpoint)
            - best_epoch: int (epoch number of the best model, or -1)
            - best_metric_value: float (best validation metric value)
            - epoch_metrics: list[dict] (per-epoch loss data)
            - error_message: str (if success is False)
    """
    global _heartbeat_callback, _use_tqdm

    try:
        # Validate arguments
        if not os.path.exists(train_dataset_path):
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': f"Training dataset not found at: {train_dataset_path}"
            }

        if val_split < 0.0 or val_split >= 1.0:
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': f"val_split must be between 0.0 and 1.0 (got {val_split})"
            }

        if checkpoint and not os.path.exists(checkpoint):
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': f"Specified checkpoint not found at: {checkpoint}"
            }

        if pretrained_model and not os.path.exists(pretrained_model):
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': f"Specified pretrained model not found at: {pretrained_model}"
            }

        if checkpoint and pretrained_model:
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': "Cannot specify both checkpoint and pretrained_model"
            }

        # Check for CUDA
        world_size = torch.cuda.device_count()
        if world_size == 0:
            return {
                'success': False,
                'epochs_completed': 0,
                'best_model_path': None,
                'best_epoch': -1,
                'best_metric_value': None,
                'epoch_metrics': [],
                'error_message': "No CUDA devices available. Training requires GPU(s)."
            }

        # Set globals for child processes
        _heartbeat_callback = heartbeat_callback
        _use_tqdm = use_tqdm

        # Create args namespace (TrainingArgs is at module level for pickling)
        args = TrainingArgs()
        args.train_dataset_path = train_dataset_path
        args.model_save_path = model_save_path
        args.log_dir = log_dir
        args.num_epochs = num_epochs
        args.batch_size = batch_size
        args.learning_rate = learning_rate
        args.log_interval = log_interval
        args.save_interval = save_interval
        args.checkpoint = checkpoint
        args.pretrained_model = pretrained_model
        args.val_split = val_split
        args.best_metric = best_metric
        args.log_file_path = log_file_path

        # Set thread limits (guard against being called after parallel work
        # has already started, e.g. when preprocess_stabletts ran a Pool first)
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        # Run training
        torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size)

        # Read epoch metrics written by the child process
        metrics_file = os.path.join(model_save_path, '_epoch_metrics.json')
        epoch_metrics = []
        best_epoch = -1
        best_metric_value = None
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            epoch_metrics = metrics_data.get('epoch_metrics', [])
            best_epoch = metrics_data.get('best_epoch', -1)
            best_metric_value = metrics_data.get('best_val_metric')

        # Check for best model
        best_model_path = os.path.join(model_save_path, 'best_model.pt')
        if not os.path.exists(best_model_path):
            # Fall back to latest checkpoint
            checkpoints = [f for f in os.listdir(model_save_path) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if checkpoints:
                # Get latest checkpoint
                latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                best_model_path = os.path.join(model_save_path, latest)
            else:
                best_model_path = None

        return {
            'success': True,
            'epochs_completed': num_epochs,
            'best_model_path': best_model_path,
            'best_epoch': best_epoch,
            'best_metric_value': best_metric_value,
            'epoch_metrics': epoch_metrics,
            'error_message': None
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error during training: {error_msg}")
        return {
            'success': False,
            'epochs_completed': 0,
            'best_model_path': None,
            'best_epoch': -1,
            'best_metric_value': None,
            'epoch_metrics': [],
            'error_message': error_msg
        }
    finally:
        _heartbeat_callback = None
        _use_tqdm = True


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