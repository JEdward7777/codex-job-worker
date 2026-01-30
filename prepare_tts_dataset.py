#!/usr/bin/env python3
"""
Dataset Preparation Script for TTS Fine-tuning
Prepares HuggingFace dataset from preprocessed audio, metadata, and speaker IDs.
Creates train/validation splits for model training.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
import csv
import random
import shutil
import yaml
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Audio, Features, Value
import numpy as np
import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TTSDatasetPreparer:
    """Prepares HuggingFace dataset for TTS training."""

    def __init__(
        self,
        train_split: float = 0.9,
        val_split: float = 0.1,
        test_split: float = 0.0,
        shuffle: bool = True,
        seed: int = 42,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        """
        Initialize the dataset preparer.

        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for reproducibility
            min_duration: Minimum audio duration in seconds (filter out shorter)
            max_duration: Maximum audio duration in seconds (filter out longer)
        """
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.seed = seed
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Validate splits
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0, got {total}")

    def load_metadata(self, metadata_path: Path) -> List[Dict]:
        """
        Load metadata from CSV file.

        Args:
            metadata_path: Path to metadata.csv

        Returns:
            List of metadata dictionaries
        """
        logger.info("Loading metadata from %s", metadata_path)

        rows = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info("Loaded %s entries from metadata", len(rows))
        return rows

    def load_speaker_ids(self, speaker_ids_path: Path) -> Dict[str, int]:
        """
        Load speaker ID mapping from JSON file.

        Args:
            speaker_ids_path: Path to speaker_ids.json

        Returns:
            Dictionary mapping filename to speaker ID
        """
        logger.info("Loading speaker IDs from %s", speaker_ids_path)

        with open(speaker_ids_path, "r", encoding="utf-8") as f:
            speaker_ids = json.load(f)

        logger.info("Loaded speaker IDs for %s files", len(speaker_ids))
        logger.info("Unique speakers: %s", len(set(speaker_ids.values())))

        return speaker_ids

    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """
        Get audio duration in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds or None if failed
        """
        try:


            info = sf.info(str(audio_path))
            return info.duration
        except Exception as e:
            logger.warning("Could not get duration for %s: %s", audio_path, e)
            return None

    def prepare_dataset_dict(self, audio_dir: Path, metadata: List[Dict], speaker_ids: Dict[str, int]) -> List[Dict]:
        """
        Prepare dataset dictionary with all required fields.

        Args:
            audio_dir: Directory containing audio files
            metadata: List of metadata dictionaries
            speaker_ids: Dictionary mapping filename to speaker ID

        Returns:
            List of dataset examples
        """
        logger.info("Preparing dataset examples")

        examples = []
        skipped_no_audio = 0
        skipped_no_speaker = 0
        skipped_duration = 0

        for row in tqdm(metadata, desc="Processing examples"):
            filename = row["file_name"]
            transcription = row["transcription"]

            # Get audio path
            audio_path = audio_dir / filename

            # Check if audio file exists
            if not audio_path.exists():
                logger.warning("Audio file not found: %s", audio_path)
                skipped_no_audio += 1
                continue

            # Get speaker ID
            speaker_id = speaker_ids.get(filename)
            if speaker_id is None:
                logger.warning("No speaker ID for: %s", filename)
                skipped_no_speaker += 1
                continue

            # Get audio duration
            duration = self.get_audio_duration(audio_path)

            # Filter by duration if specified
            if duration is not None:
                if self.min_duration and duration < self.min_duration:
                    skipped_duration += 1
                    continue
                if self.max_duration and duration > self.max_duration:
                    skipped_duration += 1
                    continue

            # Create example
            example = {
                "audio": str(audio_path),
                "text": transcription,
                "speaker_id": speaker_id,
                "duration": duration,
                "file_name": filename,
                "audio_path": str(audio_path),  # Keep original path for later
            }

            examples.append(example)

        logger.info("Prepared %s examples", len(examples))
        if skipped_no_audio > 0:
            logger.warning("Skipped %s examples (audio file not found)", skipped_no_audio)
        if skipped_no_speaker > 0:
            logger.warning("Skipped %s examples (no speaker ID)", skipped_no_speaker)
        if skipped_duration > 0:
            logger.warning("Skipped %s examples (duration filter)", skipped_duration)

        return examples

    def create_splits(self, examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split examples into train/val/test sets.

        Args:
            examples: List of dataset examples

        Returns:
            Tuple of (train_examples, val_examples, test_examples)
        """

        # Shuffle if requested
        if self.shuffle:
            random.seed(self.seed)
            examples = examples.copy()
            random.shuffle(examples)

        # Calculate split indices
        n_total = len(examples)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        # Split data
        train_examples = examples[:n_train]
        val_examples = examples[n_train : n_train + n_val]
        test_examples = examples[n_train + n_val :]

        logger.info("Split sizes:")
        logger.info("  Train: %s (%.1f%%)", len(train_examples), len(train_examples) / n_total * 100)
        logger.info("  Validation: %s (%.1f%%)", len(val_examples), len(val_examples) / n_total * 100)
        if test_examples:
            logger.info("  Test: %s (%.1f%%)", len(test_examples), len(test_examples) / n_total * 100)

        return train_examples, val_examples, test_examples

    def create_huggingface_dataset(
        self, train_examples: List[Dict], val_examples: List[Dict], test_examples: Optional[List[Dict]] = None
    ) -> DatasetDict:
        """
        Create HuggingFace DatasetDict from examples.

        Args:
            train_examples: Training examples
            val_examples: Validation examples
            test_examples: Test examples (optional)

        Returns:
            HuggingFace DatasetDict
        """
        logger.info("Creating HuggingFace dataset")

        # Define features
        features = Features(
            {
                "audio": Audio(sampling_rate=16000),
                "text": Value("string"),
                "speaker_id": Value("int32"),
                "duration": Value("float32"),
                "file_name": Value("string"),
                "audio_path": Value("string"),  # Keep original path
            }
        )

        # Create datasets
        dataset_dict = {}

        if train_examples:
            train_dataset = Dataset.from_list(train_examples, features=features)
            dataset_dict["train"] = train_dataset
            logger.info("Created train dataset with %s examples", len(train_dataset))

        if val_examples:
            val_dataset = Dataset.from_list(val_examples, features=features)
            dataset_dict["validation"] = val_dataset
            logger.info("Created validation dataset with %s examples", len(val_dataset))

        if test_examples:
            test_dataset = Dataset.from_list(test_examples, features=features)
            dataset_dict["test"] = test_dataset
            logger.info("Created test dataset with %s examples", len(test_dataset))

        return DatasetDict(dataset_dict)

    def save_dataset(self, dataset: DatasetDict, output_dir: Path):
        """
        Save dataset in CSV + audio folder format (compatible with HuggingFace audiofolder).

        Args:
            dataset: HuggingFace DatasetDict
            output_dir: Output directory
        """

        logger.info("Saving dataset to %s in CSV + audio format", output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name, split_dataset in dataset.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create audio subdirectory
            audio_dir = split_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            # Prepare metadata rows
            metadata_rows = []

            logger.info("Saving %s split (%s examples)...", split_name, len(split_dataset))
            for example in tqdm(split_dataset, desc=f"Saving {split_name}"):
                # Use the stored audio_path
                source_audio = Path(example["audio_path"])
                dest_audio = audio_dir / source_audio.name

                if not dest_audio.exists() and source_audio.exists():
                    shutil.copy2(source_audio, dest_audio)

                # Add metadata row with audio/ prefix for the subdirectory
                metadata_rows.append(
                    {
                        "file_name": f"audio/{source_audio.name}",
                        "transcription": example["text"],
                        "speaker_id": example["speaker_id"],
                        "duration": example["duration"],
                    }
                )

            # Write metadata.csv
            metadata_path = split_dir / "metadata.csv"
            with open(metadata_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file_name", "transcription", "speaker_id", "duration"])
                writer.writeheader()
                writer.writerows(metadata_rows)

            logger.info("  Saved %s to %s", split_name, split_dir)
            logger.info("    Audio files: %s", len(metadata_rows))
            logger.info("    Metadata: %s", metadata_path)

        logger.info("\\nDataset saved successfully in CSV + audio format")

        # Print dataset info
        logger.info("\nDataset structure:")
        for split_name, split_dataset in dataset.items():
            logger.info("  %s: %s examples", split_name, len(split_dataset))
            logger.info("    Location: %s", output_dir / split_name)

    def print_statistics(self, dataset: DatasetDict):
        """
        Print dataset statistics.

        Args:
            dataset: HuggingFace DatasetDict
        """
        logger.info("\n%s", "=" * 60)
        logger.info("Dataset Statistics")
        logger.info("=" * 60)

        for split_name, split_dataset in dataset.items():
            logger.info("\\n%s SET:", split_name.upper())
            logger.info("  Total examples: %s", len(split_dataset))

            # Speaker statistics
            speaker_ids = [ex["speaker_id"] for ex in split_dataset]
            unique_speakers = len(set(speaker_ids))
            logger.info("  Unique speakers: %s", unique_speakers)

            # Duration statistics
            durations = [ex["duration"] for ex in split_dataset if ex["duration"] is not None]
            if durations:

                logger.info("  Total duration: %.2f hours", sum(durations) / 3600)
                logger.info("  Average duration: %.2f seconds", np.mean(durations))
                logger.info("  Min duration: %.2f seconds", min(durations))
                logger.info("  Max duration: %.2f seconds", max(durations))

            # Text length statistics
            text_lengths = [len(ex["text"]) for ex in split_dataset]
            if text_lengths:

                logger.info("  Average text length: %.1f characters", np.mean(text_lengths))
                logger.info("  Min text length: %s characters", min(text_lengths))
                logger.info("  Max text length: %s characters", max(text_lengths))


def main():
    parser = argparse.ArgumentParser(description="Prepare HuggingFace dataset for TTS fine-tuning")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file (e.g., config.yaml)")
    parser.add_argument("--audio_dir", type=str, help="Directory containing preprocessed audio files")
    parser.add_argument("--metadata", type=str, help="Path to metadata.csv file")
    parser.add_argument("--speaker_ids", type=str, help="Path to speaker_ids.json file")
    parser.add_argument("--output_dir", type=str, help="Output directory for HuggingFace dataset")
    parser.add_argument("--train_split", type=float, default=0.9, help="Training split fraction (default: 0.9)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    parser.add_argument("--test_split", type=float, default=0.0, help="Test split fraction (default: 0.0)")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load config if provided
    config = {}
    if args.config:
        logger.info("Loading configuration from %s", args.config)
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Get parameters from config or args
    if config:
        tts_config = config.get("tts", {})
        dataset_config = tts_config.get("dataset", {})
        train_split = args.train_split if args.train_split != 0.9 else dataset_config.get("train_split", 0.9)
        val_split = args.val_split if args.val_split != 0.1 else dataset_config.get("val_split", 0.1)
        test_split = args.test_split if args.test_split != 0.0 else dataset_config.get("test_split", 0.0)
        shuffle = not args.no_shuffle and dataset_config.get("shuffle", True)
        seed = args.seed if args.seed != 42 else dataset_config.get("seed", 42)
        min_duration = dataset_config.get("min_duration")
        max_duration = dataset_config.get("max_duration")

        # Get paths from config
        paths = tts_config.get("paths", {})
        audio_dir = Path(args.audio_dir or paths.get("preprocessed_audio", ""))
        metadata_path = Path(args.metadata or paths.get("preprocessed_metadata", ""))
        speaker_ids_path = Path(args.speaker_ids or paths.get("speaker_ids", ""))
        output_dir = Path(args.output_dir or paths.get("hf_dataset", ""))
    else:
        # Use command line arguments
        train_split = args.train_split
        val_split = args.val_split
        test_split = args.test_split
        shuffle = not args.no_shuffle
        seed = args.seed
        min_duration = None
        max_duration = None

        if not all([args.audio_dir, args.metadata, args.speaker_ids, args.output_dir]):
            logger.error(
                "Either --config or all of --audio_dir, --metadata, --speaker_ids, and --output_dir must be provided"
            )
            return 1

        audio_dir = Path(args.audio_dir)
        metadata_path = Path(args.metadata)
        speaker_ids_path = Path(args.speaker_ids)
        output_dir = Path(args.output_dir)

    # Validate inputs
    if not audio_dir.exists():
        logger.error("Audio directory does not exist: %s", audio_dir)
        return 1

    if not metadata_path.exists():
        logger.error("Metadata file does not exist: %s", metadata_path)
        return 1

    if not speaker_ids_path.exists():
        logger.error("Speaker IDs file does not exist: %s", speaker_ids_path)
        return 1

    # Create dataset preparer
    preparer = TTSDatasetPreparer(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        shuffle=shuffle,
        seed=seed,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    # Load metadata and speaker IDs
    metadata = preparer.load_metadata(metadata_path)
    speaker_ids = preparer.load_speaker_ids(speaker_ids_path)

    # Prepare dataset examples
    examples = preparer.prepare_dataset_dict(audio_dir, metadata, speaker_ids)

    if not examples:
        logger.error("No valid examples found")
        return 1

    # Create splits
    train_examples, val_examples, test_examples = preparer.create_splits(examples)

    # Create HuggingFace dataset
    dataset = preparer.create_huggingface_dataset(
        train_examples, val_examples, test_examples if test_examples else None
    )

    # Save dataset
    preparer.save_dataset(dataset, output_dir)

    # Print statistics
    preparer.print_statistics(dataset)

    logger.info("\nDataset preparation complete!")
    logger.info("Dataset saved to: %s", output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
