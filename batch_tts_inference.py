#!/usr/bin/env python3
"""
Batch TTS Inference Script
Generates audio for missing verses using a fine-tuned MMS-TTS model.
"""

import os
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
from tqdm import tqdm
import soundfile as sf

# Import transformers pipeline for TTS and datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

import torch

from uroman import Uroman

# Import find_best_checkpoint logic
from find_best_checkpoint import find_best_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BatchTTSInference:
    """Handles batch TTS generation for missing verses."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        logger.info("Reading configuration from: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.mv_config = self.config.get("tts", {}).get("missing_verses_generation", {})
        self.tts_config = self.config.get("tts", {})

        # Paths
        self.output_dir = Path(self.mv_config.get("output_dir", "generated_missing"))
        self.audio_dir = self.output_dir / self.mv_config.get("audio_dir", "audio")
        self.metadata_file = self.output_dir / self.mv_config.get("metadata_file", "metadata.csv")
        self.input_metadata = self.output_dir / "metadata.csv"  # Input from extraction step

        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.device = 0 if torch.cuda.is_available() else -1  # Pipeline uses 0 for GPU, -1 for CPU
        logger.info("Using device: %s", "cuda" if self.device == 0 else "cpu")

        # Uroman setup (Python version)
        self.use_uroman = self.config.get("uroman", {}).get("enabled", True)
        self.uroman_lang = self.config.get("uroman", {}).get("language", "hsn")

        if self.use_uroman:
            self.uroman = Uroman()
            logger.info("Uroman initialized for language:%s", self.uroman_lang)
        else:
            self.uroman = None
            if self.use_uroman:
                logger.warning("Uroman enabled in config but not available")

        # Load model using pipeline
        self.load_model()

    def load_model(self):
        """Load the TTS model using transformers pipeline."""
        model_path = self.mv_config.get("tts_model_path", "auto")

        # Resolve "auto" path
        if model_path == "auto":
            checkpoints_dir = self.tts_config.get("paths", {}).get("checkpoints", "")
            log_dir = os.path.join(checkpoints_dir, "runs")

            logger.info("Finding best checkpoint in %s...", log_dir)
            result = find_best_checkpoint(log_dir)

            if result and result["best_checkpoint"]:
                model_path = os.path.join(checkpoints_dir, result["best_checkpoint"])
                logger.info("Selected best checkpoint: %s", model_path)
            else:
                raise ValueError("Could not find best checkpoint automatically.")

        logger.info("Loading TTS pipeline from %s...", model_path)

        # Use the transformers pipeline for TTS (the recommended way)
        self.synthesiser = pipeline("text-to-speech", model=model_path, device=self.device)

        logger.info("TTS pipeline loaded successfully!")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using uroman if enabled (Python version)."""
        if self.uroman:
            return self.uroman.romanize_string(text, lcode=self.uroman_lang)
        return text

    def generate_audio_from_dataset(self, dataset: Dataset) -> List[Dict]:
        """Generate audio using HuggingFace Dataset format (eliminates pipeline warning).

        Args:
            dataset: HuggingFace Dataset with 'text' column

        Returns:
            List of results with audio arrays
        """
        try:
            # Preprocess texts with uroman if needed
            def preprocess_fn(batch):
                batch["processed_text"] = [self.preprocess_text(text) for text in batch["text"]]
                return batch

            dataset = dataset.map(preprocess_fn, batched=True, batch_size=len(dataset))

            # Use KeyDataset to properly pass the Dataset to the pipeline
            # This enables efficient DataLoader-based batching and eliminates the warning
            results = []
            for output in self.synthesiser(KeyDataset(dataset, "processed_text"), batch_size=len(dataset)):
                audio = output["audio"]

                # Handle different audio formats
                if isinstance(audio, np.ndarray):
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                else:
                    audio = np.array(audio).squeeze()

                results.append({"audio": audio})

            return results

        except Exception as e:
            logger.error("Dataset generation failed: %s", e)
            return [{"audio": None}] * len(dataset)

    def process_dataset(self):
        """Process the missing verses dataset using batch processing."""
        # Load input metadata (from extraction step)
        input_file = self.output_dir / "metadata.csv"
        if not input_file.exists():
            logger.error("Input metadata file not found: %s", input_file)
            return

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info("Found %s verses to process.", len(rows))

        # Configuration
        speaker_id = self.mv_config.get("speaker_id", 0)
        min_len = self.mv_config.get("min_text_length", 5)
        max_len = self.mv_config.get("max_text_length", 500)
        resume = self.mv_config.get("resume_from_checkpoint", True)
        batch_size = self.mv_config.get("batch_size", 8)  # Process 8 at a time

        # Get actual sample rate from model
        actual_sample_rate = self.synthesiser.model.config.sampling_rate

        # Prepare batches
        batches = []
        current_batch = []
        current_batch_rows = []

        success_count = 0
        skip_count = 0
        fail_count = 0
        output_rows = []

        # First pass: filter and prepare batches
        for row in rows:
            # Handle both input format (from extraction) and output format (already processed)
            text = row.get("text") or row.get("transcription", "")
            filename = row.get("filename") or row.get("file_name", "").replace("audio/", "")

            if not text or not filename:
                logger.warning("Skipping row with missing text or filename: %s", row)
                skip_count += 1
                continue
            filename_wav = filename.replace(".webm", ".wav")
            output_path = self.audio_dir / filename_wav

            # Skip if too short or too long
            if len(text) < min_len or len(text) > max_len:
                skip_count += 1
                continue

            # Resume check
            if resume and output_path.exists():
                duration = sf.info(str(output_path)).duration
                output_rows.append(
                    {
                        "file_name": f"audio/{filename_wav}",
                        "transcription": text,
                        "speaker_id": speaker_id,
                        "duration": duration,
                    }
                )
                skip_count += 1
                continue

            # Add to current batch
            current_batch.append(text)
            current_batch_rows.append((row, filename_wav, output_path))

            # Process batch when full
            if len(current_batch) >= batch_size:
                batches.append((current_batch, current_batch_rows))
                current_batch = []
                current_batch_rows = []

        # Add remaining items as final batch
        if current_batch:
            batches.append((current_batch, current_batch_rows))

        # Process batches using Dataset format (eliminates pipeline warning)
        logger.info("Processing %s batches of up to %s items each...", len(batches), batch_size)

        for batch_texts, batch_rows in tqdm(batches, desc="Generating Audio (batched)"):
            # Create a HuggingFace Dataset from the batch
            batch_dataset = Dataset.from_dict({"text": batch_texts})

            # Generate audio using Dataset format
            results = self.generate_audio_from_dataset(batch_dataset)

            # Save each audio file
            for result, (row, filename_wav, output_path) in zip(results, batch_rows):
                audio = result.get("audio")
                if audio is not None:
                    # Save audio
                    sf.write(str(output_path), audio, actual_sample_rate)
                    duration = len(audio) / actual_sample_rate

                    # Get text from either column name
                    text = row.get("text") or row.get("transcription", "")

                    output_rows.append(
                        {
                            "file_name": f"audio/{filename_wav}",
                            "transcription": text,
                            "speaker_id": speaker_id,
                            "duration": duration,
                        }
                    )
                    success_count += 1
                else:
                    logger.error("Failed to generate audio for %s", filename_wav)
                    fail_count += 1

        # Save updated metadata
        with open(input_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["file_name", "transcription", "speaker_id", "duration"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        logger.info("Generation complete.")
        logger.info("  Success: %s", success_count)
        logger.info("  Skipped: %s", skip_count)
        logger.info("  Failed:  %s", fail_count)
        logger.info("Metadata saved to %s", input_file)


def main():
    """Main entry point for batch TTS inference script.

    Parses command-line arguments and orchestrates the batch text-to-speech
    generation process for missing verses. Initializes the BatchTTSInference
    class with the provided configuration file and processes the dataset.

    Command-line Arguments:
        --config (str): Path to the YAML configuration file containing TTS
                       settings, model paths, and generation parameters.

    Example:
        $ python batch_tts_inference.py --config config.yaml
    """
    parser = argparse.ArgumentParser(description="Batch TTS Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    inference = BatchTTSInference(config_path=args.config)
    inference.process_dataset()


if __name__ == "__main__":
    main()
