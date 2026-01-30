#!/usr/bin/env python3
"""
Audio Preprocessing Script for TTS Fine-tuning
Converts audio files to 16kHz, 16-bit, mono WAV format with normalization.
Supports multiple input formats including webm, mp3, wav, flac, ogg, m4a, etc.
Can also output Opus codec within Ogg container for efficient storage.
Reads input metadata.csv and creates new output metadata.csv with preprocessed files.
"""

import os
import csv
import warnings
import time
import logging
from typing import List, Optional
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing for TTS training."""

    def __init__(
        self,
        target_sr: int = 16000,
        target_channels: int = 1,
        normalize: bool = True,
        target_bit_depth: int = 16,
        trim_silence: bool = True,
        silence_threshold: float = 20.0,
        output_format: str = "wav",
        opus_bitrate: int = 64,
    ):
        """
        Initialize the audio preprocessor.

        Args:
            target_sr: Target sample rate in Hz
            target_channels: Number of channels (1 for mono, 2 for stereo)
            normalize: Whether to normalize audio amplitude
            target_bit_depth: Target bit depth (16 or 24)
            trim_silence: Whether to trim leading and trailing silence
            silence_threshold: Threshold in dB below reference for silence detection (default: 20.0)
            output_format: Output format - 'wav' or 'opus' (default: 'wav')
            opus_bitrate: Bitrate for Opus encoding in kbps (default: 64)
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.normalize = normalize
        self.target_bit_depth = target_bit_depth
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.output_format = output_format.lower()
        self.opus_bitrate = opus_bitrate

        # Validate output format
        if self.output_format not in ["wav", "opus"]:
            raise ValueError(f"Unsupported output format: {output_format}. Must be 'wav' or 'opus'")

        # Track silence trimming statistics
        self.trim_stats = []  # List of (input_path, output_path, silence_removed_seconds)

        # Supported input formats
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".aac", ".wma"}

    def trim_silence_from_audio(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """
        Trim leading and trailing silence from audio.

        Args:
            audio: Input audio array
            sr: Sample rate of the audio

        Returns:
            Tuple of (trimmed audio array, silence removed in seconds)
        """
        # Use librosa's trim function to remove silence
        # top_db: threshold in dB below reference to consider as silence
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=self.silence_threshold, frame_length=2048, hop_length=512)

        # Calculate silence removed
        original_duration = len(audio) / sr
        trimmed_duration = len(trimmed_audio) / sr
        silence_removed = original_duration - trimmed_duration

        if silence_removed > 0.1:  # Log if more than 0.1 seconds removed
            logger.debug(
                "Trimmed %.2fs of silence (original: %.2fs, trimmed: %.2fs)",
                silence_removed,
                original_duration,
                trimmed_duration,
            )

        return trimmed_audio, silence_removed

    def load_audio_with_fallback(self, audio_path: Path, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
        """
        Load audio file with fallback from soundfile to librosa.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (None = keep original)

        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            # Try soundfile first (faster and no deprecation warnings)
            audio, sr = sf.read(str(audio_path))

            # Convert to mono if needed
            if self.target_channels == 1 and len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if target_sr is not None and sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        except Exception:
            # Fall back to librosa for formats soundfile doesn't support
            audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=(self.target_channels == 1))

        return audio, sr

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            # Try soundfile first (faster)
            audio_info = sf.info(str(audio_path))
            return audio_info.duration
        except Exception:
            # Fall back to librosa
            audio, sr = librosa.load(str(audio_path), sr=None)
            return len(audio) / sr

    def normalize_audio(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level.

        Args:
            audio: Input audio array
            target_level: Target level in dB (default: -20.0 dB)

        Returns:
            Normalized audio array
        """
        # Calculate current RMS level
        rms = np.sqrt(np.mean(audio**2))

        if rms == 0:
            logger.warning("Audio has zero RMS, skipping normalization")
            return audio

        # Convert target level from dB to linear scale
        target_linear = 10 ** (target_level / 20.0)

        # Calculate scaling factor
        scaling_factor = target_linear / rms

        # Apply scaling
        normalized = audio * scaling_factor

        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val * 0.99

        return normalized

    def process_audio_file(self, input_path: Path, output_path: Path, skip_existing: bool = True) -> bool:
        """
        Process a single audio file.

        Args:
            input_path: Path to input audio file
            output_path: Path to output file (WAV or Opus/Ogg)
            skip_existing: Skip if output file already exists

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if output exists and skip if requested
            if skip_existing and output_path.exists():
                logger.debug("Skipping existing file: %s", output_path)
                return True

            # Load audio file using helper method with fallback
            audio, _sr = self.load_audio_with_fallback(input_path, self.target_sr)

            # Trim silence if requested (before normalization for better results)
            silence_removed = 0.0
            if self.trim_silence:
                audio, silence_removed = self.trim_silence_from_audio(audio, self.target_sr)
                # Track trimming statistics
                if silence_removed > 0:
                    self.trim_stats.append((str(input_path), str(output_path), silence_removed))

            # Normalize if requested
            if self.normalize:
                audio = self.normalize_audio(audio)

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save based on output format
            if self.output_format == "wav":
                # Ensure audio is in correct range for bit depth
                if self.target_bit_depth == 16:
                    subtype = "PCM_16"
                elif self.target_bit_depth == 24:
                    subtype = "PCM_24"
                else:
                    subtype = "PCM_16"

                # Save as WAV file
                sf.write(str(output_path), audio, self.target_sr, subtype=subtype)
            elif self.output_format == "opus":
                # Convert numpy array to AudioSegment for Opus encoding
                # pydub expects int16 samples
                audio_int16 = (audio * 32767).astype(np.int16)

                # Create AudioSegment
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.target_sr,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=self.target_channels,
                )

                # Export as Opus in Ogg container
                audio_segment.export(str(output_path), format="opus", codec="libopus", bitrate=f"{self.opus_bitrate}k")

            return True

        except Exception as e:
            logger.error("Error processing %s: %s", input_path, str(e))
            return False

    def process_directory(
        self, input_dir: Path, output_dir: Path, recursive: bool = True, skip_existing: bool = True
    ) -> tuple[int, int]:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            recursive: Process subdirectories recursively
            skip_existing: Skip files that already exist in output

        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Find all audio files
        audio_files = []

        if recursive:
            for ext in self.supported_formats:
                audio_files.extend(input_dir.rglob(f"*{ext}"))
        else:
            for ext in self.supported_formats:
                audio_files.extend(input_dir.glob(f"*{ext}"))

        if not audio_files:
            logger.warning("No audio files found in %s", input_dir)
            return 0, 0

        logger.info("Found %s audio files to process", len(audio_files))

        successful = 0
        failed = 0

        # Determine output extension based on format
        output_ext = ".wav" if self.output_format == "wav" else ".ogg"

        # Process each file with enhanced progress bar
        for input_path in tqdm(
            audio_files,
            desc="Processing audio files",
            unit="file",
            mininterval=0.5,  # Update display every 0.5 seconds
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            # Calculate relative path to maintain directory structure
            try:
                rel_path = input_path.relative_to(input_dir)
            except ValueError:
                rel_path = input_path.name

            # Change extension based on output format
            output_path = output_dir / rel_path.with_suffix(output_ext)

            # Process the file
            if self.process_audio_file(input_path, output_path, skip_existing):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def get_top_trimmed_files(self, n: int = 5) -> List[tuple]:
        """
        Get the top N files with the most silence trimmed.

        Args:
            n: Number of top files to return

        Returns:
            List of tuples (input_path, output_path, silence_removed_seconds)
        """
        # Sort by silence removed (descending)
        sorted_stats = sorted(self.trim_stats, key=lambda x: x[2], reverse=True)
        return sorted_stats[:n]

    def process_with_metadata(
        self,
        input_metadata_path: Path,
        input_audio_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
        max_std_devs: Optional[float] = None,
        min_chars_per_second: Optional[float] = None,
        max_chars_per_second: Optional[float] = None,
    ) -> tuple[int, int, int, Path]:
        """
        Process audio files based on metadata.csv file.

        Args:
            input_metadata_path: Path to input metadata.csv
            input_audio_dir: Directory containing input audio files
            output_dir: Output directory for processed files
            skip_existing: Skip files that already exist in output
            max_std_devs: Maximum standard deviations from median chars_per_second (None = no filtering)
            min_chars_per_second: Minimum acceptable chars_per_second (None = no minimum)
            max_chars_per_second: Maximum acceptable chars_per_second (None = no maximum)

        Returns:
            Tuple of (successful_count, failed_count, filtered_count, output_metadata_path)
        """
        # Create output directories
        output_audio_dir = output_dir / "audio"
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        output_metadata_path = output_dir / "metadata.csv"

        # Read input metadata
        logger.info("Reading metadata from %s", input_metadata_path)
        rows = []
        with open(input_metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info("Found %s entries in metadata", len(rows))

        successful = 0
        failed = 0
        output_rows = []

        # Process each audio file with enhanced progress bar
        for row in tqdm(
            rows,
            desc="Processing audio files",
            unit="file",
            mininterval=0.5,  # Update display every 0.5 seconds
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            input_filename = row["file_name"]
            transcription = row["transcription"]

            # Find input file
            input_path = input_audio_dir / input_filename

            if not input_path.exists():
                logger.warning("Input file not found: %s", input_path)
                failed += 1
                continue

            # Generate output filename (change extension based on output format)
            output_ext = ".wav" if self.output_format == "wav" else ".ogg"
            output_filename = Path(input_filename).stem + output_ext
            output_path = output_audio_dir / output_filename

            # Process the file
            if self.process_audio_file(input_path, output_path, skip_existing):
                successful += 1

                # Calculate text length
                text_length = len(transcription)

                # Calculate audio duration using helper method
                audio_duration = self.get_audio_duration(output_path)

                # Calculate characters per second
                chars_per_second = text_length / audio_duration if audio_duration > 0 else 0.0

                # Calculate relative path from metadata.csv to audio file
                # metadata.csv is in output_dir, audio files are in output_dir/audio/
                relative_audio_path = output_path.relative_to(output_dir)

                # Add to output metadata
                output_rows.append(
                    {
                        "file_name": str(relative_audio_path),
                        "transcription": transcription,
                        "text_length": text_length,
                        "audio_duration": audio_duration,
                        "chars_per_second": chars_per_second,
                    }
                )
            else:
                failed += 1

        # Track filtering statistics
        filtered_count = 0

        # Determine filtering method and bounds
        lower_bound = None
        upper_bound = None

        if max_std_devs is not None and output_rows:
            # Statistical filtering based on standard deviations from median
            logger.info("\\nApplying chars_per_second filter: max %s standard deviations from median", max_std_devs)

            # Extract chars_per_second values
            cps_values = [row["chars_per_second"] for row in output_rows]

            # Calculate median and standard deviation
            median_cps = np.median(cps_values)
            std_cps = np.std(cps_values)

            logger.info("Chars per second - Median: %.2f, Std Dev: %.2f", median_cps, std_cps)

            # Calculate acceptable range
            lower_bound = median_cps - (max_std_devs * std_cps)
            upper_bound = median_cps + (max_std_devs * std_cps)

        elif (min_chars_per_second is not None or max_chars_per_second is not None) and output_rows:
            # Direct min/max filtering
            logger.info("\nApplying chars_per_second filter with explicit bounds")

            lower_bound = min_chars_per_second if min_chars_per_second is not None else float("-inf")
            upper_bound = max_chars_per_second if max_chars_per_second is not None else float("inf")

        # Apply filtering if bounds were set
        if lower_bound is not None and upper_bound is not None and output_rows:
            logger.info("Acceptable range: [%.2f, %.2f]", lower_bound, upper_bound)

            # Filter rows and delete files outside the range
            filtered_rows = []
            removed_count = 0

            for row in output_rows:
                cps = row["chars_per_second"]

                if lower_bound <= cps <= upper_bound:
                    # Keep this row
                    filtered_rows.append(row)
                else:
                    # Remove the audio file using file_name to reconstruct path
                    try:
                        file_path = output_dir / row["file_name"]
                        if file_path.exists():
                            file_path.unlink()
                            logger.debug("Removed outlier file: %s (cps=%.2f)", file_path, cps)
                        removed_count += 1
                    except Exception as e:
                        logger.error("Error removing file %s: %s", file_path, e)

            logger.info("Filtered out %s files outside acceptable range", removed_count)
            logger.info("Kept %s files within acceptable range", len(filtered_rows))

            filtered_count = removed_count
            output_rows = filtered_rows

        # Write output metadata
        logger.info("Writing output metadata to %s", output_metadata_path)
        with open(output_metadata_path, "w", newline="", encoding="utf-8") as f:
            if output_rows:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["file_name", "transcription", "text_length", "audio_duration", "chars_per_second"],
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()
                writer.writerows(output_rows)

        return successful, failed, filtered_count, output_metadata_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files for TTS fine-tuning")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file (e.g., config.yaml)")
    parser.add_argument("--input_dir", type=str, help="Input directory containing audio files")
    parser.add_argument("--input_metadata", type=str, help="Path to input metadata.csv file")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed audio files")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)")
    parser.add_argument(
        "--channels", type=int, default=1, choices=[1, 2], help="Number of channels: 1=mono, 2=stereo (default: 1)"
    )
    parser.add_argument("--bit_depth", type=int, default=16, choices=[16, 24], help="Target bit depth (default: 16)")
    parser.add_argument(
        "--normalize", action="store_true", default=True, help="Normalize audio amplitude (default: True)"
    )
    parser.add_argument("--no_normalize", action="store_true", help="Disable audio normalization")
    parser.add_argument(
        "--trim_silence", action="store_true", default=True, help="Trim leading and trailing silence (default: True)"
    )
    parser.add_argument("--no_trim_silence", action="store_true", help="Disable silence trimming")
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=20.0,
        help="Threshold in dB below reference for silence detection (default: 20.0)",
    )
    parser.add_argument(
        "--recursive", action="store_true", default=True, help="Process subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already exist in output (default: True)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--max_std_devs",
        type=float,
        default=None,
        help="Filter files by chars_per_second: keep only files within N standard deviations from median (default: None, no filtering)",
    )
    parser.add_argument(
        "--min_chars_per_second",
        type=float,
        default=None,
        help="Minimum acceptable chars_per_second value (alternative to --max_std_devs)",
    )
    parser.add_argument(
        "--max_chars_per_second",
        type=float,
        default=None,
        help="Maximum acceptable chars_per_second value (alternative to --max_std_devs)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=["wav", "opus"],
        help="Output audio format: 'wav' for WAV files or 'opus' for Opus codec in Ogg container (default: wav)",
    )
    parser.add_argument(
        "--opus_bitrate",
        type=int,
        default=64,
        help="Bitrate for Opus encoding in kbps (default: 64). Only used when --output_format=opus",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        logger.info("Loading configuration from %s", args.config)
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Get parameters from config or args
    if config:
        tts_config = config.get("tts", {})
        audio_config = tts_config.get("audio", {})
        sample_rate = audio_config.get("sample_rate", 16000)
        channels = audio_config.get("channels", 1)
        bit_depth = audio_config.get("bit_depth", 16)
        normalize = audio_config.get("normalize", True)
        trim_silence = audio_config.get("trim_silence", True)
        silence_threshold = audio_config.get("silence_threshold", 20.0)
        output_format = audio_config.get("output_format", args.output_format)
        opus_bitrate = audio_config.get("opus_bitrate", args.opus_bitrate)

        # Get paths from config
        dataset_config = config.get("dataset", {})
        paths_config = config.get("paths", {})
        input_dir = Path(
            args.input_dir or str(Path(dataset_config.get("output_dir", "")) / dataset_config.get("audio_dir", "audio"))
        )
        input_metadata = (
            Path(
                args.input_metadata
                or str(Path(dataset_config.get("output_dir", "")) / dataset_config.get("csv_filename", "metadata.csv"))
            )
            if (args.input_metadata or (dataset_config.get("output_dir") and dataset_config.get("csv_filename")))
            else None
        )
        output_dir = Path(args.output_dir or paths_config.get("preprocessed", ""))
    else:
        # Use command line arguments
        sample_rate = args.sample_rate
        channels = args.channels
        bit_depth = args.bit_depth
        normalize = args.normalize and not args.no_normalize
        trim_silence = args.trim_silence and not args.no_trim_silence
        silence_threshold = args.silence_threshold
        output_format = args.output_format
        opus_bitrate = args.opus_bitrate

        if not args.input_dir or not args.output_dir:
            logger.error("Either --config or both --input_dir and --output_dir must be provided")
            return 1

        input_dir = Path(args.input_dir)
        input_metadata = Path(args.input_metadata) if args.input_metadata else None
        output_dir = Path(args.output_dir)

    # Create preprocessor
    preprocessor = AudioPreprocessor(
        target_sr=sample_rate,
        target_channels=channels,
        normalize=normalize,
        target_bit_depth=bit_depth,
        trim_silence=trim_silence,
        silence_threshold=silence_threshold,
        output_format=output_format,
        opus_bitrate=opus_bitrate,
    )

    # Validate input directory
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    logger.info("Processing audio files from %s to %s", input_dir, output_dir)
    logger.info(
        "Settings: %sHz, %sch, %s-bit, normalize=%s, trim_silence=%s",
        sample_rate,
        channels,
        bit_depth,
        normalize,
        trim_silence,
    )
    logger.info("Output format: %s", output_format.upper())
    if trim_silence:
        logger.info("Silence threshold: %s dB", silence_threshold)

    # Check if we have metadata file
    filtered_count = 0
    if input_metadata and input_metadata.exists():
        logger.info("Using metadata file: %s", input_metadata)
        if args.max_std_devs is not None:
            logger.info("Filtering enabled: max %s standard deviations from median chars_per_second", args.max_std_devs)
        elif args.min_chars_per_second is not None or args.max_chars_per_second is not None:
            filter_desc = []
            if args.min_chars_per_second is not None:
                filter_desc.append(f"min={args.min_chars_per_second}")
            if args.max_chars_per_second is not None:
                filter_desc.append(f"max={args.max_chars_per_second}")
            logger.info("Filtering enabled: chars_per_second %s", ", ".join(filter_desc))
        successful, failed, filtered_count, output_metadata = preprocessor.process_with_metadata(
            input_metadata,
            input_dir,
            output_dir,
            skip_existing=args.skip_existing,
            max_std_devs=args.max_std_devs,
            min_chars_per_second=args.min_chars_per_second,
            max_chars_per_second=args.max_chars_per_second,
        )
        logger.info("Output metadata written to: %s", output_metadata)
    else:
        if input_metadata:
            logger.warning("Metadata file not found: %s", input_metadata)
        logger.info("Processing all audio files in directory")
        successful, failed = preprocessor.process_directory(
            input_dir, output_dir, recursive=args.recursive, skip_existing=args.skip_existing
        )

    # Print summary
    logger.info("\\n%s", "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("%s", "=" * 80)
    logger.info("Successfully processed: %s", successful)
    logger.info("Failed to process: %s", failed)
    if filtered_count > 0:
        logger.info("Filtered out (chars_per_second outliers): %s", filtered_count)
        logger.info("Final dataset size: %s", successful - filtered_count)
    logger.info("Total files attempted: %s", successful + failed)
    logger.info("%s", "=" * 80)

    # Report top files with most silence trimmed
    if trim_silence and preprocessor.trim_stats:
        logger.info("\\n%s", "=" * 80)
        logger.info("TOP 5 FILES WITH MOST SILENCE TRIMMED:")
        logger.info("%s", "=" * 80)

        top_trimmed = preprocessor.get_top_trimmed_files(n=5)

        if top_trimmed:
            for i, (input_path, output_path, silence_removed) in enumerate(top_trimmed, 1):
                logger.info("\\n#%s - Trimmed %.2f seconds:", i, silence_removed)
                logger.info("  Original:  %s", input_path)
                logger.info("  Processed: %s", output_path)
        else:
            logger.info("No files had significant silence trimmed (< 0.01s)")

        logger.info("\\n%s", "=" * 80)
        logger.info("Total files with silence trimmed: %s", len(preprocessor.trim_stats))
        total_silence = sum(stat[2] for stat in preprocessor.trim_stats)
        logger.info("Total silence removed: %.2f seconds (%.2f minutes)", total_silence, total_silence / 60)
        logger.info("%s\\n", "=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
