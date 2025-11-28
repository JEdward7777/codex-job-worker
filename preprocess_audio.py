#!/usr/bin/env python3
"""
Audio Preprocessing Script for TTS Fine-tuning
Converts audio files to 16kHz, 16-bit, mono WAV format with normalization.
Supports multiple input formats including webm, mp3, wav, flac, ogg, m4a, etc.
Reads input metadata.csv and creates new output metadata.csv with preprocessed files.
"""

import os
import csv
import yaml
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import logging
import time
import warnings
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing for TTS training."""
    
    def __init__(
        self,
        target_sr: int = 16000,
        target_channels: int = 1,
        normalize: bool = True,
        target_bit_depth: int = 16
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sr: Target sample rate in Hz
            target_channels: Number of channels (1 for mono, 2 for stereo)
            normalize: Whether to normalize audio amplitude
            target_bit_depth: Target bit depth (16 or 24)
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.normalize = normalize
        self.target_bit_depth = target_bit_depth
        
        # Supported input formats
        self.supported_formats = {
            '.wav', '.mp3', '.flac', '.ogg', '.m4a', 
            '.webm', '.opus', '.aac', '.wma'
        }
    
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
        rms = np.sqrt(np.mean(audio ** 2))
        
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
    
    def process_audio_file(
        self,
        input_path: Path,
        output_path: Path,
        skip_existing: bool = True
    ) -> bool:
        """
        Process a single audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV file
            skip_existing: Skip if output file already exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if output exists and skip if requested
            if skip_existing and output_path.exists():
                logger.debug(f"Skipping existing file: {output_path}")
                return True
            
            # Load audio file using soundfile first, fall back to librosa if needed
            try:
                # Try soundfile first (faster and no deprecation warnings)
                audio, sr = sf.read(str(input_path))
                
                # Convert to mono if needed
                if self.target_channels == 1 and len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample if needed
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    
            except Exception:
                # Fall back to librosa for formats soundfile doesn't support (like webm)
                audio, sr = librosa.load(
                    str(input_path),
                    sr=self.target_sr,
                    mono=(self.target_channels == 1)
                )
            
            # Normalize if requested
            if self.normalize:
                audio = self.normalize_audio(audio)
            
            # Ensure audio is in correct range for bit depth
            if self.target_bit_depth == 16:
                subtype = 'PCM_16'
            elif self.target_bit_depth == 24:
                subtype = 'PCM_24'
            else:
                subtype = 'PCM_16'
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as WAV file
            sf.write(
                str(output_path),
                audio,
                self.target_sr,
                subtype=subtype
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            return False
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
        skip_existing: bool = True
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
            logger.warning(f"No audio files found in {input_dir}")
            return 0, 0
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        successful = 0
        failed = 0
        
        # Process each file with enhanced progress bar
        for input_path in tqdm(
            audio_files,
            desc="Processing audio files",
            unit="file",
            mininterval=0.5,  # Update display every 0.5 seconds
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            # Calculate relative path to maintain directory structure
            try:
                rel_path = input_path.relative_to(input_dir)
            except ValueError:
                rel_path = input_path.name
            
            # Change extension to .wav
            output_path = output_dir / rel_path.with_suffix('.wav')
            
            # Process the file
            if self.process_audio_file(input_path, output_path, skip_existing):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def process_with_metadata(
        self,
        input_metadata_path: Path,
        input_audio_dir: Path,
        output_dir: Path,
        skip_existing: bool = True
    ) -> tuple[int, int, Path]:
        """
        Process audio files based on metadata.csv file.
        
        Args:
            input_metadata_path: Path to input metadata.csv
            input_audio_dir: Directory containing input audio files
            output_dir: Output directory for processed files
            skip_existing: Skip files that already exist in output
            
        Returns:
            Tuple of (successful_count, failed_count, output_metadata_path)
        """
        # Create output directories
        output_audio_dir = output_dir / "audio"
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        output_metadata_path = output_dir / "metadata.csv"
        
        # Read input metadata
        logger.info(f"Reading metadata from {input_metadata_path}")
        rows = []
        with open(input_metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Found {len(rows)} entries in metadata")
        
        successful = 0
        failed = 0
        output_rows = []
        
        # Process each audio file with enhanced progress bar
        for row in tqdm(
            rows,
            desc="Processing audio files",
            unit="file",
            mininterval=0.5,  # Update display every 0.5 seconds
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            input_filename = row['file_name']
            transcription = row['transcription']
            
            # Find input file
            input_path = input_audio_dir / input_filename
            
            if not input_path.exists():
                logger.warning(f"Input file not found: {input_path}")
                failed += 1
                continue
            
            # Generate output filename (change extension to .wav)
            output_filename = Path(input_filename).stem + '.wav'
            output_path = output_audio_dir / output_filename
            
            # Process the file
            if self.process_audio_file(input_path, output_path, skip_existing):
                successful += 1
                # Add to output metadata
                output_rows.append({
                    'file_name': output_filename,
                    'transcription': transcription
                })
            else:
                failed += 1
        
        # Write output metadata
        logger.info(f"Writing output metadata to {output_metadata_path}")
        with open(output_metadata_path, 'w', newline='', encoding='utf-8') as f:
            if output_rows:
                writer = csv.DictWriter(f, fieldnames=['file_name', 'transcription'], quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(output_rows)
        
        return successful, failed, output_metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio files for TTS fine-tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (e.g., xiang_tts.yaml)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--input_metadata",
        type=str,
        help="Path to input metadata.csv file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for processed audio files"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of channels: 1=mono, 2=stereo (default: 1)"
    )
    parser.add_argument(
        "--bit_depth",
        type=int,
        default=16,
        choices=[16, 24],
        help="Target bit depth (default: 16)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize audio amplitude (default: True)"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable audio normalization"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Process subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already exist in output (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get parameters from config or args
    if config:
        audio_config = config.get('audio', {})
        sample_rate = audio_config.get('sample_rate', 16000)
        channels = audio_config.get('channels', 1)
        bit_depth = audio_config.get('bit_depth', 16)
        normalize = audio_config.get('normalize', True)
        
        # Get paths from config
        paths = config.get('paths', {})
        input_dir = Path(args.input_dir or paths.get('raw_audio', ''))
        input_metadata = Path(args.input_metadata or paths.get('raw_metadata', '')) if (args.input_metadata or paths.get('raw_metadata')) else None
        output_dir = Path(args.output_dir or paths.get('preprocessed', ''))
    else:
        # Use command line arguments
        sample_rate = args.sample_rate
        channels = args.channels
        bit_depth = args.bit_depth
        normalize = args.normalize and not args.no_normalize
        
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
        target_bit_depth=bit_depth
    )
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    logger.info(f"Processing audio files from {input_dir} to {output_dir}")
    logger.info(f"Settings: {sample_rate}Hz, {channels}ch, {bit_depth}-bit, normalize={normalize}")
    
    # Check if we have metadata file
    if input_metadata and input_metadata.exists():
        logger.info(f"Using metadata file: {input_metadata}")
        successful, failed, output_metadata = preprocessor.process_with_metadata(
            input_metadata,
            input_dir,
            output_dir,
            skip_existing=args.skip_existing
        )
        logger.info(f"Output metadata written to: {output_metadata}")
    else:
        if input_metadata:
            logger.warning(f"Metadata file not found: {input_metadata}")
        logger.info("Processing all audio files in directory")
        successful, failed = preprocessor.process_directory(
            input_dir,
            output_dir,
            recursive=args.recursive,
            skip_existing=args.skip_existing
        )
    
    # Print summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {successful + failed}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())