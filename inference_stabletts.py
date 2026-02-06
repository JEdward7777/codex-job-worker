#!/usr/bin/env python3
"""
StableTTS Inference Script
Generates audio from text using trained StableTTS checkpoints

This module provides both a command-line interface and a Python API.

Python API Usage:
    from inference_stabletts import StableTTSInference, inference_stabletts_api

    # Using the API function
    result = inference_stabletts_api(
        checkpoint='/path/to/checkpoint.pt',
        input_csv='/path/to/input.csv',
        output_dir='/path/to/output',
        reference_audio='/path/to/reference.wav',
        language='english',
        heartbeat_callback=lambda: None  # Optional
    )

    # Or using the class directly
    inference = StableTTSInference(args)
    audio = inference.synthesize("Hello world")
"""
import os
import sys
import csv
import argparse
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import asdict
import traceback
from tqdm import tqdm

import torch
import torchaudio
# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from config import MelConfig, ModelConfig                           #pylint: disable=import-error, wrong-import-position
from models.model import StableTTS                                  #pylint: disable=import-error, wrong-import-position
from text import symbols, cleaned_text_to_sequence                  #pylint: disable=import-error, wrong-import-position
from text.mandarin import chinese_to_cnm3                           #pylint: disable=import-error, wrong-import-position
from text.english import english_to_ipa2                            #pylint: disable=import-error, wrong-import-position
from text.japanese import japanese_to_ipa2                          #pylint: disable=import-error, wrong-import-position
from datas.dataset import intersperse                               #pylint: disable=import-error, wrong-import-position
from utils.audio import LogMelSpectrogram, load_and_resample_audio  #pylint: disable=import-error, wrong-import-position
from api import get_vocoder                                         #pylint: disable=import-error, wrong-import-position

# Optional uroman import
try:
    from uroman import Uroman
    UROMAN_AVAILABLE = True
except ImportError:
    UROMAN_AVAILABLE = False

def sanitize_filename(text: str, max_length: int = 200) -> str:
    """Convert text to a safe filename by removing/replacing invalid characters."""
    # Remove or replace invalid filename characters
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text)
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Strip leading/trailing underscores and dots
    text = text.strip('_.')
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    # Ensure not empty
    if not text:
        text = 'unnamed'
    return text

def get_audio_format_params(audio_format: str) -> Dict[str, any]:
    """Get torchaudio.save parameters for the specified audio format.

    Args:
        audio_format: One of 'wav', 'ogg', 'mp3', 'webm'

    Returns:
        Dictionary with format-specific parameters for torchaudio.save

    Note:
        For WebM, TorchCodec determines format from file extension and only
        supports the 'compression' parameter (as bit_rate).
    """
    format_params = {
        'wav': {
            'format': 'wav',
            'encoding': 'PCM_S',  # 16-bit signed PCM
            'bits_per_sample': 16
        },
        'ogg': {
            'format': 'ogg',
            'encoding': 'OPUS',  # Opus codec in OGG container
            'compression': 5  # Compression level 0-10, 5 is balanced
        },
        'mp3': {
            'format': 'mp3',
            'compression': 128  # 128 kbps bitrate
        },
        'webm': {
            # TorchCodec determines format from .webm extension
            # Only compression (bit_rate) is supported
            'compression': 128000  # 128 kbps in bits per second for Opus
        }
    }

    if audio_format not in format_params:
        raise ValueError(f"Unsupported audio format: {audio_format}. Choose from: {list(format_params.keys())}")

    return format_params[audio_format]

class StableTTSInference:
    """Handles batch inference for StableTTS model."""

    def __init__(self, args):
        """Initialize inference engine with arguments."""
        self.args = args

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize configs
        self.mel_config = MelConfig()
        self.model_config = ModelConfig()

        # Setup language mapping
        self.g2p_mapping = {
            'chinese': chinese_to_cnm3,
            'japanese': japanese_to_ipa2,
            'english': english_to_ipa2,
        }

        # Setup uroman if enabled
        self.uroman = None
        if args.use_uroman:
            if UROMAN_AVAILABLE:
                self.uroman = Uroman()
                print(f"Uroman initialized for language: {args.uroman_lang}")
            else:
                raise RuntimeError("Uroman requested but not available. Install with: pip install uroman")

        # Load model and vocoder
        self.load_model()
        self.load_vocoder()

        # Load reference audio if provided
        self.reference_audio = None
        if args.reference_audio:
            print(f"Loading reference audio: {args.reference_audio}")
            self.reference_audio = self.load_reference_audio(args.reference_audio)

    def load_model(self):
        """Load the StableTTS model from checkpoint."""
        print(f"Loading StableTTS model from: {self.args.checkpoint}")

        self.mel_extractor = LogMelSpectrogram(**asdict(self.mel_config)).to(self.device)

        self.model = StableTTS(
            len(symbols),
            self.mel_config.n_mels,
            **asdict(self.model_config)
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(self.args.checkpoint, map_location='cpu', weights_only=True)
        )
        self.model.eval()
        print("Model loaded successfully")

    def load_vocoder(self):
        """Load the vocoder for mel-to-audio conversion."""
        # Use vocos as the primary vocoder (as referenced in StableTTS)
        vocoder_path = os.path.join(
            os.path.dirname(__file__),
            'StableTTS',
            'vocoders',
            'pretrained',
            'vocos.pt'
        )

        if not os.path.exists(vocoder_path):
            raise FileNotFoundError(
                f"Vocoder not found at {vocoder_path}. "
                "Please ensure StableTTS vocoders are properly installed. "
                "The Vocos vocoder can be downloaded from: "
                "https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt "
                "(HF Hub coordinates: repo_id='KdaiP/StableTTS1.1', filename='vocoders/vocos.pt')"
            )

        print(f"Loading vocoder from: {vocoder_path}")
        self.vocoder = get_vocoder(vocoder_path, 'vocos')
        self.vocoder.to(self.device)
        self.vocoder.eval()
        print("Vocoder loaded successfully")

    def load_reference_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess reference audio."""
        audio = load_and_resample_audio(audio_path, self.mel_config.sample_rate)
        if audio is None:
            raise ValueError(f"Failed to load reference audio: {audio_path}")
        audio = audio.to(self.device)
        mel = self.mel_extractor(audio)
        return mel

    def preprocess_text(self, text: str) -> str:
        """Preprocess text with uroman if enabled."""
        if self.uroman:
            return self.uroman.romanize_string(text, lcode=self.args.uroman_lang)
        return text

    def text_to_phonemes(self, text: str) -> str:
        """Convert text to phonemes using language-specific g2p."""
        phonemizer = self.g2p_mapping.get(self.args.language)
        if phonemizer is None:
            raise ValueError(f"Unsupported language: {self.args.language}")
        return phonemizer(text)

    @torch.inference_mode()
    def synthesize(self, text: str, reference_audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Synthesize audio from text.

        Args:
            text: Input text to synthesize
            reference_audio: Optional reference mel spectrogram for voice cloning

        Returns:
            Audio waveform as torch.Tensor
        """
        # Preprocess text with uroman if enabled
        text = self.preprocess_text(text)

        # Convert to phonemes
        phonemes = self.text_to_phonemes(text)

        if len(phonemes) == 0:
            raise ValueError(f"Empty phoneme sequence for text: {text}")

        # Convert phonemes to tensor
        phone_ids = cleaned_text_to_sequence(phonemes)
        phone_ids = intersperse(phone_ids, item=0)
        text_tensor = torch.tensor(phone_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        text_length = torch.tensor([text_tensor.size(-1)], dtype=torch.long, device=self.device)

        # Use provided reference or default
        ref_audio = reference_audio if reference_audio is not None else self.reference_audio

        if ref_audio is None:
            raise ValueError(
                "No reference audio provided. Please specify --reference_audio or provide "
                "a reference audio file for voice cloning."
            )

        # Synthesize mel spectrogram
        result = self.model.synthesise(
            text_tensor,
            text_length,
            n_timesteps=self.args.diffusion_steps,
            temperature=self.args.temperature,
            y=ref_audio,
            length_scale=self.args.length_scale,
            solver=self.args.solver,
            cfg=self.args.cfg_scale
        )

        mel_output = result['decoder_outputs']

        # Convert mel to audio using vocoder
        audio_output = self.vocoder(mel_output)

        return audio_output.cpu().squeeze()

    def process_batch(self, batch_texts: List[str]) -> List[torch.Tensor]:
        """Process a batch of texts (currently processes one at a time).

        Note: StableTTS synthesise method processes one sample at a time.
        This method is structured for future batch processing support.
        """
        results = []
        for text in batch_texts:
            audio = self.synthesize(text)
            results.append(audio)
        return results

    def run_inference(self):
        """Main inference loop."""
        # Validate output directory handling
        output_dir = Path(self.args.output_dir)
        audio_dir = output_dir / 'audio'
        metadata_path = output_dir / 'metadata.csv'

        # Check if output exists
        output_exists = metadata_path.exists() or (audio_dir.exists() and any(audio_dir.iterdir()))

        if output_exists:
            if not self.args.resume and not self.args.overwrite:
                raise ValueError(
                    "Output directory contains existing files. "
                    "Please specify either --resume or --overwrite flag."
                )

            if self.args.resume and self.args.overwrite:
                raise ValueError(
                    "Cannot specify both --resume and --overwrite flags. "
                    "Please choose one."
                )

        # Create directories
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Load input CSV
        print(f"\nLoading input CSV: {self.args.input_csv}")
        with open(self.args.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"No data found in input CSV: {self.args.input_csv}")

        # Validate text column exists
        if self.args.text_column not in rows[0]:
            raise ValueError(
                f"Text column '{self.args.text_column}' not found in CSV. "
                f"Available columns: {list(rows[0].keys())}"
            )

        print(f"Found {len(rows)} samples to process")
        print(f"Text column: {self.args.text_column}")
        print(f"Language: {self.args.language}")
        print(f"Batch size: {self.args.batch_size}")

        # Track statistics
        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        # Prepare output rows
        output_rows = []

        # Process in batches
        batch_texts = []
        batch_rows = []
        batch_indices = []

        for idx, row in enumerate(tqdm(rows, desc="Processing samples")):
            text = row.get(self.args.text_column, '').strip()

            if not text:
                print(f"\nWarning: Empty text at row {idx}, skipping")
                skipped_count += 1
                continue

            # Determine output filename
            file_extension = f'.{self.args.audio_format}'
            if 'verse_id' in row and row['verse_id']:
                filename = sanitize_filename(row['verse_id']) + file_extension
            elif 'file_name' in row and row['file_name']:
                #get only the filename and not the path.
                filename = os.path.basename(row['file_name'])
                #replace the extension with the specified audio format.
                filename = os.path.splitext(filename)[0] + file_extension
            else:
                filename = f"{idx}{file_extension}"

            output_path = audio_dir / filename

            # Handle resume mode
            if self.args.resume and output_path.exists():
                # File already exists, add to output and skip
                output_row = row.copy()
                output_row['file_name'] = f"audio/{filename}"
                output_rows.append(output_row)
                skipped_count += 1
                continue

            # Add to batch
            batch_texts.append(text)
            batch_rows.append(row)
            batch_indices.append((idx, filename, output_path))

            # Process batch when full
            if len(batch_texts) >= self.args.batch_size:
                audio_outputs = self.process_batch(batch_texts)

                # Get format-specific parameters
                format_params = get_audio_format_params(self.args.audio_format)

                # Determine target sample rate for Opus codec compatibility (WebM and OGG)
                target_sample_rate = self.mel_config.sample_rate
                if self.args.audio_format in ['webm', 'ogg'] and target_sample_rate not in [48000, 24000, 16000, 12000, 8000]:
                    # Resample to nearest supported rate (48000 is best quality)
                    target_sample_rate = 48000

                # Save audio files
                for audio, (_, filename, output_path), row in zip(audio_outputs, batch_indices, batch_rows):
                    # Debug: Print sample rate info
                    print(f"\nDEBUG: Saving {filename}")
                    print(f"  Audio format: {self.args.audio_format}")
                    print(f"  Original sample rate: {self.mel_config.sample_rate}")
                    print(f"  Target sample rate: {target_sample_rate}")
                    print(f"  Format params: {format_params}")

                    # Resample if needed
                    if target_sample_rate != self.mel_config.sample_rate:
                        print(f"  Resampling from {self.mel_config.sample_rate} to {target_sample_rate}")
                        audio_resampled = torchaudio.functional.resample(
                            audio.unsqueeze(0),
                            orig_freq=self.mel_config.sample_rate,
                            new_freq=target_sample_rate
                        )
                    else:
                        print("  No resampling needed")
                        audio_resampled = audio.unsqueeze(0)

                    torchaudio.save(
                        str(output_path),
                        audio_resampled,
                        target_sample_rate,
                        **format_params
                    )

                    # Add to output CSV
                    output_row = row.copy()
                    output_row['file_name'] = f"audio/{filename}"
                    output_rows.append(output_row)
                    processed_count += 1

                # Reset batch
                batch_texts = []
                batch_rows = []
                batch_indices = []

        # Process remaining batch
        if batch_texts:
            audio_outputs = self.process_batch(batch_texts)

            # Get format-specific parameters
            format_params = get_audio_format_params(self.args.audio_format)

            # Determine target sample rate for Opus codec compatibility (WebM and OGG)
            target_sample_rate = self.mel_config.sample_rate
            if self.args.audio_format in ['webm', 'ogg'] and target_sample_rate not in [48000, 24000, 16000, 12000, 8000]:
                # Resample to nearest supported rate (48000 is best quality)
                target_sample_rate = 48000

            for audio, (_, filename, output_path), row in zip(audio_outputs, batch_indices, batch_rows):
                # Debug: Print sample rate info
                print(f"\nDEBUG: Saving {filename}")
                print(f"  Audio format: {self.args.audio_format}")
                print(f"  Original sample rate: {self.mel_config.sample_rate}")
                print(f"  Target sample rate: {target_sample_rate}")
                print(f"  Format params: {format_params}")

                # Resample if needed
                if target_sample_rate != self.mel_config.sample_rate:
                    print(f"  Resampling from {self.mel_config.sample_rate} to {target_sample_rate}")
                    audio_resampled = torchaudio.functional.resample(
                        audio.unsqueeze(0),
                        orig_freq=self.mel_config.sample_rate,
                        new_freq=target_sample_rate
                    )
                else:
                    print("  No resampling needed")
                    audio_resampled = audio.unsqueeze(0)

                torchaudio.save(
                    str(output_path),
                    audio_resampled,
                    target_sample_rate,
                    **format_params
                )

                output_row = row.copy()
                output_row['file_name'] = f"audio/{filename}"
                output_rows.append(output_row)
                processed_count += 1

        # Save output CSV
        print(f"\nSaving output metadata to: {metadata_path}")
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            if output_rows:
                fieldnames = list(output_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Inference Complete!")
        print("="*60)
        print(f"Total samples: {len(rows)}")
        print(f"Processed: {processed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        if processed_count > 0:
            print(f"Average time per sample: {elapsed_time/processed_count:.2f} seconds")
        print(f"Output directory: {output_dir}")
        print(f"Audio files: {audio_dir}")
        print(f"Metadata: {metadata_path}")
        print("="*60)


def inference_stabletts_api(
    checkpoint: str,
    input_csv: str,
    output_dir: str,
    reference_audio: str,
    language: str = 'english',
    text_column: str = 'transcription',
    diffusion_steps: int = 10,
    temperature: float = 1.0,
    length_scale: float = 1.0,
    cfg_scale: float = 3.0,
    solver: Optional[str] = None,
    use_uroman: bool = False,
    uroman_lang: str = 'eng',
    batch_size: int = 1,
    audio_format: str = 'webm',
    resume: bool = False,
    overwrite: bool = True,
    heartbeat_callback: Optional[Callable[[], None]] = None
) -> Dict[str, Any]:
    """
    Python API for StableTTS inference.

    Args:
        checkpoint: Path to trained StableTTS checkpoint (.pt file)
        input_csv: Path to input CSV file with text to synthesize
        output_dir: Directory to save output audio and metadata
        reference_audio: Path to reference audio file for voice cloning
        language: Language for text-to-phoneme conversion
        text_column: Name of the column containing text to synthesize
        diffusion_steps: Number of diffusion steps for synthesis
        temperature: Sampling temperature (higher = more variation)
        length_scale: Speech pace control (higher = slower speech)
        cfg_scale: Classifier-free guidance scale
        solver: ODE solver for diffusion
        use_uroman: Enable uroman text romanization preprocessing
        uroman_lang: Language code for uroman romanization
        batch_size: Batch size for processing
        audio_format: Output audio format ('wav', 'ogg', 'mp3', 'webm')
        resume: Resume from existing output (skip already generated files)
        overwrite: Overwrite existing output files
        heartbeat_callback: Optional callback function to call periodically

    Returns:
        Dictionary with:
            - success: bool
            - processed_count: int
            - skipped_count: int
            - output_dir: str
            - metadata_csv: str
            - error_message: str (if success is False)
    """
    try:
        # Validate arguments
        if not os.path.exists(checkpoint):
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': f"Checkpoint not found at: {checkpoint}"
            }

        if not os.path.exists(input_csv):
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': f"Input CSV not found at: {input_csv}"
            }

        if not os.path.exists(reference_audio):
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': f"Reference audio not found at: {reference_audio}"
            }

        if use_uroman and not UROMAN_AVAILABLE:
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': "uroman requested but not installed"
            }

        # Create args namespace
        class Args:
            pass

        args = Args()
        args.checkpoint = checkpoint
        args.input_csv = input_csv
        args.output_dir = output_dir
        args.reference_audio = reference_audio
        args.language = language
        args.text_column = text_column
        args.diffusion_steps = diffusion_steps
        args.temperature = temperature
        args.length_scale = length_scale
        args.cfg_scale = cfg_scale
        args.solver = solver
        args.use_uroman = use_uroman
        args.uroman_lang = uroman_lang
        args.batch_size = batch_size
        args.audio_format = audio_format
        args.resume = resume
        args.overwrite = overwrite

        # Create inference engine
        inference = StableTTSInference(args)

        # Run inference with heartbeat support
        output_path = Path(output_dir)
        audio_dir = output_path / 'audio'
        metadata_path = output_path / 'metadata.csv'

        # Check if output exists
        output_exists = metadata_path.exists() or (audio_dir.exists() and any(audio_dir.iterdir()))

        if output_exists:
            if not resume and not overwrite:
                return {
                    'success': False,
                    'processed_count': 0,
                    'skipped_count': 0,
                    'output_dir': output_dir,
                    'metadata_csv': None,
                    'error_message': "Output directory contains existing files. Set resume=True or overwrite=True."
                }

        # Create directories
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Load input CSV
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': f"No data found in input CSV: {input_csv}"
            }

        # Validate text column exists
        if text_column not in rows[0]:
            return {
                'success': False,
                'processed_count': 0,
                'skipped_count': 0,
                'output_dir': output_dir,
                'metadata_csv': None,
                'error_message': f"Text column '{text_column}' not found in CSV. Available: {list(rows[0].keys())}"
            }

        # Process samples
        processed_count = 0
        skipped_count = 0
        output_rows = []

        for idx, row in enumerate(tqdm(rows, desc="Processing samples")):
            text = row.get(text_column, '').strip()

            if not text:
                skipped_count += 1
                continue

            # Determine output filename
            file_extension = f'.{audio_format}'
            if 'verse_id' in row and row['verse_id']:
                filename = sanitize_filename(row['verse_id']) + file_extension
            elif 'file_name' in row and row['file_name']:
                filename = os.path.basename(row['file_name'])
                filename = os.path.splitext(filename)[0] + file_extension
            else:
                filename = f"{idx}{file_extension}"

            file_output_path = audio_dir / filename

            # Handle resume mode
            if resume and file_output_path.exists():
                output_row = row.copy()
                output_row['file_name'] = f"audio/{filename}"
                output_rows.append(output_row)
                skipped_count += 1
                continue

            try:
                # Synthesize audio
                audio = inference.synthesize(text)

                # Get format-specific parameters
                format_params = get_audio_format_params(audio_format)

                # Determine target sample rate
                target_sample_rate = inference.mel_config.sample_rate
                if audio_format in ['webm', 'ogg'] and target_sample_rate not in [48000, 24000, 16000, 12000, 8000]:
                    target_sample_rate = 48000

                # Resample if needed
                if target_sample_rate != inference.mel_config.sample_rate:
                    audio_resampled = torchaudio.functional.resample(
                        audio.unsqueeze(0),
                        orig_freq=inference.mel_config.sample_rate,
                        new_freq=target_sample_rate
                    )
                else:
                    audio_resampled = audio.unsqueeze(0)

                # Save audio
                torchaudio.save(
                    str(file_output_path),
                    audio_resampled,
                    target_sample_rate,
                    **format_params
                )

                output_row = row.copy()
                output_row['file_name'] = f"audio/{filename}"
                output_rows.append(output_row)
                processed_count += 1

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                skipped_count += 1

            # Call heartbeat callback periodically
            if heartbeat_callback and idx % 10 == 0:
                heartbeat_callback()

        # Save output CSV
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            if output_rows:
                fieldnames = list(output_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

        return {
            'success': True,
            'processed_count': processed_count,
            'skipped_count': skipped_count,
            'output_dir': output_dir,
            'metadata_csv': str(metadata_path),
            'error_message': None
        }

    except Exception as e:

        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error during inference: {error_msg}")
        return {
            'success': False,
            'processed_count': 0,
            'skipped_count': 0,
            'output_dir': output_dir,
            'metadata_csv': None,
            'error_message': error_msg
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained StableTTS model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained StableTTS checkpoint (.pt file)')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file with text to synthesize')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output audio and metadata')

    # CSV configuration
    parser.add_argument('--text_column', type=str, default='transcription',
                        help='Name of the column containing text to synthesize')

    # Language configuration
    parser.add_argument('--language', type=str, default='english',
                        choices=['chinese', 'english', 'japanese'],
                        help='Language for text-to-phoneme conversion')

    # Reference audio
    parser.add_argument('--reference_audio', type=str, default=None,
                        help='Path to reference audio file for voice cloning (optional)')

    # Inference parameters (matching StableTTS defaults)
    parser.add_argument('--diffusion_steps', type=int, default=10,
                        help='Number of diffusion steps for synthesis')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more variation)')
    parser.add_argument('--length_scale', type=float, default=1.0,
                        help='Speech pace control (higher = slower speech)')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--solver', type=str, default=None,
                        help='ODE solver for diffusion (e.g., "dopri5", None for default)')

    # Uroman configuration
    parser.add_argument('--use_uroman', action='store_true',
                        help='Enable uroman text romanization preprocessing')
    parser.add_argument('--uroman_lang', type=str, default='eng',
                        help='Language code for uroman romanization')

    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (currently processes one at a time)')

    # Audio format configuration
    parser.add_argument('--audio_format', type=str, default='wav',
                        choices=['wav', 'ogg', 'mp3', 'webm'],
                        help='Output audio format (wav=uncompressed, ogg/webm=Opus codec, mp3=compressed)')

    # Output handling
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output (skip already generated files)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found at: {args.input_csv}")
        sys.exit(1)

    if args.reference_audio and not os.path.exists(args.reference_audio):
        print(f"Error: Reference audio not found at: {args.reference_audio}")
        sys.exit(1)

    if args.use_uroman and not UROMAN_AVAILABLE:
        print("Error: uroman requested but not installed. Install with: pip install uroman")
        sys.exit(1)

    # Print configuration
    print("\n" + "="*60)
    print("StableTTS Inference Configuration")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Text column: {args.text_column}")
    print(f"Language: {args.language}")
    print(f"Reference audio: {args.reference_audio or 'None (required)'}")
    print("\nInference Parameters:")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Length scale: {args.length_scale}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Solver: {args.solver or 'default'}")
    print(f"\nUroman: {'enabled' if args.use_uroman else 'disabled'}")
    if args.use_uroman:
        print(f"  Language code: {args.uroman_lang}")
    print(f"\nAudio format: {args.audio_format}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resume mode: {args.resume}")
    print(f"Overwrite mode: {args.overwrite}")
    print("="*60 + "\n")

    # Run inference
    inference = StableTTSInference(args)
    inference.run_inference()

if __name__ == '__main__':
    main()