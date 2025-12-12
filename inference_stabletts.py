#!/usr/bin/env python3
"""
StableTTS Inference Script
Generates audio from text using trained StableTTS checkpoints
"""
import os
import sys
import csv
import argparse
import time
import re
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

import torch
import torchaudio
from dataclasses import asdict

# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from config import MelConfig, ModelConfig
from models.model import StableTTS
from text import symbols, cleaned_text_to_sequence
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2
from datas.dataset import intersperse
from utils.audio import LogMelSpectrogram, load_and_resample_audio
from api import get_vocoder

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
                "Please ensure StableTTS vocoders are properly installed."
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
            if 'verse_id' in row and row['verse_id']:
                filename = sanitize_filename(row['verse_id']) + '.wav'
            elif 'file_name' in row and row['file_name']:
                #get only the filename and not the path.
                filename = os.path.basename(row['file_name'])
                #replace the extension with .wav if it is something else.
                filename = os.path.splitext(filename)[0] + '.wav'
            else:
                filename = f"{idx}.wav"
            
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
                
                # Save audio files
                for audio, (_, filename, output_path), row in zip(audio_outputs, batch_indices, batch_rows):
                    torchaudio.save(
                        str(output_path),
                        audio.unsqueeze(0),
                        self.mel_config.sample_rate
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
            
            for audio, (_, filename, output_path), row in zip(audio_outputs, batch_indices, batch_rows):
                torchaudio.save(
                    str(output_path),
                    audio.unsqueeze(0),
                    self.mel_config.sample_rate
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
    print(f"\nInference Parameters:")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Length scale: {args.length_scale}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Solver: {args.solver or 'default'}")
    print(f"\nUroman: {'enabled' if args.use_uroman else 'disabled'}")
    if args.use_uroman:
        print(f"  Language code: {args.uroman_lang}")
    print(f"\nBatch size: {args.batch_size}")
    print(f"Resume mode: {args.resume}")
    print(f"Overwrite mode: {args.overwrite}")
    print("="*60 + "\n")
    
    # Run inference
    inference = StableTTSInference(args)
    inference.run_inference()

if __name__ == '__main__':
    main()