#!/usr/bin/env python3
"""
StableTTS Checkpoint Comparison Interface
A Gradio-based UI for comparing different TTS model checkpoints
"""
import os
import sys
import csv
import argparse
import hashlib
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import asdict
import traceback
import gradio as gr

import torch
import torchaudio

# Add StableTTS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from config import MelConfig, ModelConfig                          #pylint: disable=import-error, wrong-import-position
from models.model import StableTTS                                 #pylint: disable=import-error, wrong-import-position
from text import symbols, cleaned_text_to_sequence                 #pylint: disable=import-error, wrong-import-position
from text.mandarin import chinese_to_cnm3                          #pylint: disable=import-error, wrong-import-position
from text.english import english_to_ipa2                           #pylint: disable=import-error, wrong-import-position
from text.japanese import japanese_to_ipa2                         #pylint: disable=import-error, wrong-import-position
from datas.dataset import intersperse                              #pylint: disable=import-error, wrong-import-position
from utils.audio import LogMelSpectrogram, load_and_resample_audio #pylint: disable=import-error, wrong-import-position
from api import get_vocoder                                        #pylint: disable=import-error, wrong-import-position

# Optional uroman import
try:
    from uroman import Uroman
    UROMAN_AVAILABLE = True
except ImportError:
    UROMAN_AVAILABLE = False


class CheckpointComparison:
    """Manages TTS inference for checkpoint comparison."""

    def __init__(self, args):
        """Initialize the comparison system."""
        self.args = args
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
                print("Uroman initialized")
            else:
                raise RuntimeError("Uroman requested but not available. Install with: pip install uroman")

        # Setup cache directory
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")

        # Initialize mel extractor (needed for reference audio loading)
        self.mel_extractor = LogMelSpectrogram(**asdict(self.mel_config)).to(self.device)

        # Load vocoder (shared across all checkpoints)
        self.load_vocoder()

        # Load default reference audio
        self.default_reference_mel = None
        if args.default_reference_audio:
            print(f"Loading default reference audio: {args.default_reference_audio}")
            self.default_reference_mel = self.load_reference_audio(args.default_reference_audio)

        # Discover checkpoints
        self.checkpoints = self.discover_checkpoints()
        print(f"Found {len(self.checkpoints)} checkpoints")

        # Load validation dataset
        self.validation_samples = self.load_validation_dataset()
        print(f"Loaded {len(self.validation_samples)} validation samples")

        # Cache for loaded models
        self.model_cache = {}

    def discover_checkpoints(self) -> List[Tuple[str, Path]]:
        """Discover and sort checkpoint files.

        Excludes the latest checkpoint if it's smaller than the previous one
        (indicating a partial/incomplete download).
        """
        checkpoint_dir = Path(self.args.checkpoints_dir)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

        checkpoints = []
        for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
            # Only include files that match checkpoint_*.pt pattern
            # Extract iteration number from filename
            match = re.search(r'checkpoint_(\d+)', file_path.stem)
            if match:
                iteration = int(match.group(1))
                file_size = file_path.stat().st_size
                checkpoints.append((iteration, file_path, file_size))

        # Sort by iteration number
        checkpoints.sort(key=lambda x: x[0])

        # Exclude latest checkpoint if it's smaller than the previous one
        if len(checkpoints) >= 2:
            latest_size = checkpoints[-1][2]
            previous_size = checkpoints[-2][2]
            if latest_size < previous_size:
                print(f"Warning: Excluding latest checkpoint {checkpoints[-1][1].name} "
                      f"(size: {latest_size:,} bytes) as it's smaller than previous "
                      f"checkpoint (size: {previous_size:,} bytes) - likely incomplete")
                checkpoints = checkpoints[:-1]

        # Return as (display_name, path) tuples
        return [(f"checkpoint_{iter}", path) for iter, path, _ in checkpoints]

    def load_validation_dataset(self) -> List[Dict]:
        """Load validation dataset from CSV."""
        if not self.args.validation_csv:
            return []

        csv_path = Path(self.args.validation_csv)
        if not csv_path.exists():
            print(f"Warning: Validation CSV not found: {csv_path}")
            return []

        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get the audio file path relative to CSV location
                audio_path = csv_path.parent / row[self.args.filename_column]
                samples.append({
                    'text': row[self.args.text_column],
                    'audio_path': audio_path,
                    'filename': row[self.args.filename_column]
                })

        return samples

    def load_vocoder(self):
        """Load the vocoder for mel-to-audio conversion.

        Looks for vocos.pt in the local StableTTS/vocoders/pretrained/ directory
        first.  If not found, auto-downloads from HuggingFace Hub.
        """
        vocoder_path = os.path.join(
            os.path.dirname(__file__),
            'StableTTS',
            'vocoders',
            'pretrained',
            'vocos.pt'
        )

        if not os.path.exists(vocoder_path):
            print(f"Vocoder not found at {vocoder_path}, downloading from HuggingFace Hub...")
            from handlers.base import download_pretrained_model  # pylint: disable=import-outside-toplevel
            vocoder_path = download_pretrained_model(
                repo_id='KdaiP/StableTTS1.1',
                filename='vocoders/vocos.pt',
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

    def load_model(self, checkpoint_path: Path) -> StableTTS:
        """Load a TTS model from checkpoint (with caching)."""
        checkpoint_str = str(checkpoint_path)

        if checkpoint_str in self.model_cache:
            return self.model_cache[checkpoint_str]

        print(f"Loading model from: {checkpoint_path}")
        model = StableTTS(
            len(symbols),
            self.mel_config.n_mels,
            **asdict(self.model_config)
        ).to(self.device)

        model.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        )
        model.eval()

        # Cache the model (limit cache size to avoid memory issues)
        if len(self.model_cache) >= 3:
            # Remove oldest entry
            self.model_cache.pop(next(iter(self.model_cache)))

        self.model_cache[checkpoint_str] = model
        return model

    def preprocess_text(self, text: str) -> str:
        """Preprocess text with uroman if enabled."""
        if self.uroman:
            return self.uroman.romanize_string(text, lcode=self.args.uroman_lang)
        return text

    def text_to_phonemes(self, text: str) -> str:
        """Convert text to phonemes using g2p."""
        phonemizer = self.g2p_mapping.get(self.args.g2p_type)
        if phonemizer is None:
            raise ValueError(f"Unsupported g2p type: {self.args.g2p_type}")
        return phonemizer(text)

    def get_cache_path(self, checkpoint_name: str, text: str, ref_audio_hash: str) -> Path:
        """Generate cache path for a specific generation."""
        cache_key = hashlib.md5(
            f"{checkpoint_name}_{text}_{ref_audio_hash}".encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.wav"

    @torch.inference_mode()
    def synthesize(
        self,
        checkpoint_path: Path,
        checkpoint_name: str,
        text: str,
        reference_mel: Optional[torch.Tensor] = None
    ) -> Path:
        """Synthesize audio from text using specified checkpoint.

        Returns path to generated audio file (cached or newly generated).
        """
        # Use default reference if none provided
        if reference_mel is None:
            reference_mel = self.default_reference_mel

        if reference_mel is None:
            raise ValueError("No reference audio provided")

        # Generate reference audio hash for caching
        ref_hash = hashlib.md5(reference_mel.cpu().numpy().tobytes()).hexdigest()[:8]

        # Check cache
        cache_path = self.get_cache_path(checkpoint_name, text, ref_hash)
        if cache_path.exists():
            print(f"Using cached audio: {cache_path}")
            return cache_path

        # Preprocess text
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

        # Load model
        model = self.load_model(checkpoint_path)

        # Synthesize mel spectrogram
        result = model.synthesise(
            text_tensor,
            text_length,
            n_timesteps=self.args.diffusion_steps,
            temperature=self.args.temperature,
            y=reference_mel,
            length_scale=self.args.length_scale,
            solver=self.args.solver,
            cfg=self.args.cfg_scale
        )

        mel_output = result['decoder_outputs']

        # Convert mel to audio using vocoder
        audio_output = self.vocoder(mel_output)
        audio_output = audio_output.cpu().squeeze()

        # Save to cache
        torchaudio.save(
            str(cache_path),
            audio_output.unsqueeze(0),
            self.mel_config.sample_rate
        )

        print(f"Generated and cached audio: {cache_path}")
        return cache_path


def create_gradio_interface(comparison_system: CheckpointComparison):
    """Create the Gradio interface."""

    def get_current_checkpoints():
        """Dynamically get current checkpoint list."""
        comparison_system.checkpoints = comparison_system.discover_checkpoints()
        return [name for name, _ in comparison_system.checkpoints]

    def get_validation_choices():
        """Get validation sample choices."""
        return [
            f"{i}: {sample['text'][:50]}..." if len(sample['text']) > 50 else f"{i}: {sample['text']}"
            for i, sample in enumerate(comparison_system.validation_samples)
        ]

    # Initial checkpoint names and validation choices
    checkpoint_names = get_current_checkpoints()
    validation_choices = get_validation_choices()

    # Get language for title
    language = comparison_system.args.title_language if hasattr(comparison_system.args, 'title_language') and comparison_system.args.title_language else None
    title = f"TTS Checkpoint Comparison - {language}" if language else "TTS Checkpoint Comparison"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown("Compare different model checkpoints to find the best performing one.")

        with gr.Tabs():
            # Tab 1: Free-form TTS
            with gr.Tab("Free-form TTS"):
                gr.Markdown("### Generate speech from custom text")

                with gr.Row():
                    with gr.Column():
                        free_text_input = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="Enter text here...",
                            lines=5
                        )
                        free_checkpoint = gr.Dropdown(
                            choices=checkpoint_names,
                            label="Select Checkpoint",
                            value=checkpoint_names[-1] if checkpoint_names else None
                        )
                        with gr.Accordion("Advanced Options", open=False):
                            free_reference = gr.Audio(
                                label="Reference Audio (optional - uses default if not provided)",
                                type="filepath"
                            )
                        free_generate_btn = gr.Button("Generate Audio", variant="primary")

                    with gr.Column():
                        free_output = gr.Audio(label="Generated Audio", type="filepath")
                        free_status = gr.Textbox(label="Status", interactive=False)

                def generate_free_tts(text, checkpoint_name, reference_audio):
                    """Generate TTS for free-form text."""
                    try:
                        if not text:
                            return None, "Error: Please enter text"

                        if not checkpoint_name:
                            return None, "Error: Please select a checkpoint"

                        # Find checkpoint path
                        checkpoint_path = None
                        for name, path in comparison_system.checkpoints:
                            if name == checkpoint_name:
                                checkpoint_path = path
                                break

                        if checkpoint_path is None:
                            return None, f"Error: Checkpoint not found: {checkpoint_name}"

                        # Load reference audio if provided
                        reference_mel = None
                        if reference_audio:
                            reference_mel = comparison_system.load_reference_audio(reference_audio)

                        # Generate audio
                        audio_path = comparison_system.synthesize(
                            checkpoint_path,
                            checkpoint_name,
                            text,
                            reference_mel
                        )

                        return str(audio_path), f"✓ Generated successfully using {checkpoint_name}"

                    except Exception as e:
                        return None, f"Error: {str(e)}"

                free_generate_btn.click( #pylint: disable=no-member
                    fn=generate_free_tts,
                    inputs=[free_text_input, free_checkpoint, free_reference],
                    outputs=[free_output, free_status]
                )

            # Tab 2: Validation Comparison (only shown if enabled)
            if comparison_system.args.enable_validation_tab:
                with gr.Tab("Validation Comparison"):
                    gr.Markdown("### Compare checkpoints on validation samples")

                    if not comparison_system.validation_samples:
                        gr.Markdown("⚠️ No validation dataset loaded. Please provide --validation-csv argument.")
                    else:
                        with gr.Row():
                            validation_selector = gr.Dropdown(
                                choices=validation_choices,
                                label="Select Validation Sample",
                                value=validation_choices[0] if validation_choices else None,
                                scale=4
                            )
                            refresh_btn = gr.Button("🔄 Refresh Checkpoints", scale=1)

                        # Ground truth section
                        gr.Markdown("### Ground Truth (Original Recording)")
                        ground_truth_text = gr.Textbox(label="Text", interactive=False)
                        ground_truth_audio = gr.Audio(label="Original Audio", type="filepath")

                        # Comparison rows - 12 slots for checkpoint comparisons
                        gr.Markdown("### Checkpoint Comparisons")
                        gr.Markdown("Select a checkpoint from each dropdown to generate audio for that slot.")

                        # Create 12 comparison slots
                        comparison_components = []
                        for i in range(12):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    dropdown = gr.Dropdown(
                                        choices=[""] + checkpoint_names,
                                        label=f"Slot {i+1} - Select Checkpoint",
                                        value="",
                                        interactive=True
                                    )
                                with gr.Column(scale=3):
                                    audio = gr.Audio(
                                        label=f"Generated Audio (Slot {i+1})",
                                        type="filepath",
                                        interactive=False
                                    )
                                    status = gr.Textbox(
                                        label="Status",
                                        value="No checkpoint selected",
                                        interactive=False,
                                        max_lines=1
                                    )
                            comparison_components.append((dropdown, audio, status))

                        def load_validation_sample(sample_choice):
                            """Load selected validation sample and clear all audio slots."""
                            if not sample_choice:
                                # Return empty values for text, audio, and reset all 12 slots
                                return ["", None] + ["", None, "No checkpoint selected"] * 12

                            # Extract index from choice
                            idx = int(sample_choice.split(":")[0])
                            sample = comparison_system.validation_samples[idx]

                            audio_path = sample['audio_path']
                            ground_truth = str(audio_path) if audio_path.exists() else None

                            # Return ground truth info + reset all 12 slots (dropdown, audio, status for each)
                            return [sample['text'], ground_truth] + ["", None, "No checkpoint selected"] * 12

                        def refresh_checkpoints():
                            """Refresh the checkpoint list and update all dropdowns."""
                            current_checkpoints = get_current_checkpoints()
                            choices = [""] + current_checkpoints
                            # Return updated choices for all 12 dropdowns
                            return [gr.update(choices=choices) for _ in range(12)]

                        def generate_audio_for_slot(sample_choice, checkpoint_name):
                            """Generate audio when a checkpoint is selected in a dropdown."""
                            try:
                                if not checkpoint_name or checkpoint_name == "":
                                    return None, "No checkpoint selected"

                                if not sample_choice:
                                    return None, "Error: No validation sample selected"

                                # Get sample text
                                idx = int(sample_choice.split(":")[0])
                                sample = comparison_system.validation_samples[idx]
                                text = sample['text']

                                # Find checkpoint path
                                checkpoint_path = None
                                for name, path in comparison_system.checkpoints:
                                    if name == checkpoint_name:
                                        checkpoint_path = path
                                        break

                                if checkpoint_path is None:
                                    return None, f"Error: Checkpoint not found: {checkpoint_name}"

                                # Generate audio
                                audio_path = comparison_system.synthesize(
                                    checkpoint_path,
                                    checkpoint_name,
                                    text,
                                    None  # Use default reference
                                )

                                return str(audio_path), f"✓ Generated with {checkpoint_name}"

                            except Exception as e:

                                error_msg = f"Error: {str(e)}"
                                print(f"Error in generate_audio_for_slot: {e}")
                                print(traceback.format_exc())
                                return None, error_msg

                        # Event handlers
                        # When validation sample changes, reset everything
                        validation_outputs = [ground_truth_text, ground_truth_audio]
                        for dropdown, audio, status in comparison_components:
                            validation_outputs.extend([dropdown, audio, status])

                        validation_selector.change( #pylint: disable=no-member
                            fn=load_validation_sample,
                            inputs=[validation_selector],
                            outputs=validation_outputs
                        )

                        # For each slot, when dropdown changes, generate audio
                        for dropdown, audio, status in comparison_components:
                            dropdown.change(
                                fn=generate_audio_for_slot,
                                inputs=[validation_selector, dropdown],
                                outputs=[audio, status]
                            )

                        # Refresh button updates all dropdowns
                        refresh_btn.click( #pylint: disable=no-member
                            fn=refresh_checkpoints,
                            outputs=[dropdown for dropdown, _, _ in comparison_components]
                        )

                        # Load first sample on startup and refresh checkpoints
                        demo.load( #pylint: disable=no-member
                            fn=load_validation_sample,
                            inputs=[validation_selector],
                            outputs=validation_outputs
                        ).then(
                            fn=refresh_checkpoints,
                            outputs=[dropdown for dropdown, _, _ in comparison_components]
                        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description='TTS Checkpoint Comparison Interface',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                        help='Directory containing checkpoint files')

    # Optional arguments
    parser.add_argument('--validation-csv', type=str, default=None,
                        help='Path to validation metadata CSV file')
    parser.add_argument('--filename-column', type=str, default='file_name',
                        help='Column name for audio filenames in CSV')
    parser.add_argument('--text-column', type=str, default='transcription',
                        help='Column name for text in CSV')
    parser.add_argument('--default-reference-audio', type=str, default=None,
                        help='Path to default reference audio file')
    parser.add_argument('--cache-dir', type=str, default='./gradio_cache',
                        help='Directory for caching generated audio')

    # Language and G2P configuration
    parser.add_argument('--title-language', type=str, default=None,
                        help='Language name to display in the interface title (e.g., "Chinese", "English", "Japanese")')
    parser.add_argument('--g2p-type', type=str, default='english',
                        choices=['chinese', 'english', 'japanese'],
                        help='Text-to-phoneme conversion type')

    # Uroman configuration
    parser.add_argument('--use-uroman', action='store_true',
                        help='Enable uroman text romanization preprocessing')
    parser.add_argument('--uroman-lang', type=str, default='eng',
                        help='ISO code for uroman romanization')

    # Inference parameters
    parser.add_argument('--diffusion-steps', type=int, default=10,
                        help='Number of diffusion steps for synthesis')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--length-scale', type=float, default=1.0,
                        help='Speech pace control')
    parser.add_argument('--cfg-scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--solver', type=str, default=None,
                        help='ODE solver for diffusion')

    # Server configuration
    parser.add_argument('--share', action='store_true',
                        help='Create a public Gradio link')
    parser.add_argument('--server-port', type=int, default=7860,
                        help='Port for Gradio server')
    parser.add_argument('--server-name', type=str, default='0.0.0.0',
                        help='Server name for Gradio')
    parser.add_argument('--enable-validation-tab', action='store_true',
                        help='Enable the Validation Comparison tab (requires --validation-csv)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.checkpoints_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoints_dir}")
        sys.exit(1)

    if args.validation_csv and not os.path.exists(args.validation_csv):
        print(f"Warning: Validation CSV not found: {args.validation_csv}")

    if args.default_reference_audio and not os.path.exists(args.default_reference_audio):
        print(f"Error: Default reference audio not found: {args.default_reference_audio}")
        sys.exit(1)

    if args.use_uroman and not UROMAN_AVAILABLE:
        print("Error: uroman requested but not installed. Install with: pip install uroman")
        sys.exit(1)

    # Print configuration
    print("\n" + "="*60)
    print("TTS Checkpoint Comparison Interface")
    print("="*60)
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Validation CSV: {args.validation_csv or 'None'}")
    print(f"Default reference audio: {args.default_reference_audio or 'None'}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Title Language: {args.title_language or 'Not specified'}")
    print(f"G2P type: {args.g2p_type}")
    print(f"Uroman: {'enabled' if args.use_uroman else 'disabled'}")
    if args.use_uroman:
        print(f"  ISO code: {args.uroman_lang}")
    print("\nInference Parameters:")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Length scale: {args.length_scale}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Solver: {args.solver or 'default'}")
    print("="*60 + "\n")

    # Initialize comparison system
    print("Initializing comparison system...")
    comparison_system = CheckpointComparison(args)

    # Create and launch Gradio interface
    print("Creating Gradio interface...")
    demo = create_gradio_interface(comparison_system)

    print(f"\nLaunching server on {args.server_name}:{args.server_port}")
    print("="*60 + "\n")

    # Add cache directory and validation audio directory to allowed paths
    allowed_paths = [str(comparison_system.cache_dir)]
    if args.validation_csv:
        validation_csv_dir = str(Path(args.validation_csv).parent)
        allowed_paths.append(validation_csv_dir)
    if args.default_reference_audio:
        ref_audio_dir = str(Path(args.default_reference_audio).parent)
        allowed_paths.append(ref_audio_dir)

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=allowed_paths
    )


if __name__ == '__main__':
    main()