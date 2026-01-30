#!/usr/bin/env python3
"""
W2V2-BERT ASR Gradio Demo

A Gradio-based web interface for real-time audio transcription using
fine-tuned Wav2Vec2-BERT or Wav2Vec2 models.

Features:
- File upload and microphone recording support
- Automatic transcription with confidence scores
- Support for multiple audio formats (WAV, MP3, FLAC, OGG, M4A)
- Optional example audio files
- Auto-detection of model architecture
- Optional post-processing with SentenceTransmogrifier for punctuation/capitalization

Usage:
    python gradio_demo_w2v2bert_asr.py \
        --model_path outputs/w2v2bert_asr/final_model \
        --language "English"
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch

# Import from inference script
from inference_w2v2bert_asr import (
    load_model_and_processor,
    transcribe_audio,
)

# Optional SentenceTransmogrifier import
try:
    from sentence_transmorgrifier.transmorgrify import Transmorgrifier

    TRANSMOGRIFIER_AVAILABLE = True
except ImportError:
    TRANSMOGRIFIER_AVAILABLE = False
    Transmorgrifier = None

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ASRDemo:
    """Manages ASR model and transcription for Gradio interface."""

    def __init__(self, args):
        """Initialize the ASR demo system."""
        self.args = args
        self.device = args.device

        logger.info("Using device: %s", self.device)
        logger.info("Loading model from: %s", args.model_path)

        # Load model and processor
        self.model, self.processor, self.use_wav2vec2_base = load_model_and_processor(
            args.model_path, self.device, args.use_wav2vec2_base
        )

        model_type = "Wav2Vec2" if self.use_wav2vec2_base else "Wav2Vec2-BERT"
        logger.info("Model loaded successfully: %s", model_type)

        # Load SentenceTransmogrifier if provided
        self.transmogrifier = None
        if args.tm_model:
            if not TRANSMOGRIFIER_AVAILABLE:
                raise RuntimeError(
                    "SentenceTransmogrifier model specified but library not available. "
                    "Install with: pip install sentence-transmorgrifier"
                )
            logger.info("Loading SentenceTransmogrifier from: %s", args.tm_model)
            self.transmogrifier = Transmorgrifier()
            self.transmogrifier.load(args.tm_model)
            logger.info("SentenceTransmogrifier loaded successfully")

        # Load example audio files if provided
        self.examples = self.load_examples()
        if self.examples:
            logger.info("Loaded %s example audio files", len(self.examples))

    def load_examples(self) -> list:
        """Load example audio files from directory."""
        if not self.args.examples_dir:
            return []

        examples_path = Path(self.args.examples_dir)
        if not examples_path.exists():
            logger.warning("Examples directory not found: %s", examples_path)
            return []

        # Supported audio extensions
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

        examples = []
        for ext in extensions:
            examples.extend(examples_path.glob(f"*{ext}"))
            examples.extend(examples_path.glob(f"*{ext.upper()}"))

        # Sort and limit to max_examples
        examples = sorted([str(f) for f in examples])
        if self.args.max_examples and len(examples) > self.args.max_examples:
            examples = examples[: self.args.max_examples]
            logger.info("Limited to %s example files", self.args.max_examples)

        return examples

    def transcribe(self, audio_path: Optional[str]) -> Tuple[str, str]:
        """
        Transcribe audio file and return results.

        Args:
            audio_path: Path to audio file (from upload or microphone)

        Returns:
            Tuple of (transcription_text, status_message)
        """
        if audio_path is None:
            return "", ""

        try:
            start_time = time.time()

            # Transcribe audio
            result = transcribe_audio(
                audio_path,
                self.model,
                self.processor,
                self.use_wav2vec2_base,
                self.device,
                return_confidence=True,
                return_alternatives=False,
                unk_token_replacement="*",
            )

            asr_time = time.time() - start_time

            # Check for errors
            if result.get("error"):
                error_msg = f"Error: {result['error']}"
                logger.error(error_msg)
                raise gr.Error(error_msg)

            transcription = result["transcription"]
            confidence = result.get("confidence")

            # Post-process with SentenceTransmogrifier if available
            if self.transmogrifier:
                tm_start = time.time()
                # Execute returns a generator, get the first result
                processed = list(self.transmogrifier.execute([transcription]))[0]
                tm_time = time.time() - tm_start

                logger.info("Post-processing: '%s' -> '%s'", transcription, processed)
                transcription = processed

                total_time = time.time() - start_time

                # Format status message with post-processing info
                if confidence is not None:
                    status = (
                        f"✓ Transcription complete (with post-processing)\n"
                        f"Confidence: {confidence:.2%}\n"
                        f"ASR time: {asr_time:.2f}s | Post-processing: {tm_time:.2f}s | Total: {total_time:.2f}s"
                    )
                else:
                    status = (
                        f"✓ Transcription complete (with post-processing)\n"
                        f"ASR time: {asr_time:.2f}s | Post-processing: {tm_time:.2f}s | Total: {total_time:.2f}s"
                    )
            else:
                # Format status message without post-processing
                if confidence is not None:
                    status = (
                        f"✓ Transcription complete\n"
                        f"Confidence: {confidence:.2%}\n"
                        f"Processing time: {asr_time:.2f}s"
                    )
                else:
                    status = f"✓ Transcription complete\nProcessing time: {asr_time:.2f}s"

            logger.info("Transcription successful: %s characters", len(transcription))

            return transcription, status

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise gr.Error(error_msg)


def create_gradio_interface(asr_demo: ASRDemo) -> gr.Blocks:
    """Create the Gradio interface."""

    # Determine title
    language = asr_demo.args.language
    title = f"ASR Demo - {language}" if language else "ASR Demo"

    # Create interface
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "Upload an audio file or record using your microphone to get an automatic transcription. "
            "Supports WAV, MP3, FLAC, OGG, and M4A formats."
        )

        with gr.Row():
            with gr.Column():
                # Audio input (supports both upload and microphone)
                audio_input = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])

                # Show examples if available
                if asr_demo.examples:
                    gr.Markdown("### Example Audio Files")
                    gr.Examples(
                        examples=[[ex] for ex in asr_demo.examples],
                        inputs=[audio_input],
                        label="Click an example to load it",
                    )

            with gr.Column():
                # Transcription output
                transcription_output = gr.Textbox(
                    label="Transcription", placeholder="Transcription will appear here...", lines=8, interactive=False
                )

                # Status output
                status_output = gr.Textbox(label="Status", placeholder="Ready", lines=3, interactive=False)

        # Auto-transcribe when audio is provided
        audio_input.change(  # pylint: disable=no-member
            fn=asr_demo.transcribe, inputs=[audio_input], outputs=[transcription_output, status_output]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="W2V2-BERT ASR Gradio Demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")

    # Optional arguments
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help='Language name to display in the interface title (e.g., "English", "Chinese")',
    )
    parser.add_argument("--examples-dir", type=str, default=None, help="Directory containing example audio files")
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum number of example files to display")
    parser.add_argument(
        "--tm_model",
        type=str,
        default=None,
        help="Path to SentenceTransmogrifier model for post-processing (adds punctuation/capitalization)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--use_wav2vec2_base",
        action="store_true",
        default=None,
        help="Force use of traditional Wav2Vec2 architecture (auto-detected if not specified)",
    )

    # Server configuration
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name for Gradio")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.model_path):
        logger.error("Model path not found: %s", args.model_path)
        sys.exit(1)

    if args.examples_dir and not os.path.exists(args.examples_dir):
        logger.warning("Examples directory not found: %s", args.examples_dir)

    if args.tm_model:
        if not os.path.exists(args.tm_model):
            logger.error("SentenceTransmogrifier model not found: %s", args.tm_model)
            sys.exit(1)
        if not TRANSMOGRIFIER_AVAILABLE:
            logger.error(
                "SentenceTransmogrifier requested but not installed. Install with: pip install sentence-transmorgrifier"
            )
            sys.exit(1)

    # Print configuration
    print("\n" + "=" * 60)
    print("W2V2-BERT ASR Gradio Demo")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Language: {args.language or 'Not specified'}")
    print(f"Device: {args.device}")
    print(f"Examples directory: {args.examples_dir or 'None'}")
    print(f"Post-processing model: {args.tm_model or 'None'}")
    print(f"Architecture override: {args.use_wav2vec2_base or 'Auto-detect'}")
    print("=" * 60 + "\n")

    # Initialize ASR demo
    logger.info("Initializing ASR demo...")
    asr_demo = ASRDemo(args)

    # Create Gradio interface
    logger.info("Creating Gradio interface...")
    demo = create_gradio_interface(asr_demo)

    # Launch
    logger.info("Launching server on %s:%s", args.server_name, args.server_port)
    print("=" * 60 + "\n")

    # Set up allowed paths for examples
    allowed_paths = []
    if args.examples_dir:
        allowed_paths.append(str(Path(args.examples_dir).resolve()))

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=allowed_paths if allowed_paths else None,
    )


if __name__ == "__main__":
    main()
