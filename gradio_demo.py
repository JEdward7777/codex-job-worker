#!/usr/bin/env python3
"""
Gradio Demo Interface
Web interface to browse generated missing verses and generate custom audio.
"""

import os
import csv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import argparse
import yaml
import torch
import gradio as gr
import numpy as np
import soundfile as sf

# Import uroman for text normalization
try:
    from uroman import Uroman

    UROMAN_AVAILABLE = True
except ImportError:
    UROMAN_AVAILABLE = False

# Import transformers pipeline for TTS
from transformers import pipeline

# Import find_best_checkpoint logic
from find_best_checkpoint import find_best_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GradioDemo:
    """Gradio demo for TTS generation."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        logger.info("Reading configuration from: %s", config_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tts_config = self.config.get("tts", {})
        self.mv_config = self.tts_config.get("missing_verses_generation", {})
        self.demo_config = self.tts_config.get("demo", {})

        # Paths
        self.output_dir = Path(self.mv_config.get("output_dir", "generated_missing"))
        self.audio_dir = self.output_dir / self.mv_config.get("audio_dir", "audio")
        self.metadata_file = self.output_dir / self.mv_config.get("metadata_file", "metadata.csv")

        # Model configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        # Uroman setup
        self.uroman = Uroman() if UROMAN_AVAILABLE else None
        self.uroman_lang = self.config.get("uroman", {}).get("language", "hsn")

        # Load model
        self.load_model()

        # Load generated verses metadata
        self.verses = self.load_verses()

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
        device = 0 if self.device == "cuda" else -1  # Pipeline uses 0 for GPU, -1 for CPU
        self.synthesiser = pipeline("text-to-speech", model=model_path, device=device)

        logger.info("TTS pipeline loaded successfully!")

    def load_verses(self) -> List[Dict]:
        """Load generated verses metadata."""
        verses = []

        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add full path to audio
                    filename = row["file_name"]
                    # Handle "audio/" prefix if present
                    if filename.startswith("audio/"):
                        filename = filename.replace("audio/", "")

                    audio_path = self.audio_dir / filename
                    if audio_path.exists():
                        row["full_path"] = str(audio_path)
                        verses.append(row)

        logger.info("Loaded %s generated verses.", len(verses))
        return verses

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using uroman if enabled."""
        if self.uroman:
            return self.uroman.romanize_string(text, lcode=self.uroman_lang)
        return text

    def generate_audio(self, text: str) -> Tuple[int, np.ndarray]:
        """Generate audio for a single text input using the pipeline."""
        try:
            if not text.strip():
                return None

            # Preprocess text with uroman
            processed_text = self.preprocess_text(text)

            # Generate speech using the pipeline
            speech = self.synthesiser(processed_text)

            # Extract audio and sample rate
            audio = speech["audio"]
            sample_rate = speech["sampling_rate"]

            # Handle different audio formats
            if isinstance(audio, np.ndarray):
                if audio.ndim > 1:
                    audio = audio.squeeze()
            else:
                audio = np.array(audio).squeeze()

            return (sample_rate, audio)

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return None

    def get_books(self) -> List[str]:
        """Get list of unique books from verses."""
        books = set()
        for verse in self.verses:
            # Parse book from filename (e.g., "TIT-001-005.wav" -> "TIT")
            filename = verse["file_name"].replace("audio/", "")
            book = filename.split("-")[0]
            books.add(book)
        return sorted(list(books))

    def get_chapters(self, book: str) -> gr.Dropdown:
        """Get list of chapters for a given book."""
        if not book:
            return gr.Dropdown(choices=[], value=None)

        chapters = set()
        for verse in self.verses:
            filename = verse["file_name"].replace("audio/", "")
            parts = filename.split("-")
            if parts[0] == book:
                chapters.add(parts[1])

        chapter_list = sorted(list(chapters))
        return gr.Dropdown(choices=chapter_list, value=None)

    def get_verse_list(self, book: str, chapter: str) -> gr.Dropdown:
        """Get list of verse identifiers for a chapter."""
        if not book or not chapter:
            return gr.Dropdown(choices=[], value=None)

        # Find all verses for this book/chapter
        chapter_verses = []
        for verse in self.verses:
            filename = verse["file_name"].replace("audio/", "")
            parts = filename.split("-")
            if len(parts) >= 3 and parts[0] == book and parts[1] == chapter:
                verse_num = parts[2].replace(".wav", "")
                chapter_verses.append(f"{book}-{chapter}-{verse_num}")

        # Sort by verse number
        chapter_verses.sort()
        return gr.Dropdown(choices=chapter_verses, value=None)

    def get_verse_details(self, book: str, chapter: str, verse_id: str) -> Tuple[Optional[str], str]:
        """Get audio path and transcription for a specific verse."""
        if not verse_id:
            return None, ""

        # Find the verse
        for verse in self.verses:
            filename = verse["file_name"].replace("audio/", "")
            parts = filename.split("-")
            if len(parts) >= 3:
                verse_num = parts[2].replace(".wav", "")
                current_id = f"{parts[0]}-{parts[1]}-{verse_num}"
                if current_id == verse_id:
                    return verse["full_path"], verse["transcription"]

        return None, ""

    def launch(self):
        """Launch the Gradio interface."""

        # Get list of books
        books = self.get_books()
        logger.info("Found %s books: %s", len(books), books)

        with gr.Blocks(title="TTS Demo") as demo:
            gr.Markdown("# TTS Generation Demo")

            with gr.Tab("Browse Generated Verses"):
                gr.Markdown("Select a book, chapter, and verse:")

                with gr.Row():
                    book_dropdown = gr.Dropdown(choices=books, label="Select Book", interactive=True, value=None)
                    chapter_dropdown = gr.Dropdown(choices=[], label="Select Chapter", interactive=True, value=None)
                    verse_dropdown = gr.Dropdown(choices=[], label="Select Verse", interactive=True, value=None)

                with gr.Row():
                    verse_text = gr.Textbox(label="Text", interactive=False, lines=5)
                    audio_player = gr.Audio(label="Audio", type="filepath")

                # Update chapters when book is selected
                book_dropdown.change(fn=self.get_chapters, inputs=[book_dropdown], outputs=[chapter_dropdown]) #pylint: disable=no-member

                # Update verse list when chapter is selected
                chapter_dropdown.change( #pylint: disable=no-member
                    fn=self.get_verse_list, inputs=[book_dropdown, chapter_dropdown], outputs=[verse_dropdown]
                )

                # Update audio and text when verse is selected
                verse_dropdown.change( #pylint: disable=no-member
                    fn=self.get_verse_details,
                    inputs=[book_dropdown, chapter_dropdown, verse_dropdown],
                    outputs=[audio_player, verse_text],
                )

            with gr.Tab("Custom Generation"):
                custom_text = gr.Textbox(label="Enter Text", placeholder="Enter text here...", lines=3)
                generate_btn = gr.Button("Generate Audio")
                custom_audio = gr.Audio(label="Generated Audio", type="numpy")

                generate_btn.click(fn=self.generate_audio, inputs=[custom_text], outputs=[custom_audio]) #pylint: disable=no-member

        # Launch
        demo.launch(
            server_name=self.demo_config.get("host", "0.0.0.0"),
            server_port=self.demo_config.get("port", 7860),
            share=self.demo_config.get("share", False),
            allowed_paths=[str(self.audio_dir.parent)],  # Allow access to generated_missing directory
        )


def main():
    parser = argparse.ArgumentParser(description="Gradio Demo Interface")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    demo = GradioDemo(config_path=args.config)
    demo.launch()


if __name__ == "__main__":
    main()
