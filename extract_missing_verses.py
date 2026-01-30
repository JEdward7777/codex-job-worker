#!/usr/bin/env python3
"""
Script to extract missing verses from CODEX files for TTS generation.
Identifies verses that don't have paired audio data and prepares a dataset for generation.
"""

import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import yaml

# Import the downloader to reuse logic
from gitlab_to_hf_dataset import GitLabDatasetDownloader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MissingVersesExtractor:
    """Extracts missing verses from CODEX files."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        logger.info("Reading configuration from: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Initialize downloader to reuse its methods
        self.downloader = GitLabDatasetDownloader(config_path=config_path)

        # Get missing verses configuration (nested under tts)
        self.mv_config = self.config.get("tts", {}).get("missing_verses_generation", {})

        output_dir_str = self.mv_config.get("output_dir", "generated_missing")
        self.output_dir = Path(output_dir_str)
        self.metadata_file = self.mv_config.get("metadata_file", "metadata.csv")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_verse_id(self, verse_id: str) -> Tuple[str, str, str]:
        """
        Parse verse ID into book, chapter, verse.
        Handles formats like 'MAT 1:1', 'MAT 1:1-3', etc.
        Returns (book, chapter, verse)
        """
        # Basic parsing logic - can be enhanced based on actual ID formats
        # Assuming format like "BOOK CHAPTER:VERSE" or similar

        # Remove any non-alphanumeric characters except spaces, colons, dashes
        clean_id = re.sub(r"[^a-zA-Z0-9\s:\-]", "", verse_id)

        parts = clean_id.split()
        if len(parts) >= 2:
            book = parts[0]
            rest = " ".join(parts[1:])

            if ":" in rest:
                chapter, verse = rest.split(":", 1)
            else:
                # Fallback if no colon
                chapter = "0"
                verse = rest

            return book, chapter, verse

        return "UNK", "0", "0"

    def format_filename(self, book: str, chapter: str, verse: str) -> str:
        """
        Create a standardized filename from verse reference.
        Format: BOOK-CHAPTER-VERSE.webm (e.g., MAT-001-001.webm)
        """

        # Clean and pad numbers
        def clean_num(n):
            # Extract first number found
            nums = re.findall(r"\d+", n)
            if nums:
                return nums[0].zfill(3)
            return "000"

        c_pad = clean_num(chapter)
        v_pad = clean_num(verse)

        # Clean book code (keep first 3 chars usually)
        b_code = book[:3].upper()

        return f"{b_code}-{c_pad}-{v_pad}.webm"

    def is_verse_missing_audio(self, cell: Dict, audio_files_map: Dict[str, str]) -> bool:
        """
        Check if a verse is missing valid audio.
        Returns True if:
        - selectedAudioId is null/empty
        - OR selectedAudioId exists but file is missing in repo
        - OR selectedAudioId refers to a deleted/missing recording
        """
        metadata = cell.get("metadata", {})
        attachments = metadata.get("attachments", {})
        selected_audio_id = metadata.get("selectedAudioId")

        # Case 1: No audio selected
        if not selected_audio_id:
            return True

        # Case 2: Selected audio marked as deleted/missing
        if selected_audio_id in attachments:
            audio_info = attachments[selected_audio_id]
            if audio_info.get("isDeleted", False) or audio_info.get("isMissing", False):
                return True

        # Case 3: Audio file not found in repository
        if selected_audio_id not in audio_files_map:
            return True

        return False

    def extract_missing_verses(self) -> List[Dict]:
        """
        Iterate through CODEX files and extract missing verses.
        Returns list of dicts with verse info.
        """
        logger.info("Scanning for missing verses...")

        # Get file lists
        items = self.downloader.list_repository_tree()
        codex_files = [item for item in items if item["name"].endswith(".codex")]

        # Build audio map to check existence
        audio_files_map = self.downloader.build_audio_files_map(items)

        missing_verses = []
        processed_ids = set()

        for idx, codex_file in enumerate(codex_files, 1):
            codex_path = codex_file["path"]
            logger.info("[%s/%s] Processing %s...", idx, len(codex_files), codex_path)

            codex_data = self.downloader.download_json_file(codex_path)
            if not codex_data or "cells" not in codex_data:
                continue

            for cell in codex_data["cells"]:
                verse_id = cell.get("metadata", {}).get("id", "unknown")

                # Skip if already processed (avoid duplicates)
                if verse_id in processed_ids:
                    continue

                # Check if audio is missing
                if self.is_verse_missing_audio(cell, audio_files_map):
                    # Extract text WITHOUT uroman romanization (keep original Chinese)
                    # Romanization will be done during TTS generation
                    text = self.downloader.extract_text_from_cell(cell, suppress_uroman=True)

                    if text:
                        # Parse ID and generate filename
                        book, chapter, verse = self.parse_verse_id(verse_id)
                        filename = self.format_filename(book, chapter, verse)

                        missing_verses.append(
                            {
                                "verse_id": verse_id,
                                "text": text,
                                "book": book,
                                "chapter": chapter,
                                "verse": verse,
                                "filename": filename,
                            }
                        )
                        processed_ids.add(verse_id)

        return missing_verses

    def save_dataset(self, verses: List[Dict]):
        """Save the extracted verses to a CSV file."""
        output_path = self.output_dir / self.metadata_file

        logger.info("Saving %s missing verses to %s", len(verses), output_path)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["verse_id", "text", "book", "chapter", "verse", "filename"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(verses)

        logger.info("Dataset saved successfully.")


def main():
    """
    Main entry point for extracting missing verses from CODEX files.

    Parses command-line arguments, initializes the MissingVersesExtractor,
    scans CODEX files for verses without valid audio recordings, and saves
    the results to a CSV file for subsequent TTS generation.

    Command-line Arguments:
        --config (str): Path to the YAML configuration file (default: "config.yaml")

    The function will:
        1. Load configuration from the specified YAML file
        2. Scan all CODEX files in the GitLab repository
        3. Identify verses missing audio (no selectedAudioId, deleted, or file not found)
        4. Extract verse text and metadata
        5. Save results to a CSV file in the configured output directory

    Prints a summary of found missing verses or a message if none are found.
    """

    parser = argparse.ArgumentParser(description="Extract missing verses for TTS generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    extractor = MissingVersesExtractor(config_path=args.config)
    verses = extractor.extract_missing_verses()

    if verses:
        extractor.save_dataset(verses)
        print(f"\nFound {len(verses)} missing verses.")
        print(f"Output saved to {extractor.output_dir}")
    else:
        print("\nNo missing verses found (or no text available for them).")


if __name__ == "__main__":
    main()
