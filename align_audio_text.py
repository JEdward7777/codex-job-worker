#!/usr/bin/env python3
"""
Audio-Text Alignment Tool using Dynamic Programming

This script aligns MP3 audio files with text files by detecting silence regions
and matching them to verse boundaries using dynamic programming optimization.
"""
#pylint: disable=too-many-lines

import os
import json
import csv
import subprocess
import time
import uuid
import shutil
import re
import tempfile
from typing import List, Tuple, Dict, Optional
from pydub.silence import detect_silence
from pydub import AudioSegment
import fire
from gitlab_to_hf_dataset import GitLabDatasetDownloader


class AudioTextAligner:
    """Aligns audio files with text files using silence detection and dynamic programming."""

    def __init__(self):
        self.silence_thresh_db = -40
        self.min_silence_len_ms = 500
        self.audio_duration = 0
        self.silence_positions = []
        self.verses = []
        self.total_chars = 0
        self.time_to_char_ratio = 0

    def detect_silence_regions(self, mp3_path: str,
                              silence_thresh: int = -40,
                              min_silence_len: int = 500) -> List[Tuple[float, float]]:
        """
        Detect silence regions in an audio file (MP3 or WAV).

        Args:
            mp3_path: Path to the audio file (MP3 or WAV)
            silence_thresh: Silence threshold in dB (default: -40)
            min_silence_len: Minimum silence length in milliseconds (default: 500)

        Returns:
            List of tuples (start_time, end_time) in seconds
        """
        print(f"Loading audio file: {mp3_path}")

        # Detect file type and load accordingly
        if mp3_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(mp3_path)
        elif mp3_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(mp3_path)
        else:
            # Try to load as generic audio file
            audio = AudioSegment.from_file(mp3_path)

        self.audio_duration = len(audio) / 1000.0  # Convert to seconds

        print(f"Audio duration: {self.audio_duration:.2f} seconds")
        print(f"Detecting silence (threshold: {silence_thresh}dB, min length: {min_silence_len}ms)...")

        # Detect silence regions (returns list of [start_ms, end_ms])
        silence_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        # Convert to seconds and store as tuples
        silence_positions = [
            (start / 1000.0, end / 1000.0)
            for start, end in silence_ranges
        ]

        print(f"Detected {len(silence_positions)} silence regions")

        return silence_positions

    def load_verses(self, text_path: str) -> List[str]:
        """
        Load verses from a text file.

        Args:
            text_path: Path to the text file

        Returns:
            List of verse strings
        """
        print(f"Loading text file: {text_path}")
        with open(text_path, 'r', encoding='utf-8') as f:
            verses = [line.rstrip('\n') for line in f]

        # Filter out empty lines
        verses = [v for v in verses if v.strip()]

        print(f"Loaded {len(verses)} verses")
        self.total_chars = sum(len(v) for v in verses)
        print(f"Total characters: {self.total_chars}")

        return verses


    def dynamic_programming_alignment__tree_version(self, verbose: bool = False, silence_factor: float = 0.2) -> Tuple[List[Tuple[int, int]], float]:
        """
        Perform dynamic programming alignment using a recursive tree-based approach.

        Args:
            verbose: If True, print debug information
            silence_factor: Penalty factor for smaller silences. Higher values encourage selection of larger
                          silence regions as verse boundaries. The penalty is calculated as silence_factor/(silence_size^2),
                          so smaller silences get higher penalties. (default: 0.2)

        Returns:
            Tuple of (path, total_cost)
            - path: List of (silence_idx, verse_idx) tuples representing the alignment
            - total_cost: Total cost of the optimal path
        """
        num_verses = len(self.verses)
        num_silences = len(self.silence_positions)

        # Memoization cache
        memo = {}

        def try_split(start_silence_index, end_silence_index, start_verse_index, end_verse_index_plus_one):
            # Check memoization cache
            cache_key = (start_silence_index, end_silence_index, start_verse_index, end_verse_index_plus_one)
            if cache_key in memo:
                return memo[cache_key]

            assert end_silence_index >= start_silence_index, f"start_silence_index: {start_silence_index}, end_silence_index: {end_silence_index}"
            assert end_verse_index_plus_one >= start_verse_index, f"start_verse_index: {start_verse_index}, end_verse_index_plus_one: {end_verse_index_plus_one}"

            # Check for zero audio length case
            if start_silence_index == end_silence_index:
                result = {
                    "error": float('inf'),
                    "path": [],
                }
                memo[cache_key] = result
                return result

            # If the number of silences is less than or equal to one less than the number of verses, this is a problem
            num_verses_in_split = end_verse_index_plus_one - start_verse_index
            num_silences_in_split = end_silence_index - start_silence_index
            if num_silences_in_split <= num_verses_in_split - 1:
                result = {
                    "error": float('inf'),
                    "path": [],
                }
                memo[cache_key] = result
                return result

            # Base case: single verse
            if num_verses_in_split == 1:
                if end_silence_index == num_silences:
                    audio_end = self.audio_duration
                else:
                    # Use middle of silence instead of start
                    silence_start, silence_end = self.silence_positions[end_silence_index]
                    audio_end = (silence_start + silence_end) / 2.0

                if start_silence_index < 0:
                    audio_start = 0
                else:
                    # Use middle of silence instead of end
                    silence_start, silence_end = self.silence_positions[start_silence_index]
                    audio_start = (silence_start + silence_end) / 2.0

                audio_length = audio_end - audio_start
                text_length = len(self.verses[start_verse_index])
                expected_length = self.time_to_char_ratio * text_length

                error = (audio_length - expected_length)**2

                #The silence factor is stupposed to help the fit pick larger silences. (assuming these are more likely the inbetween verse silences)
                if end_silence_index < num_silences:
                    silence_start, silence_end = self.silence_positions[end_silence_index]
                    silence_size = silence_end - silence_start
                    error += silence_factor/(silence_size**2)

                result = {
                    "error": error,
                    "path": [{
                        "verse": start_verse_index,
                        "audio_start": start_silence_index,
                        "audio_end": end_silence_index
                    }],
                }
                memo[cache_key] = result
                return result

            # Recursive case: split into two parts
            best_error = float("inf")
            best_result = None
            middle_verse = start_verse_index + (end_verse_index_plus_one - start_verse_index) // 2

            for audio_split_point in range(start_silence_index + 1, end_silence_index):
                test_result_lhs = try_split(start_silence_index, audio_split_point, start_verse_index, middle_verse)
                test_result_rhs = try_split(audio_split_point, end_silence_index, middle_verse, end_verse_index_plus_one)

                if test_result_lhs["error"] + test_result_rhs["error"] < best_error:
                    best_error = test_result_lhs["error"] + test_result_rhs["error"]
                    best_result = {
                        "error": best_error,
                        "path": test_result_lhs["path"] + test_result_rhs["path"]
                    }

            # Handle case where no valid split was found
            if best_result is None:
                best_result = {
                    "error": float('inf'),
                    "path": []
                }

            memo[cache_key] = best_result
            return best_result

        # Call try_split with initial parameters
        result = try_split(-1, num_silences, 0, num_verses)

        if result["error"] == float('inf'):
            raise ValueError("No valid alignment found using tree version algorithm.")

        # Convert path format from tree version to match original format
        # Tree version path: [{"verse": v, "audio_start": s, "audio_end": e}, ...]
        # Original format: [(silence_idx, verse_idx), ...]
        # We need to convert to [(audio_end, verse), ...] and prepend (-1, -1)

        converted_path = [(-1, -1)]  # Start with initial position
        for item in result["path"]:
            converted_path.append((item["audio_end"], item["verse"]))

        if verbose:
            print("\nTree version alignment complete")
            print(f"Number of cache hits: {len(memo)}")

        return converted_path, result["error"]


    def dynamic_programming_alignment(self, verbose: bool = False) -> Tuple[List[Tuple[int, int]], float]:
        """
        Perform dynamic programming to find optimal alignment.

        Args:
            verbose: If True, print the full DP matrix as a Markdown table

        Returns:
            Tuple of (path, total_cost)
            - path: List of (silence_idx, verse_idx) tuples representing the alignment
            - total_cost: Total cost of the optimal path
        """
        num_silences = len(self.silence_positions)
        num_verses = len(self.verses)

        print("\nBuilding DP matrix")

        # location_best: dict mapping (silence_idx, verse_idx) to {"cost": float, "previouse": tuple or None}
        # silence_idx ranges from 0 to num_silences (inclusive, where num_silences represents end of audio)
        # verse_idx ranges from 0 to num_verses-1

        location_best = {}
        default_entry = {
            "cost": 0,
            "previouse": None
        }

        # Initialize base case: before any verses, at the start
        location_best[(-1, -1)] = default_entry

        # Fill the DP matrix
        for silence_index in range(0, num_silences + 1):
            for verse_index in range(0, num_verses):

                # Default to worse cost
                current_cost = float('inf')

                text_length = len(self.verses[verse_index])
                expected_length = self.time_to_char_ratio * text_length

                # Determine audio end position
                if silence_index == num_silences:
                    audio_end = self.audio_duration
                else:
                    # Use middle of silence instead of start
                    silence_start, silence_end = self.silence_positions[silence_index]
                    audio_end = (silence_start + silence_end) / 2.0


                # Try diagonal move: match this verse to this silence boundary
                possible_previous_loc = (silence_index - 1, verse_index - 1)
                if possible_previous_loc in location_best:
                    drag_start = location_best[possible_previous_loc]

                    if possible_previous_loc[0] < 0:
                        audio_start = 0
                    else:
                        # Use middle of silence instead of end
                        silence_start, silence_end = self.silence_positions[possible_previous_loc[0]]
                        audio_start = (silence_start + silence_end) / 2.0

                    audio_length = audio_end - audio_start
                    error = audio_length - expected_length
                    diagonal_cost = error**2 + drag_start["cost"]

                    if diagonal_cost < current_cost:
                        current_cost = diagonal_cost
                        location_best[(silence_index, verse_index)] = {
                            "cost": diagonal_cost,
                            "previouse": possible_previous_loc,
                        }


                # Try up move: skip a silence (merge multiple silences for one verse)
                up_location = (silence_index - 1, verse_index)
                if up_location in location_best:
                    up_best = location_best[up_location]
                    possible_previous_loc = up_best["previouse"]

                    if possible_previous_loc is not None:
                        drag_start = location_best.get(possible_previous_loc, default_entry)

                        if possible_previous_loc[0] < 0:
                            audio_start = 0
                        else:
                            # Use middle of silence instead of end
                            silence_start, silence_end = self.silence_positions[possible_previous_loc[0]]
                            audio_start = (silence_start + silence_end) / 2.0

                        audio_length = audio_end - audio_start
                        error = audio_length - expected_length
                        diagonal_cost = error**2 + drag_start["cost"]

                        if diagonal_cost < current_cost:
                            current_cost = diagonal_cost
                            location_best[(silence_index, verse_index)] = {
                                "cost": diagonal_cost,
                                "previouse": possible_previous_loc,
                            }


        # Print verbose output if requested
        if verbose:
            print("\n## DP Matrix (Markdown Table)\n")

            # Create header row
            header = "| Silence \\ Verse |"
            for v in range(-1, num_verses):
                header += f" {v} |"
            print(header)

            # Create separator row
            separator = "|" + "---|" * (num_verses + 2)
            print(separator)

            # Print each row
            for s in range(0, num_silences + 1):
                row = f"| {s} |"
                for v in range(-1, num_verses):
                    if (s, v) in location_best:
                        cost = location_best[(s, v)]["cost"]
                        row += f" {cost:.2f} |"
                    else:
                        row += " - |"
                print(row)
            print()

        # Backtrack to find the path
        path = []
        current = (num_silences, num_verses - 1)

        if current not in location_best:
            raise ValueError(f"No valid alignment found. Final position {current} not in location_best.")

        final_cost = location_best[current]["cost"]
        print(f"Final cost: {final_cost:.2f}")

        # Backtrack using the "previouse" pointers
        # We need to include the final position in the path
        while current is not None:
            path.append(current)
            if current in location_best:
                current = location_best[current]["previouse"]
            else:
                break

        path.reverse()

        return path, final_cost

    def extract_audio_segments(self, mp3_path: str, path: List[Tuple[int, int]],
                               output_dir: str, silence_retained: float = 0.2):
        """
        Extract audio segments based on the alignment path.

        Args:
            mp3_path: Path to the MP3 file
            path: Alignment path from DP - list of (silence_idx, verse_idx) tuples
            output_dir: Output directory for audio segments
            silence_retained: Fraction of silence to retain (0.0 = remove all, 1.0 = keep all, 0.5 = keep half)
                            The retained silence is split evenly between adjacent verses (default: 0.2)
        """
        print(f"\nExtracting audio segments to: {output_dir}")

        # Create output directory
        audio_output_dir = os.path.join(output_dir, 'audio')
        os.makedirs(audio_output_dir, exist_ok=True)

        # Load audio
        audio = AudioSegment.from_mp3(mp3_path)

        # Track segment boundaries
        segment_info = []
        segment_num = 0

        # Process each step in the path to extract verse segments
        # Path goes from (-1, -1) to (num_silences, num_verses-1)
        # Each verse i corresponds to the audio between path positions where verse_idx goes from i-1 to i
        for i in range(1, len(path)):
            prev_silence_idx, _prev_verse_idx = path[i - 1]
            curr_silence_idx, curr_verse_idx = path[i]

            # The current verse index tells us which verse we just completed
            verse_idx = curr_verse_idx

            # Determine the audio boundaries for this verse
            # Start: after the previous silence (or from beginning if prev_silence_idx < 0)
            if prev_silence_idx < 0:
                audio_start_sec = 0
            else:
                # Get the end of the previous silence
                prev_silence_start, prev_silence_end = self.silence_positions[prev_silence_idx]
                silence_duration = prev_silence_end - prev_silence_start
                # Retain half of the specified fraction at the end of the silence
                audio_start_sec = prev_silence_end - (silence_duration * silence_retained / 2.0)

            # End: at the start of the current silence (or end of audio if at last silence)
            if curr_silence_idx >= len(self.silence_positions):
                audio_end_sec = self.audio_duration
            else:
                # Get the start of the current silence
                curr_silence_start, curr_silence_end = self.silence_positions[curr_silence_idx]
                silence_duration = curr_silence_end - curr_silence_start
                # Retain half of the specified fraction at the start of the silence
                audio_end_sec = curr_silence_start + (silence_duration * silence_retained / 2.0)

            # Convert to milliseconds for pydub
            audio_start_ms = int(audio_start_sec * 1000)
            audio_end_ms = int(audio_end_sec * 1000)

            # Extract audio segment
            audio_segment = audio[audio_start_ms:audio_end_ms]

            # Verse range (1-indexed for display)
            verse_range = f"{verse_idx + 1}"

            # Save audio file
            output_filename = f"segment_{segment_num:03d}_verse_{verse_range}.mp3"
            output_path = os.path.join(audio_output_dir, output_filename)
            audio_segment.export(output_path, format="mp3")

            segment_info.append({
                'segment_num': segment_num,
                'verse_range': verse_range,
                'verse_indices': [verse_idx],
                'start_time': audio_start_sec,
                'end_time': audio_end_sec,
                'duration': audio_end_sec - audio_start_sec,
                'filename': output_filename
            })

            print(f"  Segment {segment_num}: verse {verse_range}, "
                  f"{audio_start_sec:.2f}s - {audio_end_sec:.2f}s")

            segment_num += 1

        # Save alignment info as JSON
        alignment_path = os.path.join(output_dir, 'alignment.json')
        with open(alignment_path, 'w', encoding='utf-8') as f:
            json.dump({
                'segments': segment_info,
                'total_segments': len(segment_info),
                'total_verses': len(self.verses),
                'total_silences_detected': len(self.silence_positions),
                'audio_duration': self.audio_duration
            }, f, indent=2)

        print(f"\nAlignment info saved to: {alignment_path}")

        # Save manifest.csv
        manifest_path = os.path.join(output_dir, 'manifest.csv')
        with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_file', 'text', 'text_length', 'audio_duration'])

            for seg in segment_info:
                # Get the text for this segment's verses
                verse_texts = [self.verses[idx] for idx in seg['verse_indices']]
                combined_text = ' '.join(verse_texts)
                text_length = len(combined_text)
                audio_duration = seg['duration']

                # Write relative path to audio file
                audio_rel_path = os.path.join('audio', seg['filename'])
                writer.writerow([audio_rel_path, combined_text, text_length, audio_duration])

        print(f"Manifest saved to: {manifest_path}")
        print(f"Total segments created: {len(segment_info)}")

    def align(self, mp3_file: str, text_file: str, output_dir: str,
              silence_thresh: int = -40, min_silence_len: int = 500,
              verbose: bool = False, silence_retained: float = 0.2,
              use_tree_version: bool = False, silence_factor: float = 0.2):
        """
        Main method to align audio with text.

        Args:
            mp3_file: Path to the MP3 file
            text_file: Path to the text file with verses
            output_dir: Output directory for results
            silence_thresh: Silence threshold in dB (default: -40)
            min_silence_len: Minimum silence length in milliseconds (default: 500)
            verbose: If True, print the full DP matrix as a Markdown table (default: False)
            silence_retained: Fraction of silence to retain in extracted segments (0.0-1.0, default: 0.2)
            use_tree_version: If True, use the tree-based recursive alignment algorithm instead of the standard DP (default: False)
            silence_factor: Penalty factor for tree version algorithm - encourages larger silences as boundaries (default: 0.2)
        """
        print("=" * 80)
        print("Audio-Text Alignment Tool")
        print("=" * 80)

        # Validate inputs
        if not os.path.exists(mp3_file):
            raise FileNotFoundError(f"MP3 file not found: {mp3_file}")
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        # Load data
        self.silence_positions = self.detect_silence_regions(
            mp3_file, silence_thresh, min_silence_len
        )
        self.verses = self.load_verses(text_file)

        # Calculate character to time ratio
        if self.total_chars > 0:
            self.time_to_char_ratio = self.audio_duration / self.total_chars
            print(f"Character to time ratio: {self.time_to_char_ratio:.4f} seconds/char")
        else:
            raise ValueError("No text content found in the text file")

        # Check if we have enough silences
        # We need num_verses - 1 silences to separate num_verses verses
        expected_silences = len(self.verses) - 1
        if len(self.silence_positions) < expected_silences:
            print(f"\nWARNING: Only {len(self.silence_positions)} silence regions detected "
                  f"for {len(self.verses)} verses (expected at least {expected_silences}).")
            print("Consider adjusting silence detection parameters:")
            print(f"  - Increase silence_thresh (currently {silence_thresh}dB)")
            print(f"  - Decrease min_silence_len (currently {min_silence_len}ms)")

        # Perform dynamic programming alignment
        if use_tree_version:
            print("\nUsing tree-based recursive alignment algorithm")
            path, total_cost = self.dynamic_programming_alignment__tree_version(verbose=verbose, silence_factor=silence_factor)
        else:
            path, total_cost = self.dynamic_programming_alignment(verbose=verbose)

        print(f"\nAlignment path length: {len(path)}")
        print(f"Total cost: {total_cost:.2f}")

        # Extract audio segments
        self.extract_audio_segments(mp3_file, path, output_dir, silence_retained=silence_retained)

        print("\n" + "=" * 80)
        print("Alignment complete!")
        print("=" * 80)

    def extract_chapter(self, book_name: str, chapter_number: int,
                       output_file: str, config_path: str = "config.yaml",
                       romanize: bool = False):
        """
        Extract a chapter from GitLab repository and write verses to a text file.

        This method uses the GitLabDatasetDownloader to fetch chapter data and
        writes each verse as a separate line to the output file.

        Args:
            book_name: Name of the book (e.g., "GEN", "EXO", "MAT")
            chapter_number: Chapter number (1-based)
            output_file: Path to output text file
            config_path: Path to configuration YAML file (default: config.yaml)
            romanize: If True, apply uroman romanization to the text (default: False)
        """
        print("=" * 80)
        print("Chapter Text Extraction Tool")
        print("=" * 80)
        print(f"Book: {book_name}")
        print(f"Chapter: {chapter_number}")
        print(f"Output file: {output_file}")
        print()


        # Validate config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return

        # Initialize downloader
        try:
            downloader = GitLabDatasetDownloader(config_path=config_path)
        except Exception as e:
            print(f"Error initializing GitLabDatasetDownloader: {e}")
            return

        # List all files to find available books
        print("Fetching repository structure...")
        try:
            items = downloader.list_repository_tree()
            codex_files = [item for item in items if item['name'].endswith('.codex')]
        except Exception as e:
            print(f"Error listing repository: {e}")
            return

        # Extract book names from codex files
        available_books = {}
        for codex_file in codex_files:
            filename = codex_file['name']
            book_code = filename.replace('.codex', '').upper()
            available_books[book_code] = codex_file['path']

        print(f"Found {len(available_books)} books in repository")

        # Check if requested book exists
        book_name_upper = book_name.upper()
        if book_name_upper not in available_books:
            print(f"\nError: Book '{book_name}' not found in repository.")
            print("\nAvailable books:")
            for book_code in sorted(available_books.keys()):
                print(f"  - {book_code}")
            return

        # Download the codex file
        codex_path = available_books[book_name_upper]
        print(f"\nDownloading {codex_path}...")

        try:
            codex_data = downloader.download_json_file(codex_path)
        except Exception as e:
            print(f"Error downloading codex file: {e}")
            return

        if not codex_data or 'cells' not in codex_data:
            print("Error: Invalid codex file format")
            return

        # Extract verses for the specified chapter
        print(f"Extracting verses from chapter {chapter_number}...")

        chapter_verses = []
        available_chapters = set()

        for cell in codex_data['cells']:
            metadata = cell.get('metadata', {})
            verse_id = metadata.get('id', '')

            # Parse verse ID (format: "GEN 1:1" or "MAT 3:1-2")
            # Split by space first to separate book from chapter:verse
            parts = verse_id.split(' ')
            if len(parts) >= 2:
                try:
                    # Split chapter:verse part by colon
                    chapter_verse = parts[1].split(':')
                    if len(chapter_verse) >= 2:
                        cell_chapter = int(chapter_verse[0])
                        available_chapters.add(cell_chapter)

                        if cell_chapter == chapter_number:
                            # Extract text from cell
                            # suppress_uroman=True keeps original script, False allows romanization
                            text = downloader.extract_text_from_cell(cell, suppress_uroman=not romanize)

                            if text:
                                # Replace newlines and carriage returns with spaces
                                # This ensures each verse is on a single line
                                text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                                # Clean up multiple spaces
                                text = ' '.join(text.split())

                                # Handle verse ranges like "1-2" or single verses like "1"
                                verse_part = chapter_verse[1]
                                if '-' in verse_part:
                                    # It's a range, use the first verse number
                                    verse_num = int(verse_part.split('-')[0])
                                else:
                                    verse_num = int(verse_part)
                                chapter_verses.append((verse_num, text))
                except (ValueError, IndexError):
                    continue

        # Check if chapter was found
        if not chapter_verses:
            print(f"\nError: Chapter {chapter_number} not found in book '{book_name}'.")
            print(f"\nAvailable chapters in {book_name}:")
            for ch in sorted(available_chapters):
                print(f"  - Chapter {ch}")
            return

        # Sort verses by verse number
        chapter_verses.sort(key=lambda x: x[0])

        # Write to output file
        print(f"\nWriting {len(chapter_verses)} verses to {output_file}...")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for verse_num, text in chapter_verses:
                    f.write(text + '\n')

            print(f"\n✓ Successfully extracted {len(chapter_verses)} verses")
            print(f"✓ Output written to: {output_file}")
            print("\nYou can now use this file with the 'align' command:")
            print(f"  python align_audio_text.py align --mp3_file=<audio.mp3> --text_file={output_file} --output_dir=<output>")

        except Exception as e:
            print(f"Error writing output file: {e}")
            return

        print("\n" + "=" * 80)

    def _snap_fragments_to_silence(self, fragments: List[Dict],
                                   silence_regions: List[Tuple[float, float]],
                                   silence_retained: float,
                                   max_distance_before: float,
                                   max_distance_after: float) -> List[Dict]:
        """
        Snap fragment boundaries to nearest silence regions.

        Args:
            fragments: List of fragment dictionaries with 'begin' and 'end' timestamps
            silence_regions: List of (start, end) tuples for silence regions in seconds
            silence_retained: Fraction of silence to retain (0.0-1.0)
            max_distance_before: Maximum distance in seconds to search before boundary
            max_distance_after: Maximum distance in seconds to search after boundary

        Returns:
            List of fragments with adjusted boundaries
        """
        if not silence_regions:
            return fragments

        adjusted_fragments = []

        for idx, fragment in enumerate(fragments):
            begin_sec = float(fragment['begin'])
            end_sec = float(fragment['end'])

            # Store original timestamps
            fragment['original_begin'] = begin_sec
            fragment['original_end'] = end_sec

            # Don't adjust the very first begin or very last end
            adjusted_begin = begin_sec
            adjusted_end = end_sec

            # Adjust begin (skip first fragment)
            if idx > 0:
                adjusted_begin = self._find_nearest_silence_boundary(
                    begin_sec, silence_regions, silence_retained,
                    max_distance_before, max_distance_after, is_segment_start=True
                )

            # Adjust end (skip last fragment)
            if idx < len(fragments) - 1:
                adjusted_end = self._find_nearest_silence_boundary(
                    end_sec, silence_regions, silence_retained,
                    max_distance_before, max_distance_after, is_segment_start=False
                )

            fragment['begin'] = adjusted_begin
            fragment['end'] = adjusted_end
            adjusted_fragments.append(fragment)

        return adjusted_fragments

    def _find_nearest_silence_boundary(self, timestamp: float,
                                      silence_regions: List[Tuple[float, float]],
                                      silence_retained: float,
                                      max_distance_before: float,
                                      max_distance_after: float,
                                      is_segment_start: bool) -> float:
        """
        Find the nearest silence region to a timestamp and return adjusted boundary.

        Args:
            timestamp: The original timestamp in seconds
            silence_regions: List of (start, end) tuples for silence regions
            silence_retained: Fraction of silence to retain (0.0-1.0)
            max_distance_before: Maximum distance to search before timestamp
            max_distance_after: Maximum distance to search after timestamp
            is_segment_start: True if this is a segment start, False if segment end

        Returns:
            Adjusted timestamp in seconds
        """
        min_search = timestamp - max_distance_before
        max_search = timestamp + max_distance_after

        best_silence = None
        best_distance = float('inf')

        # Find the silence region closest to the timestamp
        for silence_start, silence_end in silence_regions:
            # Check if any part of this silence region is within search range
            if silence_end < min_search or silence_start > max_search:
                continue

            # Calculate distance from timestamp to closest part of silence
            if timestamp < silence_start:
                distance = silence_start - timestamp
            elif timestamp > silence_end:
                distance = timestamp - silence_end
            else:
                # Timestamp is within the silence region
                distance = 0

            if distance < best_distance:
                best_distance = distance
                best_silence = (silence_start, silence_end)

        # If no silence found within range, return original timestamp
        if best_silence is None:
            return timestamp

        # Calculate the adjusted boundary based on silence_retained and boundary type
        silence_start, silence_end = best_silence
        silence_duration = silence_end - silence_start
        #silence_center = (silence_start + silence_end) / 2.0

        # For segment START (end of previous segment, beginning of current):
        #   - silence_retained = 0.0: boundary at silence_end (no silence included)
        #   - silence_retained = 1.0: boundary at silence_center (split evenly)
        # For segment END (end of current segment, beginning of next):
        #   - silence_retained = 0.0: boundary at silence_start (no silence included)
        #   - silence_retained = 1.0: boundary at silence_center (split evenly)

        if is_segment_start:
            # Segment starts after this boundary
            # At 0.0: start at silence_end (exclude all silence from this segment)
            # At 1.0: start at silence_center (include half the silence)
            # Formula: silence_end - (silence_duration / 2) * silence_retained
            adjusted_boundary = silence_end - (silence_duration / 2.0) * silence_retained
        else:
            # Segment ends at this boundary
            # At 0.0: end at silence_start (exclude all silence from this segment)
            # At 1.0: end at silence_center (include half the silence)
            # Formula: silence_start + (silence_duration / 2) * silence_retained
            adjusted_boundary = silence_start + (silence_duration / 2.0) * silence_retained

        return adjusted_boundary

    def align_with_aeneas(self, mp3_file: str, text_file: str, output_dir: str,
                         task_language: str = "eng", snap_to_silence: bool = False,
                         silence_thresh: int = -40, min_silence_len: int = 500,
                         silence_retained: float = 0.2,
                         max_snap_distance_before: int = 1000,
                         max_snap_distance_after: int = 1000):
        """
        Align audio with text using the aeneas Docker container for forced alignment.

        This method uses the aeneas forced alignment tool running in a Docker container
        to align an MP3 audio file with a text file (one verse per line). It then extracts
        the aligned audio segments and creates a manifest.csv file.

        Args:
            mp3_file: Path to the MP3 file to be aligned
            text_file: Path to the text file with one verse per line
            output_dir: Output directory where manifest.csv and audio/ folder will be created
            task_language: Language code for aeneas alignment (default: "eng")
            snap_to_silence: If True, adjust fragment boundaries to nearest silence regions (default: False)
            silence_thresh: Silence threshold in dB for snap_to_silence (default: -40)
            min_silence_len: Minimum silence length in ms for snap_to_silence (default: 500)
            silence_retained: Fraction of silence to retain (0.0-1.0, default: 0.2)
                            0.0 = no silence, 1.0 = split silence evenly between segments
            max_snap_distance_before: Maximum distance in ms to search before boundary (default: 1000)
            max_snap_distance_after: Maximum distance in ms to search after boundary (default: 1000)
        """
        print("=" * 80)
        print("Audio-Text Alignment Tool (Aeneas)")
        print("=" * 80)

        # Validate input files exist
        if not os.path.exists(mp3_file):
            raise FileNotFoundError(f"MP3 file not found: {mp3_file}")
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        print(f"MP3 file: {mp3_file}")
        print(f"Text file: {text_file}")
        print(f"Output directory: {output_dir}")
        print(f"Language: {task_language}")
        print()

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        audio_output_dir = os.path.join(output_dir, 'audio')
        os.makedirs(audio_output_dir, exist_ok=True)

        # Get absolute paths and directories for Docker volume mounts
        mp3_file_abs = os.path.abspath(mp3_file)
        text_file_abs = os.path.abspath(text_file)

        mp3_dir = os.path.dirname(mp3_file_abs)
        text_dir = os.path.dirname(text_file_abs)

        mp3_filename = os.path.basename(mp3_file_abs)
        text_filename = os.path.basename(text_file_abs)

        # Generate unique temporary JSON filename
        temp_json_filename = f"aeneas_output_{uuid.uuid4().hex}.json"
        temp_json_path = os.path.join(text_dir, temp_json_filename)

        # Build Docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{mp3_dir}:/audio",
            "-v", f"{text_dir}:/text",
            "aeneas-py27",
            "python", "-m", "aeneas.tools.execute_task",
            f"/audio/{mp3_filename}",
            f"/text/{text_filename}",
            f"task_language={task_language}|os_task_file_format=json|is_text_type=plain",
            f"/text/{temp_json_filename}"
        ]

        print("Running aeneas forced alignment...")
        print(f"Docker command: {' '.join(docker_cmd)}")
        print()

        # Execute Docker command with retry logic
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                _result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Check if JSON file was created
                if os.path.exists(temp_json_path):
                    print("✓ Aeneas alignment completed successfully")
                    break
                else:
                    print(f"Warning: JSON output file not found (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise FileNotFoundError(f"Aeneas did not create output file: {temp_json_path}")

            except subprocess.CalledProcessError as e:
                print(f"Docker command failed (attempt {attempt + 1}/{max_retries})")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Aeneas alignment failed after {max_retries} attempts") from e

        # Read the JSON output
        print(f"\nReading alignment results from: {temp_json_path}")
        try:
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON output: {e}") from e

        # Extract fragments
        if 'fragments' not in alignment_data:
            raise ValueError("Invalid JSON format: 'fragments' key not found")

        fragments = alignment_data['fragments']
        print(f"✓ Found {len(fragments)} aligned fragments")

        # Detect silence regions if snap_to_silence is enabled
        silence_regions = []
        if snap_to_silence:
            print("\nDetecting silence regions for boundary snapping...")
            silence_regions = self.detect_silence_regions(
                mp3_file, silence_thresh, min_silence_len
            )
            print(f"✓ Detected {len(silence_regions)} silence regions")

        # Load audio file for segmentation
        print(f"\nLoading audio file: {mp3_file}")
        audio = AudioSegment.from_mp3(mp3_file)
        audio_duration = len(audio) / 1000.0  # Convert to seconds
        print(f"Audio duration: {audio_duration:.2f} seconds")

        # Get base name for output files
        mp3_basename = os.path.splitext(mp3_filename)[0]

        # Snap boundaries to silence if enabled
        if snap_to_silence and silence_regions:
            print("\nSnapping fragment boundaries to silence regions...")
            fragments = self._snap_fragments_to_silence(
                fragments, silence_regions, silence_retained,
                max_snap_distance_before / 1000.0,  # Convert ms to seconds
                max_snap_distance_after / 1000.0
            )
            print("✓ Boundary snapping complete")

        # Extract audio segments and prepare manifest data
        print(f"\nExtracting audio segments to: {audio_output_dir}")
        manifest_rows = []

        for idx, fragment in enumerate(fragments):
            # Parse timestamps
            begin_sec = float(fragment['begin'])
            end_sec = float(fragment['end'])
            original_begin = float(fragment.get('original_begin', begin_sec))
            original_end = float(fragment.get('original_end', end_sec))

            # Join lines with spaces
            lines = fragment.get('lines', [])
            text = ' '.join(lines)
            text_length = len(text)

            # Convert to milliseconds for pydub
            begin_ms = int(begin_sec * 1000)
            end_ms = int(end_sec * 1000)

            # Extract audio segment
            audio_segment = audio[begin_ms:end_ms]

            # Generate output filename
            output_filename = f"{mp3_basename}_{idx:03d}.wav"
            output_path = os.path.join(audio_output_dir, output_filename)

            # Export as WAV
            audio_segment.export(output_path, format="wav")

            # Calculate adjustments
            start_adjustment = begin_sec - original_begin
            end_adjustment = end_sec - original_end
            segment_duration = end_sec - begin_sec

            # Add to manifest
            relative_path = os.path.join('audio', output_filename)
            manifest_rows.append({
                'file_name': relative_path,
                'transcription': text,
                'text_length': text_length,
                'audio_duration': segment_duration,
                'start_adjustment': start_adjustment,
                'end_adjustment': end_adjustment
            })

            if snap_to_silence:
                print(f"  Segment {idx:03d}: {begin_sec:.2f}s - {end_sec:.2f}s "
                      f"(Δstart: {start_adjustment:+.3f}s, Δend: {end_adjustment:+.3f}s) -> {output_filename}")
            else:
                print(f"  Segment {idx:03d}: {begin_sec:.2f}s - {end_sec:.2f}s -> {output_filename}")

        # Write manifest.csv
        manifest_path = os.path.join(output_dir, 'manifest.csv')
        print(f"\nWriting manifest to: {manifest_path}")

        with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'transcription', 'text_length', 'audio_duration',
                           'start_adjustment', 'end_adjustment'])

            for row in manifest_rows:
                writer.writerow([
                    row['file_name'],
                    row['transcription'],
                    row['text_length'],
                    row['audio_duration'],
                    row['start_adjustment'],
                    row['end_adjustment']
                ])

        print(f"✓ Manifest saved with {len(manifest_rows)} entries")

        # Clean up temporary JSON file
        try:
            os.remove(temp_json_path)
            print(f"\n✓ Cleaned up temporary file: {temp_json_filename}")
        except Exception as e:
            print(f"\nWarning: Could not remove temporary file {temp_json_path}: {e}")

        # Print summary
        print("\n" + "=" * 80)
        print("Alignment Summary")
        print("=" * 80)
        print(f"Total segments created: {len(manifest_rows)}")
        print(f"Total audio duration: {audio_duration:.2f} seconds")
        print(f"Average segment duration: {sum(r['audio_duration'] for r in manifest_rows) / len(manifest_rows):.2f} seconds")

        if snap_to_silence:
            total_start_adj = sum(abs(r['start_adjustment']) for r in manifest_rows)
            total_end_adj = sum(abs(r['end_adjustment']) for r in manifest_rows)
            print(f"Total boundary adjustments: start={total_start_adj:.2f}s, end={total_end_adj:.2f}s")

        print("\nOutput files:")
        print(f"  - Manifest: {manifest_path}")
        print(f"  - Audio segments: {audio_output_dir}/ ({len(manifest_rows)} WAV files)")
        print("\n" + "=" * 80)
        print("Alignment complete!")
        print("=" * 80)

    def _trim_audio_after_nth_silence(self, audio_path: str, num_silences_to_skip: int,
                                      silence_thresh: int = -40, min_silence_len: int = 500) -> Optional[str]:
        """
        Trim audio after the Nth silence region and save to a temporary file.

        Args:
            audio_path: Path to the audio file to trim
            num_silences_to_skip: Number of silence regions to skip (0 = no trimming)
            silence_thresh: Silence threshold in dB (default: -40)
            min_silence_len: Minimum silence length in milliseconds (default: 500)

        Returns:
            Path to temporary trimmed audio file, or None if no trimming needed
        """
        if num_silences_to_skip <= 0:
            return None

        print(f"  Trimming audio after {num_silences_to_skip} silence(s)...")

        # Detect silences
        silence_regions = self.detect_silence_regions(audio_path, silence_thresh, min_silence_len)

        if len(silence_regions) < num_silences_to_skip:
            print(f"  Warning: Only {len(silence_regions)} silence(s) detected, cannot skip {num_silences_to_skip}")
            return None

        # Get the end of the Nth silence
        trim_point_sec = silence_regions[num_silences_to_skip - 1][1]  # End of Nth silence
        trim_point_ms = int(trim_point_sec * 1000)

        print(f"  Trimming at {trim_point_sec:.2f}s (after silence #{num_silences_to_skip})")

        # Load audio
        if audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(audio_path)
        elif audio_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)

        # Trim audio
        trimmed_audio = audio[trim_point_ms:]

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='trimmed_audio_')
        os.close(temp_fd)  # Close the file descriptor

        # Export trimmed audio
        trimmed_audio.export(temp_path, format='wav')
        print(f"  Trimmed audio saved to temporary file: {temp_path}")

        return temp_path

    def align_folder_with_aeneas(self, audio_folder: str, book_chapter_regex: str,
                                 book_map_csv: str, output_dir: str,
                                 first_chapter_silence_skip: int = 3,
                                 subsequent_chapter_silence_skip: int = 2,
                                 config_path: str = "config.yaml",
                                 task_language: str = "eng",
                                 snap_to_silence: bool = False,
                                 silence_thresh: int = -40,
                                 min_silence_len: int = 500,
                                 silence_retained: float = 0.2,
                                 max_snap_distance_before: int = 1000,
                                 max_snap_distance_after: int = 1000,
                                 romanize: bool = False):
        """
        Align multiple MP3 files in a folder with text from GitLab using aeneas.

        This method processes all MP3 files in a folder, extracts book and chapter information
        from filenames using regex, fetches corresponding text from GitLab, and performs
        alignment using aeneas. All output segments go to a single audio folder with a
        combined manifest.csv file.

        Args:
            audio_folder: Path to folder containing MP3 files
            book_chapter_regex: Regular expression with named groups (?P<book>...) and (?P<chapter>...)
            book_map_csv: Path to CSV file with columns 'from_book' and 'to_book' for book name mapping
            output_dir: Output directory for all results
            first_chapter_silence_skip: Number of silences to skip for first chapter (chapter 1) (default: 2)
            subsequent_chapter_silence_skip: Number of silences to skip for other chapters (default: 1)
            config_path: Path to GitLab config YAML file (default: "config.yaml")
            task_language: Language code for aeneas alignment (default: "eng")
            snap_to_silence: If True, adjust fragment boundaries to nearest silence regions (default: False)
            silence_thresh: Silence threshold in dB (default: -40)
            min_silence_len: Minimum silence length in ms (default: 500)
            silence_retained: Fraction of silence to retain (0.0-1.0, default: 0.2)
            max_snap_distance_before: Maximum distance in ms to search before boundary (default: 1000)
            max_snap_distance_after: Maximum distance in ms to search after boundary (default: 1000)
            romanize: If True, apply uroman romanization to text (default: False)
        """
        print("=" * 80)
        print("Batch Audio-Text Alignment Tool (Aeneas)")
        print("=" * 80)
        print(f"Audio folder: {audio_folder}")
        print(f"Output directory: {output_dir}")
        print()

        # Validate inputs
        if not os.path.exists(audio_folder):
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
        if not os.path.exists(book_map_csv):
            raise FileNotFoundError(f"Book mapping CSV not found: {book_map_csv}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load book name mapping
        print(f"Loading book name mapping from: {book_map_csv}")
        book_mapping = {}
        with open(book_map_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                book_mapping[row['from_book']] = row['to_book']
        print(f"✓ Loaded {len(book_mapping)} book name mappings")
        print()

        # Compile regex pattern
        try:
            pattern = re.compile(book_chapter_regex)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        # Find all MP3 files and sort alphabetically
        mp3_files = sorted([f for f in os.listdir(audio_folder) if f.lower().endswith('.mp3')])

        if not mp3_files:
            raise ValueError(f"No MP3 files found in {audio_folder}")

        print(f"Found {len(mp3_files)} MP3 file(s) to process")
        print()

        # Parse filenames and validate
        file_info = []
        for mp3_file in mp3_files:
            match = pattern.search(mp3_file)
            if not match:
                raise ValueError(f"Filename '{mp3_file}' does not match regex pattern: {book_chapter_regex}")

            book_name = match.group('book')
            chapter_str = match.group('chapter')

            try:
                chapter_num = int(chapter_str)
            except ValueError as exc:
                raise ValueError(f"Chapter '{chapter_str}' in filename '{mp3_file}' is not a valid integer") from exc

            # Map book name
            if book_name not in book_mapping:
                raise ValueError(f"Book name '{book_name}' from file '{mp3_file}' not found in book mapping CSV")

            mapped_book = book_mapping[book_name]

            file_info.append({
                'filename': mp3_file,
                'path': os.path.join(audio_folder, mp3_file),
                'book': book_name,
                'mapped_book': mapped_book,
                'chapter': chapter_num
            })

        print("File parsing complete:")
        for info in file_info:
            print(f"  {info['filename']}: {info['book']} (→ {info['mapped_book']}) Chapter {info['chapter']}")
        print()

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        audio_output_dir = os.path.join(output_dir, 'audio')
        os.makedirs(audio_output_dir, exist_ok=True)


        print("✓ GitLab connection established")
        print()

        # Process each file
        all_manifest_rows = []
        total_segments = 0
        temp_files_to_cleanup = []

        for idx, info in enumerate(file_info, 1):
            print("=" * 80)
            print(f"Processing file {idx}/{len(file_info)}: {info['filename']}")
            print(f"Book: {info['mapped_book']}, Chapter: {info['chapter']}")
            print("=" * 80)

            try:
                # Create temporary text file for this chapter
                temp_text_fd, temp_text_path = tempfile.mkstemp(suffix='.txt', prefix=f'chapter_{info["mapped_book"]}_{info["chapter"]}_')
                os.close(temp_text_fd)
                temp_files_to_cleanup.append(temp_text_path)

                # Extract chapter text from GitLab
                print("Extracting chapter text from GitLab...")
                self.extract_chapter(
                    book_name=info['mapped_book'],
                    chapter_number=info['chapter'],
                    output_file=temp_text_path,
                    config_path=config_path,
                    romanize=romanize
                )

                # Determine number of silences to skip
                if info['chapter'] == 1:
                    num_silences_to_skip = first_chapter_silence_skip
                else:
                    num_silences_to_skip = subsequent_chapter_silence_skip

                # Trim audio if needed
                audio_to_process = info['path']
                temp_audio_path = None

                if num_silences_to_skip > 0:
                    temp_audio_path = self._trim_audio_after_nth_silence(
                        info['path'], num_silences_to_skip, silence_thresh, min_silence_len
                    )
                    if temp_audio_path:
                        audio_to_process = temp_audio_path
                        temp_files_to_cleanup.append(temp_audio_path)

                # Create temporary output directory for this file
                temp_output_fd, temp_output_marker = tempfile.mkstemp(prefix='aeneas_output_')
                os.close(temp_output_fd)
                os.remove(temp_output_marker)
                temp_output_dir = temp_output_marker + '_dir'
                os.makedirs(temp_output_dir, exist_ok=True)
                temp_files_to_cleanup.append(temp_output_dir)

                # Run aeneas alignment
                print("\nRunning aeneas alignment...")
                self.align_with_aeneas(
                    mp3_file=audio_to_process,
                    text_file=temp_text_path,
                    output_dir=temp_output_dir,
                    task_language=task_language,
                    snap_to_silence=snap_to_silence,
                    silence_thresh=silence_thresh,
                    min_silence_len=min_silence_len,
                    silence_retained=silence_retained,
                    max_snap_distance_before=max_snap_distance_before,
                    max_snap_distance_after=max_snap_distance_after
                )

                # Read the temporary manifest
                temp_manifest_path = os.path.join(temp_output_dir, 'manifest.csv')
                with open(temp_manifest_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    temp_rows = list(reader)

                # Move audio files and update manifest rows
                segment_num = 0
                for row in temp_rows:
                    # Get original audio filename
                    temp_audio_file = os.path.join(temp_output_dir, row['file_name'])

                    # Generate new filename with book and chapter
                    new_filename = f"{info['mapped_book']}_{info['chapter']}_{segment_num:03d}.wav"
                    new_path = os.path.join(audio_output_dir, new_filename)

                    # Move the audio file
                    os.rename(temp_audio_file, new_path)

                    # Update manifest row
                    all_manifest_rows.append({
                        'book': info['mapped_book'],
                        'chapter': info['chapter'],
                        'file_name': os.path.join('audio', new_filename),
                        'transcription': row['transcription'],
                        'text_length': row['text_length'],
                        'audio_duration': row['audio_duration'],
                        'start_adjustment': row['start_adjustment'],
                        'end_adjustment': row['end_adjustment']
                    })

                    segment_num += 1

                total_segments += segment_num
                print(f"\n✓ Processed {segment_num} segments from {info['filename']}")

            except Exception as e:
                print(f"\n✗ Error processing {info['filename']}: {e}")
                raise

        # Write combined manifest.csv
        print("\n" + "=" * 80)
        print("Writing combined manifest...")
        manifest_path = os.path.join(output_dir, 'manifest.csv')

        with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['book', 'chapter', 'file_name', 'transcription', 'text_length',
                           'audio_duration', 'start_adjustment', 'end_adjustment'])

            for row in all_manifest_rows:
                writer.writerow([
                    row['book'],
                    row['chapter'],
                    row['file_name'],
                    row['transcription'],
                    row['text_length'],
                    row['audio_duration'],
                    row['start_adjustment'],
                    row['end_adjustment']
                ])

        print(f"✓ Combined manifest saved to: {manifest_path}")

        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                else:
                    os.remove(temp_file)
            except Exception as e:
                print(f"  Warning: Could not remove {temp_file}: {e}")

        print("✓ Cleanup complete")

        # Print final summary
        print("\n" + "=" * 80)
        print("Batch Alignment Summary")
        print("=" * 80)
        print(f"Total files processed: {len(file_info)}")
        print(f"Total segments created: {total_segments}")
        print(f"Average segments per file: {total_segments / len(file_info):.1f}")
        print("\nOutput files:")
        print(f"  - Combined manifest: {manifest_path}")
        print(f"  - Audio segments: {audio_output_dir}/ ({total_segments} WAV files)")
        print("\n" + "=" * 80)
        print("Batch alignment complete!")
        print("=" * 80)


def main():
    """Entry point for Fire CLI."""
    fire.Fire(AudioTextAligner)


if __name__ == '__main__':
    main()
