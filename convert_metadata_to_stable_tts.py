#!/usr/bin/env python3
"""
Convert metadata.csv to Stable TTS format.

This script reads a metadata.csv file and converts it to the Stable TTS format:
- No headers
- Pipe-delimited (|)
- Format: audio_filename|transcription_text
- Audio paths are relative to the output file location
"""

import argparse
import csv
import os
from pathlib import Path


def find_text_column(headers):
    """
    Find the text column in the CSV headers.

    Args:
        headers: List of column headers

    Returns:
        The name of the text column ('text' or 'transcription')

    Raises:
        ValueError: If neither 'text' nor 'transcription' column is found
    """
    headers_lower = [h.lower() for h in headers]

    if 'text' in headers_lower:
        return headers[headers_lower.index('text')]
    elif 'transcription' in headers_lower:
        return headers[headers_lower.index('transcription')]
    else:
        raise ValueError(
            "Could not find 'text' or 'transcription' column in metadata.csv. "
            f"Available columns: {', '.join(headers)}"
        )


def find_filename_column(headers):
    """
    Find the filename column in the CSV headers.

    Args:
        headers: List of column headers

    Returns:
        The name of the filename column

    Raises:
        ValueError: If 'file_name' column is not found
    """
    headers_lower = [h.lower() for h in headers]

    if 'file_name' in headers_lower:
        return headers[headers_lower.index('file_name')]
    elif 'filename' in headers_lower:
        return headers[headers_lower.index('filename')]
    else:
        raise ValueError(
            "Could not find 'file_name' or 'filename' column in metadata.csv. "
            f"Available columns: {', '.join(headers)}"
        )


def get_relative_audio_path(metadata_csv_path, audio_filename, output_file_path):
    """
    Calculate the relative path from the output file to the audio file.

    Args:
        metadata_csv_path: Path to the input metadata.csv file
        audio_filename: The audio filename from the CSV
        output_file_path: Path to the output file

    Returns:
        Relative path from output file to audio file
    """
    # Get the directory containing the metadata.csv
    metadata_dir = Path(metadata_csv_path).parent

    # Construct the full path to the audio file
    audio_full_path = metadata_dir / audio_filename

    # Get the directory where the output file will be saved
    output_dir = Path(output_file_path).parent.resolve()

    # Calculate relative path from output directory to audio file
    try:
        relative_path = os.path.relpath(audio_full_path, output_dir)
        return relative_path
    except ValueError:
        # If on different drives (Windows), return absolute path
        return str(audio_full_path.resolve())


def convert_metadata_to_stable_tts(metadata_csv_path, output_file_path):
    """
    Convert metadata.csv to Stable TTS format.

    Args:
        metadata_csv_path: Path to the input metadata.csv file
        output_file_path: Path to the output text file
    """
    metadata_csv_path = Path(metadata_csv_path)
    output_file_path = Path(output_file_path)

    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")

    # Create output directory if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the CSV and convert
    with open(metadata_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames

        if not headers:
            raise ValueError("CSV file appears to be empty or has no headers")

        # Find the appropriate columns
        text_column = find_text_column(headers)
        filename_column = find_filename_column(headers)

        print(f"Using text column: '{text_column}'")
        print(f"Using filename column: '{filename_column}'")

        # Write output file
        lines_written = 0
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for row in reader:
                audio_filename = row[filename_column]
                text = row[text_column]

                # Calculate relative path
                relative_audio_path = get_relative_audio_path(
                    metadata_csv_path,
                    audio_filename,
                    output_file_path
                )
                text = text.replace('|', '')
                # Write in format: filename|text
                outfile.write(f"{relative_audio_path}|{text}\n")
                lines_written += 1

        print(f"Successfully converted {lines_written} entries")
        print(f"Output saved to: {output_file_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata.csv to Stable TTS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/metadata.csv output.txt
        """
    )

    parser.add_argument(
        'metadata_csv',
        help='Path to the input metadata.csv file'
    )

    parser.add_argument(
        'output_file',
        help='Path to the output text file (e.g., example.txt)'
    )

    args = parser.parse_args()

    try:
        convert_metadata_to_stable_tts(args.metadata_csv, args.output_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())