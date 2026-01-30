#!/usr/bin/env python3
"""
Convert a text file to CSV by splitting each line at a Bible verse reference.

The verse reference becomes the ID column,
and the content after becomes the text column.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def split_at_verse_reference(line: str, pattern: str) -> tuple[str, str]:
    """
    Split a line at a verse reference pattern using a regex.

    Args:
        line: The line to split
        pattern: Regex pattern to match the verse reference

    Returns:
        A tuple of (verse_reference, text_content)
    """
    match = re.match(pattern, line)

    if match:
        verse_ref = match.group(0).strip()
        text_content = line[match.end():].strip()
        return verse_ref, text_content

    # If no match found, return the whole line as ID with empty text
    return line.strip(), ""


def convert_text_to_csv(
    input_file: Path,
    output_file: Path,
    id_column: str,
    text_column: str,
    pattern: str
) -> None:
    """
    Convert a text file to CSV format.

    Args:
        input_file: Path to input text file
        output_file: Path to output CSV file
        id_column: Name of the ID column
        text_column: Name of the text column
        pattern: Regex pattern to match the verse reference
    """
    rows_processed = 0
    rows_skipped = 0

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        # Create CSV writer with proper quoting for fields containing quotes
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow([id_column, text_column])

        # Process each line
        for line in infile:
            # Skip blank lines
            if not line.strip():
                rows_skipped += 1
                continue

            # Split at the verse reference
            verse_ref, text_content = split_at_verse_reference(line, pattern)

            # Write to CSV (csv module handles quote escaping automatically)
            writer.writerow([verse_ref, text_content])
            rows_processed += 1

    print("Conversion complete!")
    print(f"  Rows processed: {rows_processed}")
    print(f"  Blank lines skipped: {rows_skipped}")
    print(f"  Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert text file to CSV by splitting at verse references using regex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default pattern and column names
  %(prog)s input.txt output.csv

  # Custom column names
  %(prog)s input.txt output.csv --id-column verse_ref --text-column content

  # Custom regex pattern
  %(prog)s input.txt output.csv --pattern "^[A-Za-z]+ \d+:\d+ +"

  # All options
  %(prog)s input.txt output.csv -i verse_id -t transcription -p "[^:]*:\d+ +"

Default pattern: [^:]*:\d+ +
  Matches: anything up to and including ":" followed by digits and spaces
  Examples: "Genesis 1:1 ", "I Kings 15:20 ", "1 John 3:16 "
        """
    )

    parser.add_argument(
        'input_file',
        type=Path,
        help='Input text file path'
    )

    parser.add_argument(
        'output_file',
        type=Path,
        help='Output CSV file path'
    )

    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default=r'[^:]*:\d+ +',
        help='Regex pattern to match verse reference (default: [^:]*:\\d+ +)'
    )

    parser.add_argument(
        '-i', '--id-column',
        type=str,
        default='verse_id',
        help='Name of the ID column (default: verse_id)'
    )

    parser.add_argument(
        '-t', '--text-column',
        type=str,
        default='transcription',
        help='Name of the text column (default: transcription)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Validate regex pattern
    try:
        re.compile(args.pattern)
    except re.error as e:
        print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert the file
    try:
        convert_text_to_csv(
            args.input_file,
            args.output_file,
            args.id_column,
            args.text_column,
            args.pattern
        )
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()