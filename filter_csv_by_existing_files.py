#!/usr/bin/env python3
"""
Filter CSV rows based on file existence in a specified folder.

By default, this script removes rows where the referenced filename exists
in a specified folder. With --keep-existing, it keeps only rows where files
exist and removes rows where files don't exist. The path portion of filenames
in the CSV is ignored for comparison (only basenames are checked).
"""

import argparse
import csv
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Filter CSV rows based on file existence in a specified folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Remove rows for files that exist (default behavior)
  %(prog)s input.csv output.csv /path/to/validation --column-name audio_file

  # Keep only rows for files that exist (inverted)
  %(prog)s input.csv output.csv ./validation_folder --keep-existing
        """
    )

    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "output_csv",
        help="Path to the output CSV file"
    )

    parser.add_argument(
        "folder_path",
        help="Path to the folder containing files to check against"
    )

    parser.add_argument(
        "--column-name",
        default="file_name",
        help="Name of the column containing filenames (default: file_name)"
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Invert filter: keep rows where files exist, remove rows where files don't exist"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Validate folder exists
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder '{args.folder_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Get list of files in the folder (non-recursive, basenames without extensions)
    print(f"Scanning folder: {args.folder_path}")
    existing_files = set()
    try:
        for item in os.listdir(args.folder_path):
            item_path = os.path.join(args.folder_path, item)
            if os.path.isfile(item_path):
                # Store filename without extension
                filename_without_ext = os.path.splitext(item)[0]
                existing_files.add(filename_without_ext)
        print(f"Found {len(existing_files)} files in folder (extensions ignored)")
    except Exception as e:
        print(f"Error reading folder '{args.folder_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Process CSV
    try:
        with open(args.input_csv, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            # Check if column exists
            if args.column_name not in reader.fieldnames:
                print(f"Error: Column '{args.column_name}' not found in CSV.", file=sys.stderr)
                print(f"Available columns: {', '.join(reader.fieldnames)}", file=sys.stderr)
                sys.exit(1)

            # Read all rows
            rows = list(reader)
            fieldnames = reader.fieldnames

        # Filter rows
        kept_rows = []
        removed_count = 0

        mode_desc = "KEEP rows with existing files, REMOVE rows with missing files" if args.keep_existing else "REMOVE rows with existing files, KEEP rows with missing files"
        print(f"\nProcessing {len(rows)} rows from input CSV...")
        print(f"Filtering based on column: '{args.column_name}'")
        print(f"Mode: {mode_desc}\n")

        for row in rows:
            filename_value = row[args.column_name]
            # Extract basename (ignore path)
            basename = os.path.basename(filename_value)
            # Remove extension for comparison
            basename_without_ext = os.path.splitext(basename)[0]

            file_exists = basename_without_ext in existing_files

            # Apply filter logic based on mode
            if args.keep_existing:
                # Keep existing, remove missing
                if file_exists:
                    print(f"KEPT: {basename} (file exists in folder)")
                    kept_rows.append(row)
                else:
                    print(f"FILTERED OUT: {basename} (file not found in folder)")
                    removed_count += 1
            else:
                # Default: Remove existing, keep missing
                if file_exists:
                    print(f"FILTERED OUT: {basename} (matches file in folder, ignoring extension)")
                    removed_count += 1
                else:
                    print(f"KEPT: {basename}")
                    kept_rows.append(row)

        # Write output CSV
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

        # Summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Input rows:     {len(rows)}")
        print(f"  Rows kept:      {len(kept_rows)}")
        print(f"  Rows removed:   {removed_count}")
        print(f"  Output written: {args.output_csv}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error processing CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
