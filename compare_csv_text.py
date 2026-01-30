#!/usr/bin/env python3
"""
CSV Text Comparison Script

Compares text columns from two CSV files based on matching key columns,
calculates edit distance and normalized edit distance, and outputs results.
"""

import argparse
import csv
import sys
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs): #pylint: disable=unused-argument
        """Fake tqdm wrapper for iterable"""
        return iterable


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The minimum number of single-character edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1) #pylint: disable=arguments-out-of-order

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def read_csv_data(
    filepath: str,
    key_column: str,
    text_column: str
) -> Tuple[Dict[str, str], List[str], str]:
    """
    Read CSV file and extract key-text pairs.

    Args:
        filepath: Path to CSV file
        key_column: Name of the key column
        text_column: Name of the text column

    Returns:
        Tuple of (data_dict, duplicate_keys, original_key_column_name)
    """
    data = {}
    duplicates = []
    row_count = 0
    null_key_count = 0
    null_text_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Verify columns exist
        if key_column not in reader.fieldnames:
            raise ValueError(f"Key column '{key_column}' not found in {filepath}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Text column '{text_column}' not found in {filepath}")

        # Convert to list to get total count for progress bar
        rows = list(reader)

        for row in tqdm(rows, desc=f"Reading {filepath}", unit="rows", file=sys.stderr):
            row_count += 1
            key = row.get(key_column)
            text = row.get(text_column)

            # Handle null/missing keys
            if key is None or key.strip() == '':
                null_key_count += 1
                continue

            # Handle null/missing text
            if text is None:
                text = ''
                null_text_count += 1

            # Check for duplicates
            if key in data:
                duplicates.append(key)

            # Store data (will overwrite if duplicate, keeping last occurrence)
            data[key] = text

    # Report issues
    if null_key_count > 0:
        print(f"WARNING: {filepath}: Found {null_key_count} rows with null/empty keys (skipped)", file=sys.stderr)

    if null_text_count > 0:
        print(f"WARNING: {filepath}: Found {null_text_count} rows with null text values (treated as empty strings)", file=sys.stderr)

    if duplicates:
        print(f"WARNING: {filepath}: Found {len(duplicates)} duplicate keys (using last occurrence): {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}", file=sys.stderr)

    return data, duplicates, key_column


def compare_texts(
    csv_a_path: str,
    csv_b_path: str,
    a_key: str,
    b_key: str,
    a_text: str,
    b_text: str,
    csv_out_path: str
) -> None:
    """
    Compare text columns from two CSV files and output results.

    Args:
        csv_a_path: Path to first CSV file
        csv_b_path: Path to second CSV file
        a_key: Key column name in csv_a
        b_key: Key column name in csv_b
        a_text: Text column name in csv_a
        b_text: Text column name in csv_b
        csv_out_path: Path to output CSV file
    """
    print("Reading CSV files...", file=sys.stderr)

    # Read both CSV files
    data_a, _dups_a, key_col_name = read_csv_data(csv_a_path, a_key, a_text)
    data_b, _dups_b, _ = read_csv_data(csv_b_path, b_key, b_text)

    print(f"CSV A: {len(data_a)} unique keys", file=sys.stderr)
    print(f"CSV B: {len(data_b)} unique keys", file=sys.stderr)

    # Find matching and non-matching keys
    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())

    matching_keys = keys_a & keys_b
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a

    print(f"\nMatching keys: {len(matching_keys)}", file=sys.stderr)

    # Report non-matching keys
    if only_in_a:
        print(f"\nKeys only in CSV A ({len(only_in_a)}):", file=sys.stderr)
        for key in sorted(list(only_in_a)[:20]):
            print(f"  - {key}", file=sys.stderr)
        if len(only_in_a) > 20:
            print(f"  ... and {len(only_in_a) - 20} more", file=sys.stderr)

    if only_in_b:
        print(f"\nKeys only in CSV B ({len(only_in_b)}):", file=sys.stderr)
        for key in sorted(list(only_in_b)[:20]):
            print(f"  - {key}", file=sys.stderr)
        if len(only_in_b) > 20:
            print(f"  ... and {len(only_in_b) - 20} more", file=sys.stderr)

    # Calculate edit distances for matching keys
    print(f"\nCalculating edit distances for {len(matching_keys)} matching pairs...", file=sys.stderr)

    results = []
    # Create a list of keys to process for progress bar
    keys_to_process = [key for key in data_a if key in matching_keys]

    for key in tqdm(keys_to_process, desc="Calculating distances", unit="pairs", file=sys.stderr):
        text_a_val = data_a[key]
        text_b_val = data_b[key]

        # Calculate edit distance
        edit_dist = levenshtein_distance(text_a_val, text_b_val)

        # Calculate normalized edit distance
        mean_length = (len(text_a_val) + len(text_b_val)) / 2.0
        if mean_length == 0:
            normalized_dist = 0.0
        else:
            normalized_dist = edit_dist / mean_length

        results.append({
            key_col_name: key,
            'text_a': text_a_val,
            'text_b': text_b_val,
            'edit_distance': edit_dist,
            'normalized_edit_distance': normalized_dist
        })

    # Write output CSV
    print(f"\nWriting results to {csv_out_path}...", file=sys.stderr)

    with open(csv_out_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [key_col_name, 'text_a', 'text_b', 'edit_distance', 'normalized_edit_distance']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write with progress bar
        for result in tqdm(results, desc="Writing output", unit="rows", file=sys.stderr):
            writer.writerow(result)

    print(f"\nDone! Wrote {len(results)} rows to {csv_out_path}", file=sys.stderr)

    # Summary statistics
    if results:
        avg_edit_dist = sum(r['edit_distance'] for r in results) / len(results)
        avg_norm_dist = sum(r['normalized_edit_distance'] for r in results) / len(results)
        print("\nSummary Statistics:", file=sys.stderr)
        print(f"  Average edit distance: {avg_edit_dist:.2f}", file=sys.stderr)
        print(f"  Average normalized edit distance: {avg_norm_dist:.4f}", file=sys.stderr)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Compare text columns from two CSV files based on matching keys.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv_a data1.csv --csv_b data2.csv \\
           --a_key id --b_key id \\
           --a_text original --b_text transcribed \\
           --csv_out comparison.csv
        """
    )

    parser.add_argument('--csv_a', required=True, help='Path to first CSV file')
    parser.add_argument('--csv_b', required=True, help='Path to second CSV file')
    parser.add_argument('--a_key', required=True, help='Key column name in csv_a')
    parser.add_argument('--b_key', required=True, help='Key column name in csv_b')
    parser.add_argument('--a_text', required=True, help='Text column name in csv_a')
    parser.add_argument('--b_text', required=True, help='Text column name in csv_b')
    parser.add_argument('--csv_out', required=True, help='Path to output CSV file')

    args = parser.parse_args()

    try:
        compare_texts(
            args.csv_a,
            args.csv_b,
            args.a_key,
            args.b_key,
            args.a_text,
            args.b_text,
            args.csv_out
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
