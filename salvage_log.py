#!/usr/bin/env python3
"""Salvage structured JSON data from a log file into a CSV.

Reads a log file line-by-line, extracts every line that is a valid JSON object,
and writes all key-value pairs into a CSV with dynamically discovered,
alphabetically sorted headers.

Usage:
    python salvage_log.py <input_log> <output_csv>

Lines that are not valid JSON objects are silently skipped.
Nested dicts/lists are serialized as JSON strings in the CSV cell.
"""

import argparse
import ast
import csv
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract JSON objects from a log file into a CSV."
    )
    parser.add_argument("input_log", help="Path to the input log file")
    parser.add_argument("output_csv", help="Path to the output CSV file")
    return parser.parse_args()


def try_parse_json(line: str) -> dict | None:
    """Return parsed dict if *line* is a JSON object, else None.

    First attempts strict JSON parsing.  If that fails, falls back to
    ``ast.literal_eval`` which handles Python-style dict literals
    (single-quoted keys/values, True/False/None instead of
    true/false/null, etc.).
    """
    stripped = line.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None

    # Attempt 1: strict JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: Python dict literal (e.g. single-quoted keys)
    try:
        obj = ast.literal_eval(stripped)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass

    return None


def serialize_value(value):
    """Return a CSV-safe string for *value*.

    Dicts and lists are serialized as compact JSON strings.
    Everything else is converted with str().
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    args = parse_args()

    # --- Pass 1: collect all rows and discover every key ---
    rows: list[dict] = []
    all_keys: set[str] = set()

    try:
        with open(args.input_log, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = try_parse_json(line)
                if obj is not None:
                    rows.append(obj)
                    all_keys.update(obj.keys())
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input_log}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Error reading input file: {exc}", file=sys.stderr)
        sys.exit(1)

    # Sort headers alphabetically
    headers = sorted(all_keys)

    # --- Pass 2: write CSV ---
    # Ensure parent directories exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                # Serialize any nested structures
                sanitized = {k: serialize_value(v) for k, v in row.items()}
                writer.writerow(sanitized)
    except OSError as exc:
        print(f"Error writing output file: {exc}", file=sys.stderr)
        sys.exit(1)

    json_count = len(rows)
    col_count = len(headers)
    print(f"Done — {json_count} JSON rows extracted, {col_count} columns, written to {args.output_csv}")


if __name__ == "__main__":
    main()
