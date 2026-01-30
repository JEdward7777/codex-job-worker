#!/usr/bin/env python3
"""
Extract audio-text pairs from CODEX files using edit history.
This looks for the initial-import or earliest edit that contains the config text.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional
from gitlab_to_hf_dataset import GitLabDatasetDownloader


def extract_text_from_edit_history(cell: Dict) -> Optional[str]:
    """Extract config text from edit history.

    Looks for the initial-import edit or the earliest edit with plain text.
    """
    metadata = cell.get('metadata', {})
    edits = metadata.get('edits', [])

    if not edits:
        return None

    # Try to find initial-import edit first
    for edit in edits:
        if edit.get('type') == 'initial-import':
            value = edit.get('value', '')
            # Check if it's plain text (not HTML)
            if value and not value.strip().startswith('<'):
                return value.strip()

    # If no initial-import, try the first edit with plain text
    for edit in edits:
        value = edit.get('value', '')
        if value and not value.strip().startswith('<'):
            return value.strip()

    return None


def process_codex_with_edit_history(downloader: GitLabDatasetDownloader,
                                    codex_path: str,
                                    audio_files_map: Dict[str, str]) -> List[Dict]:
    """Process a CODEX file and extract audio-text pairs from edit history."""
    results = []

    codex_data = downloader.download_json_file(codex_path)
    if not codex_data or 'cells' not in codex_data:
        return results

    for cell in codex_data['cells']:
        metadata = cell.get('metadata', {})
        attachments = metadata.get('attachments', {})
        selected_audio_id = metadata.get('selectedAudioId')
        verse_id = metadata.get('id', 'unknown')

        # Get the selected audio attachment
        if selected_audio_id and selected_audio_id in attachments:
            audio_info = attachments[selected_audio_id]

            # Skip deleted audio
            if audio_info.get('isDeleted', False):
                continue

            # Try to get text from edit history
            text = extract_text_from_edit_history(cell)

            # Try to find the actual audio file path
            actual_audio_path = audio_files_map.get(selected_audio_id)

            if actual_audio_path and text:
                results.append({
                    'verse_id': verse_id,
                    'audio_url': actual_audio_path,
                    'audio_id': selected_audio_id,
                    'transcription': text,
                    'metadata': {
                        'duration': audio_info.get('metadata', {}).get('durationSec'),
                        'sample_rate': audio_info.get('metadata', {}).get('sampleRate'),
                    }
                })

    return results


def main():
    """Main function."""
    print("=" * 60)
    print("Extract Audio-Text Pairs from Edit History")
    print("=" * 60)
    print()

    downloader = GitLabDatasetDownloader()

    # Get all items and build audio map
    print("Listing repository files...")
    items = downloader.list_repository_tree()
    audio_files_map = downloader.build_audio_files_map(items)

    # Process OBA.codex
    print("\nProcessing OBA.codex...")
    results = process_codex_with_edit_history(
        downloader,
        'files/target/OBA.codex',
        audio_files_map
    )

    print(f"Found {len(results)} audio-text pairs")

    if results:
        # Show first few examples
        print("\nFirst 3 examples:")
        print("-" * 60)
        for i, pair in enumerate(results[:3], 1):
            print(f"\n{i}. {pair['verse_id']}")
            print(f"   Audio: {Path(pair['audio_url']).name}")
            print(f"   Duration: {pair['metadata']['duration']:.2f}s")
            print(f"   Text: {pair['transcription'][:80]}...")

        # Create output directory
        output_dir = Path(downloader.config['dataset']['output_dir'])
        audio_dir = output_dir / downloader.config['dataset']['audio_dir']
        output_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)

        # Download audio files and create CSV
        csv_path = output_dir / downloader.config['dataset']['csv_filename']

        print(f"\n\nDownloading audio files to {audio_dir}...")
        print(f"Creating CSV at {csv_path}...")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['file_name', 'transcription'])

            for idx, pair in enumerate(results, 1):
                audio_url = pair['audio_url']
                transcription = pair['transcription']

                # Extract filename from URL
                audio_filename = Path(audio_url).name
                output_audio_path = audio_dir / audio_filename

                print(f"[{idx}/{len(results)}] Downloading {audio_filename}...")

                # Download audio file
                if downloader.download_file(audio_url, output_audio_path):
                    writer.writerow([audio_filename, transcription])
                else:
                    print(f"  Failed to download {audio_url}")

        print("\n✓ Dataset created successfully!")
        print(f"  CSV: {csv_path}")
        print(f"  Audio files: {audio_dir}")
        print(f"  Total records: {len(results)}")
    else:
        print("\nNo audio-text pairs found in edit history")


if __name__ == "__main__":
    main()