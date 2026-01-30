#!/usr/bin/env python3
"""
Calculate total duration of audio files in a directory.
Supports various audio formats including m4a, mp3, wav, flac, ogg, webm.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from mutagen import File


def get_audio_duration(file_path: Path) -> float:
    """Get duration of an audio file in seconds.

    Returns 0 if file cannot be read or is not a valid audio file.
    """
    try:
        # Try using ffprobe first (most reliable for all formats including webm)
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                duration_str = result.stdout.strip()
                if duration_str != 'N/A':
                    return float(duration_str)

            # If duration not in metadata, decode the file to get actual duration
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(file_path),
                    '-f', 'null',
                    '-'
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            # Parse duration from ffmpeg output
            for line in result.stderr.split('\n'):
                if 'Duration:' in line and 'N/A' not in line:
                    # Format: Duration: 00:00:05.23, start: 0.000000, bitrate: 128 kb/s
                    duration_part = line.split('Duration:')[1].split(',')[0].strip()
                    h, m, s = duration_part.split(':')
                    return int(h) * 3600 + int(m) * 60 + float(s)
                elif 'time=' in line:
                    # Get the last time value from progress output
                    time_parts = line.split('time=')
                    if len(time_parts) > 1:
                        time_str = time_parts[-1].split()[0]
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 3:
                                h, m, s = parts
                                duration = int(h) * 3600 + int(m) * 60 + float(s)
                                # Keep updating with the latest time value
                                if duration > 0:
                                    last_duration = duration

            # Return the last duration found
            if 'last_duration' in locals():
                return last_duration #pylint: disable=possibly-used-before-assignment

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Fallback to mutagen (handles most formats with duration in metadata)
        try:
            audio = File(file_path)
            if audio is not None and hasattr(audio.info, 'length'):
                return audio.info.length
        except ImportError:
            pass

        return 0

    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return 0


def scan_directory(directory: Path) -> List[Tuple[Path, float]]:
    """Scan directory for audio files and get their durations.

    Returns list of (file_path, duration_seconds) tuples.
    """
    audio_extensions = {'.m4a', '.mp3', '.wav', '.flac', '.ogg', '.webm', '.opus', '.aac'}
    results = []

    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        return results

    print(f"Scanning directory: {directory}")
    print()

    audio_files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]

    if not audio_files:
        print("No audio files found in directory")
        return results

    print(f"Found {len(audio_files)} audio files")
    print("Calculating durations...")
    print()

    for idx, file_path in enumerate(sorted(audio_files), 1):
        duration = get_audio_duration(file_path)
        results.append((file_path, duration))

        if duration > 0:
            print(f"[{idx}/{len(audio_files)}] {file_path.name}: {duration:.2f}s")
        else:
            print(f"[{idx}/{len(audio_files)}] {file_path.name}: Could not read")

    return results


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_audio_duration.py <directory_path>")
        print()
        print("Example:")
        print("  python calculate_audio_duration.py /path/to/audio/files")
        sys.exit(1)

    directory = Path(sys.argv[1])
    results = scan_directory(directory)

    if not results:
        sys.exit(1)

    # Calculate statistics
    total_duration = sum(duration for _, duration in results)
    valid_files = sum(1 for _, duration in results if duration > 0)
    invalid_files = len(results) - valid_files

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(results)}")
    print(f"Valid audio files: {valid_files}")
    if invalid_files > 0:
        print(f"Files that could not be read: {invalid_files}")
    print()
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"Total duration (seconds): {total_duration:.2f}s")
    print(f"Total duration (minutes): {total_duration/60:.2f}m")
    print(f"Total duration (hours): {total_duration/3600:.2f}h")
    print()

    if valid_files > 0:
        avg_duration = total_duration / valid_files
        max_duration = max(duration for _, duration in results if duration > 0)
        print(f"Average duration per file: {format_duration(avg_duration)}")
        print(f"Maximum duration: {format_duration(max_duration)}")
        print()

    # Provide context for TTS/STT training
    print("=" * 60)
    print("TRAINING DATA ASSESSMENT")
    print("=" * 60)

    hours = total_duration / 3600

    if hours < 0.5:
        print("⚠️  Very limited data (< 30 minutes)")
        print("   - Not enough for TTS training")
        print("   - May be sufficient for STT fine-tuning with pre-trained model")
    elif hours < 1:
        print("⚠️  Limited data (30min - 1 hour)")
        print("   - Minimal for TTS (may produce poor quality)")
        print("   - Reasonable for STT fine-tuning")
    elif hours < 5:
        print("✓  Moderate data (1-5 hours)")
        print("   - Acceptable for TTS with limited speaker variation")
        print("   - Good for STT fine-tuning")
    elif hours < 10:
        print("✓✓ Good data (5-10 hours)")
        print("   - Good for TTS training")
        print("   - Very good for STT fine-tuning")
    else:
        print("✓✓✓ Excellent data (10+ hours)")
        print("   - Excellent for both TTS and STT training")


if __name__ == "__main__":
    main()
