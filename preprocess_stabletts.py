#!/usr/bin/env python3
"""
Preprocess audio dataset for StableTTS training
Reads metadata.csv format and generates mel spectrograms and phonemes

This module provides both a command-line interface and a Python API.

Python API Usage:
    from preprocess_stabletts import preprocess_stabletts_api

    result = preprocess_stabletts_api(
        input_csv='/path/to/metadata.csv',
        audio_base_dir='/path/to/audio',
        output_json='/path/to/output.json',
        output_feature_dir='/path/to/features',
        language='english',
        use_uroman=False,
        uroman_language=None,  # None for auto-detection
        resample=False,
        num_workers=2,
        heartbeat_callback=lambda: None  # Optional callback for long-running jobs
    )
"""
import os
import sys
import json
import csv
import argparse
from typing import Optional, Callable, Dict, Any
from dataclasses import asdict
import traceback
from tqdm import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method
import torchaudio

# Import uroman for text romanization
from uroman import Uroman

# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from config import MelConfig #pylint: disable=import-error, wrong-import-position
from utils.audio import LogMelSpectrogram, load_and_resample_audio #pylint: disable=import-error, wrong-import-position

from text.mandarin import chinese_to_cnm3 #pylint: disable=import-error, wrong-import-position
from text.english import english_to_ipa2 #pylint: disable=import-error, wrong-import-position
from text.japanese import japanese_to_ipa2 #pylint: disable=import-error, wrong-import-position

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

g2p_mapping = {
    'chinese': chinese_to_cnm3,
    'japanese': japanese_to_ipa2,
    'english': english_to_ipa2,
}

# Global variables for multiprocessing
mel_extractor = None
g2p = None
mel_config = None
output_mel_dir = None
output_wav_dir = None
resample = None
output_json_dir = None
uroman_instance = None
uroman_lang = None
use_uroman_flag = False

def init_worker(mel_cfg, language, output_mel, output_wav, do_resample, json_dir, use_uroman=False, uroman_language=None):
    """Initialize worker process with shared resources"""
    global mel_extractor, g2p, mel_config, output_mel_dir, output_wav_dir, resample, output_json_dir
    global uroman_instance, uroman_lang, use_uroman_flag
    mel_config = mel_cfg
    mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)
    g2p = g2p_mapping.get(language)
    output_mel_dir = output_mel
    output_wav_dir = output_wav
    resample = do_resample
    output_json_dir = json_dir

    # Initialize uroman if enabled
    use_uroman_flag = use_uroman
    uroman_lang = uroman_language
    if use_uroman:
        uroman_instance = Uroman()
    else:
        uroman_instance = None

def load_metadata_csv(csv_path, audio_base_dir):
    """Load metadata from CSV file"""
    file_list = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            audio_path = row['file_name']
            text = row['transcription']

            # Resolve audio path relative to audio_base_dir
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_base_dir, audio_path)

            file_list.append((str(idx), audio_path, text))

    return file_list

@torch.inference_mode()
def process_filelist(line):
    """Process a single audio file"""
    idx, audio_path, text = line
    original_text = text  # Store original text before any processing

    try:
        audio = load_and_resample_audio(audio_path, mel_config.sample_rate, device=device)
        if audio is None:
            print(f'Failed to load audio: {audio_path}')
            return None

        # Get output path
        audio_name, _ = os.path.splitext(os.path.basename(audio_path))

        # Apply uroman romanization if enabled (before G2P)
        if use_uroman_flag and uroman_instance is not None:
            try:
                text = uroman_instance.romanize_string(text, lcode=uroman_lang)
                if not text or len(text.strip()) == 0:
                    print(f'Uroman returned empty text for: {audio_path}')
                    return None
            except Exception as uroman_error:
                print(f'Uroman failed for {audio_path}: {str(uroman_error)}')
                return None

        # Convert text to phonemes
        phone = g2p(text)
        if len(phone) == 0:
            print(f'Empty phoneme sequence for: {audio_path}')
            return None

        # Extract mel spectrogram
        mel = mel_extractor(audio.to(device)).cpu().squeeze(0)
        output_mel_path = os.path.join(output_mel_dir, f'{idx}_{audio_name}.pt')
        torch.save(mel, output_mel_path)

        # Optionally save resampled audio
        if resample:
            output_audio_path = os.path.join(output_wav_dir, f'{idx}_{audio_name}.wav')
            torchaudio.save(output_audio_path, audio.cpu(), mel_config.sample_rate)
        else:
            output_audio_path = audio_path

        # Make paths relative to the JSON file location
        mel_path_relative = os.path.relpath(output_mel_path, output_json_dir)
        audio_path_relative = os.path.relpath(output_audio_path, output_json_dir)

        # Build output dictionary
        output_data = {
            'mel_path': mel_path_relative,
            'phone': phone,
            'audio_path': audio_path_relative,
            'text': text,  # This is romanized text if uroman was used
            'mel_length': mel.size(-1)
        }

        # Add original text if uroman was used
        if use_uroman_flag:
            output_data['unuromanized_text'] = original_text

        return json.dumps(output_data, ensure_ascii=False, allow_nan=False)

    except Exception as e:
        print(f'Error processing {audio_path}: {str(e)}')
        return None


def preprocess_stabletts_api(
    input_csv: str,
    audio_base_dir: str,
    output_json: str,
    output_feature_dir: str,
    language: str = 'english',
    use_uroman: bool = False,
    uroman_language: Optional[str] = None,
    resample: bool = False,
    num_workers: int = 2,
    heartbeat_callback: Optional[Callable[[], None]] = None
) -> Dict[str, Any]:
    """
    Python API for preprocessing audio dataset for StableTTS training.

    Args:
        input_csv: Path to input metadata.csv file
        audio_base_dir: Base directory containing audio files
        output_json: Path to output JSON filelist
        output_feature_dir: Directory to save mel spectrograms
        language: Language for text-to-phoneme conversion ('chinese', 'english', 'japanese')
        use_uroman: Whether to apply uroman romanization before G2P conversion
        uroman_language: Language code for uroman (e.g., 'zho' for Chinese, None for auto-detection)
        resample: Whether to save resampled audio files
        num_workers: Number of worker processes
        heartbeat_callback: Optional callback function to call periodically for long-running jobs

    Returns:
        Dictionary with:
            - success: bool
            - processed_count: int
            - total_count: int
            - output_json: str (path to output file)
            - error_message: str (if success is False)
    """
    try:
        # Validate language
        if language not in g2p_mapping:
            return {
                'success': False,
                'processed_count': 0,
                'total_count': 0,
                'output_json': output_json,
                'error_message': f"Unsupported language: {language}. Choose from: {list(g2p_mapping.keys())}"
            }

        # Initialize configs
        mel_cfg = MelConfig()

        # Create output directories
        output_mel_dir = os.path.join(output_feature_dir, 'mels')
        os.makedirs(output_mel_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        output_wav_dir = None
        if resample:
            output_wav_dir = os.path.join(output_feature_dir, 'waves')
            os.makedirs(output_wav_dir, exist_ok=True)

        # Load filelist
        print(f"Loading metadata from: {input_csv}")
        input_filelist = load_metadata_csv(input_csv, audio_base_dir)
        total_count = len(input_filelist)
        print(f"Found {total_count} audio files")
        print(f"Language: {language}")
        print(f"Use uroman: {use_uroman}")
        if use_uroman:
            print(f"Uroman language: {uroman_language or 'auto-detect'}")
        print(f"Output directory: {output_feature_dir}")
        print(f"Device: {device}")

        # Get the directory of the output JSON for relative path calculation
        json_dir = os.path.dirname(os.path.abspath(output_json))

        # Process files
        set_start_method('spawn', force=True)
        results = []

        with Pool(processes=num_workers,
                  initializer=init_worker,
                  initargs=(mel_cfg, language, output_mel_dir, output_wav_dir, resample, json_dir, use_uroman, uroman_language)) as pool:

            for idx, result in enumerate(tqdm(pool.imap(process_filelist, input_filelist), total=total_count)):
                if result is not None:
                    results.append(f'{result}\n')

                # Call heartbeat callback periodically (every 100 files)
                if heartbeat_callback and idx % 100 == 0:
                    heartbeat_callback()

        # Save output filelist
        with open(output_json, 'w', encoding='utf-8') as f:
            f.writelines(results)

        processed_count = len(results)
        print("\nPreprocessing complete!")
        print(f"Processed {processed_count} / {total_count} files successfully")
        print(f"Output saved to: {output_json}")

        return {
            'success': True,
            'processed_count': processed_count,
            'total_count': total_count,
            'output_json': output_json,
            'error_message': None
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error during preprocessing: {error_msg}")
        return {
            'success': False,
            'processed_count': 0,
            'total_count': 0,
            'output_json': output_json,
            'error_message': error_msg
        }


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio dataset for StableTTS')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input metadata.csv file')
    parser.add_argument('--audio_base_dir', type=str, required=True,
                        help='Base directory containing audio files')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to output JSON filelist')
    parser.add_argument('--output_feature_dir', type=str, required=True,
                        help='Directory to save mel spectrograms')
    parser.add_argument('--language', type=str, default='english',
                        choices=['chinese', 'english', 'japanese'],
                        help='Language for text-to-phoneme conversion')
    parser.add_argument('--use_uroman', action='store_true',
                        help='Enable uroman text romanization before G2P conversion')
    parser.add_argument('--uroman_language', type=str, default=None,
                        help='Language code for uroman (e.g., zho for Chinese). If not specified, auto-detection is used.')
    parser.add_argument('--resample', action='store_true',
                        help='Save resampled audio files')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker processes')

    args = parser.parse_args()

    # Initialize configs
    mel_config = MelConfig()

    # Create output directories
    output_mel_dir = os.path.join(args.output_feature_dir, 'mels')
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    output_wav_dir = None
    if args.resample:
        output_wav_dir = os.path.join(args.output_feature_dir, 'waves')
        os.makedirs(output_wav_dir, exist_ok=True)

    # Load filelist
    print(f"Loading metadata from: {args.input_csv}")
    input_filelist = load_metadata_csv(args.input_csv, args.audio_base_dir)
    print(f"Found {len(input_filelist)} audio files")
    print(f"Language: {args.language}")
    print(f"Use uroman: {args.use_uroman}")
    if args.use_uroman:
        print(f"Uroman language: {args.uroman_language or 'auto-detect'}")
    print(f"Output directory: {args.output_feature_dir}")
    print(f"Device: {device}")

    # Get the directory of the output JSON for relative path calculation
    output_json_dir = os.path.dirname(os.path.abspath(args.output_json))

    # Process files
    set_start_method('spawn', force=True)
    results = []

    with Pool(processes=args.num_workers,
              initializer=init_worker,
              initargs=(mel_config, args.language, output_mel_dir, output_wav_dir, args.resample, output_json_dir, args.use_uroman, args.uroman_language)) as pool:
        for result in tqdm(pool.imap(process_filelist, input_filelist), total=len(input_filelist)):
            if result is not None:
                results.append(f'{result}\n')

    # Save output filelist
    with open(args.output_json, 'w', encoding='utf-8') as f:
        f.writelines(results)

    print("\nPreprocessing complete!")
    print(f"Processed {len(results)} / {len(input_filelist)} files successfully")
    print(f"Output saved to: {args.output_json}")

# Optimize for multiprocessing
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()