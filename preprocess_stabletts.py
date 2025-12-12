#!/usr/bin/env python3
"""
Preprocess audio dataset for StableTTS training
Reads metadata.csv format and generates mel spectrograms and phonemes
"""
import os
import sys
import json
import csv
import argparse
from tqdm import tqdm
from dataclasses import dataclass, asdict

import torch
from torch.multiprocessing import Pool, set_start_method
import torchaudio

# Add StableTTS to path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StableTTS'))

from config import MelConfig, TrainConfig
from utils.audio import LogMelSpectrogram, load_and_resample_audio

from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2

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

def init_worker(mel_cfg, language, output_mel, output_wav, do_resample, json_dir):
    """Initialize worker process with shared resources"""
    global mel_extractor, g2p, mel_config, output_mel_dir, output_wav_dir, resample, output_json_dir
    mel_config = mel_cfg
    mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)
    g2p = g2p_mapping.get(language)
    output_mel_dir = output_mel
    output_wav_dir = output_wav
    resample = do_resample
    output_json_dir = json_dir

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
    
    try:
        audio = load_and_resample_audio(audio_path, mel_config.sample_rate, device=device)
        if audio is None:
            print(f'Failed to load audio: {audio_path}')
            return None
        
        # Get output path
        audio_name, _ = os.path.splitext(os.path.basename(audio_path))
        
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
        
        return json.dumps({
            'mel_path': mel_path_relative,
            'phone': phone,
            'audio_path': audio_path_relative,
            'text': text,
            'mel_length': mel.size(-1)
        }, ensure_ascii=False, allow_nan=False)
        
    except Exception as e:
        print(f'Error processing {audio_path}: {str(e)}')
        return None

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
    print(f"Output directory: {args.output_feature_dir}")
    print(f"Device: {device}")
    
    # Get the directory of the output JSON for relative path calculation
    output_json_dir = os.path.dirname(os.path.abspath(args.output_json))
    
    # Process files
    set_start_method('spawn', force=True)
    results = []
    
    with Pool(processes=args.num_workers,
              initializer=init_worker,
              initargs=(mel_config, args.language, output_mel_dir, output_wav_dir, args.resample, output_json_dir)) as pool:
        for result in tqdm(pool.imap(process_filelist, input_filelist), total=len(input_filelist)):
            if result is not None:
                results.append(f'{result}\n')
    
    # Save output filelist
    with open(args.output_json, 'w', encoding='utf-8') as f:
        f.writelines(results)
    
    print(f"\nPreprocessing complete!")
    print(f"Processed {len(results)} / {len(input_filelist)} files successfully")
    print(f"Output saved to: {args.output_json}")

# Optimize for multiprocessing
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()