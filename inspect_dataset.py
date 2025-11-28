#!/usr/bin/env python3
"""
Inspect and validate the prepared HuggingFace dataset.
Tests that audio and text data are properly loaded and accessible.
"""

import sys
from pathlib import Path
from datasets import load_from_disk
import numpy as np
import soundfile as sf

def inspect_dataset(dataset_path: str):
    """Load and inspect the HuggingFace dataset."""
    
    print(f"Loading dataset from: {dataset_path}")
    print("=" * 80)
    
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        
        print("\n📊 DATASET STRUCTURE")
        print("-" * 80)
        print(f"Splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  - Number of examples: {len(split_data)}")
            print(f"  - Features: {list(split_data.features.keys())}")
            print(f"  - Feature types: {split_data.features}")
        
        # Test loading a few examples from each split
        print("\n" + "=" * 80)
        print("🔍 TESTING DATA LOADING")
        print("-" * 80)
        
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} split - Testing first 3 examples:")
            
            num_to_test = min(3, len(split_data))
            for i in range(num_to_test):
                example = split_data[i]
                
                print(f"\n  Example {i+1}:")
                print(f"    File: {example['file_name']}")
                print(f"    Text: {example['text'][:100]}{'...' if len(example['text']) > 100 else ''}")
                print(f"    Speaker ID: {example['speaker_id']}")
                print(f"    Duration: {example['duration']:.2f} seconds")
                
                # Check audio data
                audio_data = example['audio']
                if audio_data is not None:
                    audio_array = audio_data['array']
                    sample_rate = audio_data['sampling_rate']
                    
                    print(f"    Audio:")
                    print(f"      - Sample rate: {sample_rate} Hz")
                    print(f"      - Array shape: {audio_array.shape}")
                    print(f"      - Array dtype: {audio_array.dtype}")
                    print(f"      - Array min/max: {audio_array.min():.4f} / {audio_array.max():.4f}")
                    print(f"      - Array mean: {audio_array.mean():.4f}")
                    print(f"      - Non-zero samples: {np.count_nonzero(audio_array)} / {len(audio_array)}")
                    
                    # Verify duration matches
                    calculated_duration = len(audio_array) / sample_rate
                    duration_diff = abs(calculated_duration - example['duration'])
                    print(f"      - Calculated duration: {calculated_duration:.2f} seconds")
                    print(f"      - Duration match: {'✓' if duration_diff < 0.1 else '✗'} (diff: {duration_diff:.3f}s)")
                    
                    # Check if audio is not silent
                    is_silent = np.abs(audio_array).max() < 0.001
                    print(f"      - Audio status: {'⚠️  SILENT' if is_silent else '✓ Has audio signal'}")
                else:
                    print(f"    Audio: ❌ None")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("-" * 80)
        
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()}:")
            
            # Get all durations
            durations = [ex['duration'] for ex in split_data]
            total_duration = sum(durations)
            
            # Get all text lengths
            text_lengths = [len(ex['text']) for ex in split_data]
            
            # Get unique speakers
            speaker_ids = set(ex['speaker_id'] for ex in split_data)
            
            print(f"  Examples: {len(split_data)}")
            print(f"  Speakers: {len(speaker_ids)} unique ({sorted(speaker_ids)})")
            print(f"  Total duration: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)")
            print(f"  Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
            print(f"  Average duration: {np.mean(durations):.2f}s (±{np.std(durations):.2f}s)")
            print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)} chars")
            print(f"  Average text length: {np.mean(text_lengths):.1f} chars (±{np.std(text_lengths):.1f})")
        
        # Validation checks
        print("\n" + "=" * 80)
        print("✅ VALIDATION CHECKS")
        print("-" * 80)
        
        all_checks_passed = True
        
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()}:")
            
            # Check 1: All examples have audio
            has_audio = all(ex['audio'] is not None for ex in split_data)
            print(f"  {'✓' if has_audio else '✗'} All examples have audio data")
            all_checks_passed &= has_audio
            
            # Check 2: All examples have text
            has_text = all(ex['text'] and len(ex['text']) > 0 for ex in split_data)
            print(f"  {'✓' if has_text else '✗'} All examples have text")
            all_checks_passed &= has_text
            
            # Check 3: All examples have speaker IDs
            has_speaker = all(ex['speaker_id'] is not None for ex in split_data)
            print(f"  {'✓' if has_speaker else '✗'} All examples have speaker IDs")
            all_checks_passed &= has_speaker
            
            # Check 4: Audio is not silent (sample first 10)
            sample_size = min(10, len(split_data))
            non_silent = sum(1 for ex in split_data.select(range(sample_size)) 
                           if np.abs(ex['audio']['array']).max() > 0.001)
            print(f"  {'✓' if non_silent == sample_size else '⚠️ '} Audio has signal ({non_silent}/{sample_size} samples checked)")
            
            # Check 5: Sample rates are consistent
            sample_rates = set(ex['audio']['sampling_rate'] for ex in split_data.select(range(sample_size)))
            print(f"  {'✓' if len(sample_rates) == 1 else '✗'} Consistent sample rate: {sample_rates}")
            all_checks_passed &= len(sample_rates) == 1
        
        print("\n" + "=" * 80)
        if all_checks_passed:
            print("🎉 DATASET IS VALID AND READY FOR TRAINING!")
        else:
            print("⚠️  DATASET HAS SOME ISSUES - Review the checks above")
        print("=" * 80)
        
        return 0 if all_checks_passed else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Default path from xiang_tts.yaml
        dataset_path = "/home/lansford/vast_ai_tts/dataset/Xiang/hf_dataset"
    
    exit_code = inspect_dataset(dataset_path)
    sys.exit(exit_code)