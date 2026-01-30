#!/usr/bin/env python3
"""
W2V2-BERT ASR Inference Script

Batch transcribe audio files using a fine-tuned Wav2Vec2-BERT or Wav2Vec2 model.

This module provides both a command-line interface and a Python API.

Python API Usage:
    from inference_w2v2bert_asr import inference_w2v2bert_asr_api

    result = inference_w2v2bert_asr_api(
        model_path='/path/to/model',
        audio_files=['/path/to/audio1.wav', '/path/to/audio2.wav'],
        output_csv='/path/to/output.csv'
    )

Command-line Usage:
    python inference_w2v2bert_asr.py \
        --model_path outputs/w2v2bert_asr/final_model \
        --audio_dir data/audio \
        --output_csv transcriptions.csv
"""

import argparse
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor,
)
import librosa

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def detect_model_architecture(model_path: str) -> bool:
    """
    Detect whether the model is traditional Wav2Vec2 or Wav2Vec2-BERT.

    Args:
        model_path: Path to saved model directory

    Returns:
        True if traditional Wav2Vec2, False if Wav2Vec2-BERT
    """
    try:
        config = AutoConfig.from_pretrained(model_path)
        model_type = config.model_type

        # Check model type
        if model_type == "wav2vec2":
            logger.info("Detected traditional Wav2Vec2 architecture")
            return True
        elif model_type == "wav2vec2-bert":
            logger.info("Detected Wav2Vec2-BERT architecture")
            return False
        else:
            logger.warning("Unknown model type: %s, defaulting to Wav2Vec2-BERT", model_type)
            return False
    except Exception as e:
        logger.warning("Could not detect model type from config: %s, defaulting to Wav2Vec2-BERT", e)
        return False


def load_model_and_processor(
    model_path: str, device: str = "cuda", use_wav2vec2_base: Optional[bool] = None
) -> Tuple[Union[Wav2Vec2ForCTC, Wav2Vec2BertForCTC], Union[Wav2Vec2Processor, Wav2Vec2BertProcessor], bool]:
    """
    Load trained model and processor with auto-detection of architecture.

    Args:
        model_path: Path to saved model directory (can be checkpoint dir or parent dir with tokenizer)
        device: Device to load model on
        use_wav2vec2_base: Force architecture type (None for auto-detect)

    Returns:
        Tuple of (model, processor, use_wav2vec2_base)
    """
    logger.info("Loading model from %s", model_path)

    # Check if model_path is a checkpoint directory or the parent directory
    # Checkpoint dirs have model.safetensors but no vocab.json
    # Parent dirs have vocab.json but no model.safetensors
    has_model = os.path.exists(os.path.join(model_path, "model.safetensors")) or os.path.exists(
        os.path.join(model_path, "pytorch_model.bin")
    )
    has_vocab = os.path.exists(os.path.join(model_path, "vocab.json"))

    if has_model and not has_vocab:
        # This is a checkpoint directory - need to find parent for tokenizer
        logger.info("Detected checkpoint directory - looking for tokenizer in parent directory")
        parent_dir = os.path.dirname(model_path)
        processor_path = parent_dir
        model_load_path = model_path
    elif has_vocab and not has_model:
        # This is the parent directory with tokenizer but no model
        # Need to find the latest checkpoint
        logger.info("Detected parent directory - looking for latest checkpoint")
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise ValueError(f"No checkpoint directories found in {model_path}")
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        processor_path = model_path
        model_load_path = os.path.join(model_path, latest_checkpoint)
        logger.info("Using latest checkpoint: %s", latest_checkpoint)
    elif has_vocab and has_model:
        # Both are present - use the same path for both
        processor_path = model_path
        model_load_path = model_path
    else:
        raise ValueError(f"Invalid model path: {model_path} - missing both model and tokenizer files")

    # Auto-detect architecture if not specified
    if use_wav2vec2_base is None:
        use_wav2vec2_base = detect_model_architecture(model_load_path)

    model_type = "Wav2Vec2" if use_wav2vec2_base else "Wav2Vec2-BERT"
    logger.info("Using %s architecture", model_type)

    # Load processor from the path with tokenizer files
    logger.info("Loading processor from %s", processor_path)
    processor = AutoProcessor.from_pretrained(processor_path)

    # Load model from the path with model weights
    logger.info("Loading model weights from %s", model_load_path)
    if use_wav2vec2_base:
        model = Wav2Vec2ForCTC.from_pretrained(model_load_path)
    else:
        model = Wav2Vec2BertForCTC.from_pretrained(model_load_path)

    model.to(device)
    model.eval()

    logger.info("Model loaded on %s", device)
    return model, processor, use_wav2vec2_base


def get_audio_files(audio_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Get all audio files from directory.

    Args:
        audio_dir: Directory containing audio files
        extensions: List of audio file extensions to include

    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    audio_files = []
    audio_path = Path(audio_dir)

    for ext in extensions:
        audio_files.extend(audio_path.glob(f"*{ext}"))
        audio_files.extend(audio_path.glob(f"*{ext.upper()}"))

    audio_files = sorted([str(f) for f in audio_files])
    logger.info("Found %s audio files in %s", len(audio_files), audio_dir)

    return audio_files


def transcribe_audio(
    audio_path: str,
    model: Union[Wav2Vec2ForCTC, Wav2Vec2BertForCTC],
    processor: Union[Wav2Vec2Processor, Wav2Vec2BertProcessor],
    use_wav2vec2_base: bool,
    device: str = "cuda",
    return_confidence: bool = False,
    return_alternatives: bool = False,
    num_alternatives: int = 3,
    unk_token_replacement: Optional[str] = None,
) -> dict:
    """
    Transcribe a single audio file.

    Args:
        audio_path: Path to audio file
        model: Loaded ASR model
        processor: Loaded processor
        use_wav2vec2_base: Whether using traditional Wav2Vec2 architecture
        device: Device for inference
        return_confidence: Whether to return confidence scores
        return_alternatives: Whether to return alternative transcriptions
        num_alternatives: Number of alternative transcriptions to return
        unk_token_replacement: String to replace [UNK] tokens with (None to keep [UNK], "" to remove)

    Returns:
        Dictionary with transcription and optional metadata
    """


    # Load audio
    try:
        audio, _sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        logger.error("Error loading %s: %s", audio_path, e)
        return {
            "transcription": "",
            "error": str(e),
            "confidence": None,
            "alternatives": None,
        }

    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Move inputs to device - handle both input_values and input_features
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids)[0]

    # Handle UNK token replacement if specified
    if unk_token_replacement is not None:
        transcription = transcription.replace("[UNK]", unk_token_replacement)
        # Clean up any double spaces that might result from removal
        if unk_token_replacement == "":
            transcription = " ".join(transcription.split())

    result = {
        "transcription": transcription,
        "error": None,
    }

    # Calculate confidence if requested
    if return_confidence:
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get confidence for predicted tokens
        max_probs = torch.max(probs, dim=-1)[0]

        # Average confidence across sequence (excluding padding)
        confidence = max_probs.mean().item()
        result["confidence"] = confidence
    else:
        result["confidence"] = None

    # Get alternatives if requested
    if return_alternatives:
        # Get top-k predictions
        top_k_probs, top_k_ids = torch.topk(logits, k=num_alternatives, dim=-1)

        alternatives = []
        for k in range(1, num_alternatives):  # Skip first as it's the main transcription
            alt_ids = top_k_ids[:, :, k]
            alt_text = processor.batch_decode(alt_ids)[0]

            # Handle UNK token replacement in alternatives
            if unk_token_replacement is not None:
                alt_text = alt_text.replace("[UNK]", unk_token_replacement)
                if unk_token_replacement == "":
                    alt_text = " ".join(alt_text.split())

            # Calculate alternative confidence
            alt_probs = top_k_probs[:, :, k]
            alt_confidence = alt_probs.mean().item()

            alternatives.append(
                {
                    "text": alt_text,
                    "confidence": alt_confidence,
                }
            )

        result["alternatives"] = alternatives
    else:
        result["alternatives"] = None

    return result


def batch_transcribe(
    audio_files: List[str],
    model: Union[Wav2Vec2ForCTC, Wav2Vec2BertForCTC],
    processor: Union[Wav2Vec2Processor, Wav2Vec2BertProcessor],
    use_wav2vec2_base: bool,
    output_csv_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    return_confidence: bool = False,
    return_alternatives: bool = False,
    num_alternatives: int = 3,
    unk_token_replacement: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transcribe multiple audio files.

    Args:
        audio_files: List of audio file paths
        model: Loaded ASR model
        processor: Loaded processor
        use_wav2vec2_base: Whether using traditional Wav2Vec2 architecture
        output_csv_path: Path where the CSV will be saved (for computing relative paths)
        device: Device for inference
        batch_size: Batch size for processing (currently only supports 1)
        return_confidence: Whether to return confidence scores
        return_alternatives: Whether to return alternative transcriptions
        num_alternatives: Number of alternatives to return
        unk_token_replacement: String to replace [UNK] tokens with (None to keep [UNK], "" to remove)

    Returns:
        DataFrame with transcriptions and metadata
    """
    results = []

    # Get the directory where the CSV will be saved
    csv_dir = os.path.dirname(os.path.abspath(output_csv_path))
    if not csv_dir:
        csv_dir = os.getcwd()

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        result = transcribe_audio(
            audio_path,
            model,
            processor,
            use_wav2vec2_base,
            device,
            return_confidence,
            return_alternatives,
            num_alternatives,
            unk_token_replacement,
        )

        # Calculate relative path from CSV location to audio file
        abs_audio_path = os.path.abspath(audio_path)
        try:
            rel_path = os.path.relpath(abs_audio_path, csv_dir)
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            # Fall back to absolute path
            rel_path = abs_audio_path

        result["file_name"] = rel_path

        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ["file_name", "transcription"]
    if return_confidence:
        cols.append("confidence")
    if return_alternatives:
        cols.append("alternatives")
    if "error" in df.columns:
        cols.append("error")

    df = df[cols]

    return df


def inference_w2v2bert_asr_api(
    model_path: str,
    audio_files: List[str],
    output_csv: Optional[str] = None,
    device: Optional[str] = None,
    use_wav2vec2_base: Optional[bool] = None,
    return_confidence: bool = False,
    return_alternatives: bool = False,
    num_alternatives: int = 3,
    suppress_unk: bool = False,
    unk_replacement: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio files using W2V2-BERT ASR via Python API.

    This function provides programmatic access to ASR inference without
    requiring command-line arguments.

    Args:
        model_path: Path to trained model directory
        audio_files: List of audio file paths to transcribe
        output_csv: Optional path to save results CSV
        device: Device for inference (default: auto-detect CUDA/CPU)
        use_wav2vec2_base: Force architecture type (None for auto-detect)
        return_confidence: Whether to return confidence scores
        return_alternatives: Whether to return alternative transcriptions
        num_alternatives: Number of alternatives to return
        suppress_unk: Remove [UNK] tokens from output
        unk_replacement: String to replace [UNK] tokens with
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Dict with keys:
            - success: bool indicating if inference completed
            - transcriptions: List of dicts with file_name, transcription, etc.
            - results_df: pandas DataFrame with all results
            - files_processed: Number of files processed
            - errors_count: Number of files with errors
            - error_message: Error message if success is False
    """
    result = {
        "success": False,
        "transcriptions": [],
        "results_df": None,
        "files_processed": 0,
        "errors_count": 0,
        "error_message": None,
    }

    try:
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Handle UNK token replacement logic
        unk_token_replacement = None
        if suppress_unk:
            unk_token_replacement = ""
            logger.info("UNK tokens will be removed from output")
        elif unk_replacement is not None:
            unk_token_replacement = unk_replacement
            if unk_token_replacement == "":
                logger.info("UNK tokens will be removed from output")
            else:
                logger.info("UNK tokens will be replaced with: '%s'", unk_token_replacement)

        # Load model and processor with architecture detection
        logger.info("Loading model from %s", model_path)
        model, processor, detected_wav2vec2_base = load_model_and_processor(model_path, device, use_wav2vec2_base)

        if not audio_files:
            result["error_message"] = "No audio files provided"
            return result

        logger.info("Starting transcription of %s files...", len(audio_files))

        # Transcribe files with progress callback
        transcriptions = []
        csv_dir = os.path.dirname(os.path.abspath(output_csv)) if output_csv else os.getcwd()

        for idx, audio_path in enumerate(audio_files):
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(idx, len(audio_files))
                except Exception as e:
                    logger.warning("Progress callback failed: %s", e)
                    # Re-raise if it's a cancellation exception
                    if "cancel" in str(e).lower():
                        raise

            trans_result = transcribe_audio(
                audio_path,
                model,
                processor,
                detected_wav2vec2_base,
                device,
                return_confidence,
                return_alternatives,
                num_alternatives,
                unk_token_replacement,
            )

            # Calculate relative path from CSV location to audio file
            abs_audio_path = os.path.abspath(audio_path)
            try:
                rel_path = os.path.relpath(abs_audio_path, csv_dir)
            except ValueError:
                rel_path = abs_audio_path

            trans_result["file_name"] = rel_path
            transcriptions.append(trans_result)

        # Convert to DataFrame
        df = pd.DataFrame(transcriptions)

        # Reorder columns
        cols = ["file_name", "transcription"]
        if return_confidence:
            cols.append("confidence")
        if return_alternatives:
            cols.append("alternatives")
        if "error" in df.columns:
            cols.append("error")

        df = df[[c for c in cols if c in df.columns]]

        # Save CSV if path provided
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            df.to_csv(output_csv, index=False)
            logger.info("Results saved to %s", output_csv)

        # Count errors
        errors_count = 0
        if "error" in df.columns:
            errors_count = df["error"].notna().sum()

        result["success"] = True
        result["transcriptions"] = transcriptions
        result["results_df"] = df
        result["files_processed"] = len(transcriptions)
        result["errors_count"] = errors_count

        logger.info("Transcription complete! Processed %s files", len(transcriptions))
        if errors_count > 0:
            logger.warning("Encountered %s errors during transcription", errors_count)

    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["error_message"] = error_msg

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using W2V2-BERT ASR")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")

    # Optional arguments
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 supported)")
    parser.add_argument(
        "--audio_extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        help="Audio file extensions to process",
    )

    # Model architecture
    parser.add_argument(
        "--use_wav2vec2_base",
        action="store_true",
        default=None,
        help="Force use of traditional Wav2Vec2 architecture (auto-detected if not specified)",
    )

    # Output options
    parser.add_argument(
        "--return_confidence", action="store_true", default=False, help="Include confidence scores in output"
    )
    parser.add_argument(
        "--return_alternatives", action="store_true", default=False, help="Include alternative transcriptions in output"
    )
    parser.add_argument(
        "--num_alternatives", type=int, default=3, help="Number of alternative transcriptions to return"
    )

    # UNK token handling
    parser.add_argument(
        "--suppress_unk",
        action="store_true",
        default=False,
        help="Remove [UNK] tokens from output (equivalent to --unk_replacement '')",
    )
    parser.add_argument(
        "--unk_replacement",
        type=str,
        default=None,
        help="String to replace [UNK] tokens with (default: keep [UNK] as-is). Use empty string '' to remove.",
    )

    # Processing options
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output file if it exists")

    args = parser.parse_args()

    # Handle UNK token replacement logic
    unk_token_replacement = None
    if args.suppress_unk:
        unk_token_replacement = ""
        logger.info("UNK tokens will be removed from output")
    elif args.unk_replacement is not None:
        unk_token_replacement = args.unk_replacement
        if unk_token_replacement == "":
            logger.info("UNK tokens will be removed from output")
        else:
            logger.info("UNK tokens will be replaced with: '%s'", unk_token_replacement)

    # Check if output exists
    if os.path.exists(args.output_csv) and not args.overwrite:
        logger.error("Output file %s already exists. Use --overwrite to overwrite.", args.output_csv)
        return

    # Load model and processor with architecture detection
    model, processor, use_wav2vec2_base = load_model_and_processor(args.model_path, args.device, args.use_wav2vec2_base)

    # Get audio files
    audio_files = get_audio_files(args.audio_dir, args.audio_extensions)

    if not audio_files:
        logger.error("No audio files found in %s", args.audio_dir)
        return

    # Transcribe
    logger.info("Starting transcription of %s files...", len(audio_files))
    results_df = batch_transcribe(
        audio_files,
        model,
        processor,
        use_wav2vec2_base,
        args.output_csv,
        args.device,
        args.batch_size,
        args.return_confidence,
        args.return_alternatives,
        args.num_alternatives,
        unk_token_replacement,
    )

    # Save results
    logger.info("Saving results to %s", args.output_csv)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Save CSV
    results_df.to_csv(args.output_csv, index=False)

    # Print summary
    logger.info("Transcription complete!")
    logger.info("Processed %s files", len(results_df))

    if "error" in results_df.columns:
        errors = results_df["error"].notna().sum()
        if errors > 0:
            logger.warning("Encountered %s errors during transcription", errors)

    if args.return_confidence:
        avg_confidence = results_df["confidence"].mean()
        logger.info("Average confidence: %.4f", avg_confidence)

    logger.info("Results saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
