#!/usr/bin/env python3
"""
W2V2-BERT ASR Training Script

Fine-tune Wav2Vec2-BERT models for Automatic Speech Recognition.
Based on: https://huggingface.co/blog/fine-tune-w2v2-bert

This module provides both a command-line interface and a Python API.

Python API Usage:
    from train_w2v2bert_asr import train_w2v2bert_asr_api

    result = train_w2v2bert_asr_api(
        csv_path='/path/to/metadata.csv',
        output_dir='/path/to/output',
        model_name='facebook/w2v-bert-2.0',
        num_train_epochs=5,
        heartbeat_callback=lambda epoch: None  # Optional
    )

Command-line Usage:
    python train_w2v2bert_asr.py \
        --csv_path data/metadata.csv \
        --audio_column file_name \
        --text_column transcription \
        --output_dir outputs/w2v2bert_asr \
        --model_name facebook/w2v-bert-2.0
"""

import argparse
import json
import logging
import os
import re
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict
import evaluate
from transformers import (
    AutoConfig,
    Wav2Vec2BertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)
import librosa

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Any
    padding: Union[bool, str] = True
    use_wav2vec2_base: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        # Traditional Wav2Vec2 uses "input_values", Wav2Vec2-BERT uses "input_features"
        input_key = "input_values" if self.use_wav2vec2_base else "input_features"
        input_features = [{input_key: feature[input_key]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


class TextNormalizer:
    """Handle text normalization with configurable options."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace

    def normalize(self, text: str) -> str:
        """Normalize text according to configured options."""
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[,\.!?;:\-\"\'\(\)\[\]\{\}]', '', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text


def detect_language_from_text(texts: List[str]) -> str:
    """Auto-detect language from text samples."""
    sample_text = " ".join(texts[:100])

    if re.search(r'[\u4e00-\u9fff]', sample_text):
        lang = "zh"
    elif re.search(r'[\u0400-\u04ff]', sample_text):
        lang = "ru"
    elif re.search(r'[\u0600-\u06ff]', sample_text):
        lang = "ar"
    elif re.search(r'[\u0e00-\u0e7f]', sample_text):
        lang = "th"
    else:
        lang = "en"

    logger.info("Auto-detected language: %s", lang)
    return lang


def load_csv_dataset(
    csv_path: str,
    audio_column: str,
    text_column: str,
    audio_base_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load CSV dataset and validate columns."""
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Try to find audio column
    if audio_column not in df.columns:
        alternatives = ["file_name", "filename", "audio", "path", "audio_path"]
        for alt in alternatives:
            if alt in df.columns:
                logger.warning("Audio column '%s' not found, using '%s' instead", audio_column, alt)
                audio_column = alt
                break
        else:
            raise ValueError(f"Audio column '{audio_column}' not found in CSV. Available columns: {df.columns.tolist()}")

    # Try to find text column
    if text_column not in df.columns:
        alternatives = ["transcription", "text", "transcript", "sentence"]
        for alt in alternatives:
            if alt in df.columns:
                logger.warning("Text column '%s' not found, using '%s' instead", text_column, alt)
                text_column = alt
                break
        else:
            raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")

    # Rename columns to standard names
    df = df.rename(columns={audio_column: "audio", text_column: "text"})

    # Resolve audio paths relative to CSV directory if not absolute
    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    def resolve_audio_path(audio_path):
        # If audio_base_path is provided, use it
        if audio_base_path:
            return os.path.join(audio_base_path, audio_path)
        # If path is already absolute, use it as-is
        if os.path.isabs(audio_path):
            return audio_path
        # Otherwise, resolve relative to CSV directory
        return os.path.join(csv_dir, audio_path)

    df["audio"] = df["audio"].apply(resolve_audio_path)

    # Validate audio files exist
    missing_files = []
    for _idx, row in df.iterrows():
        if not os.path.exists(row["audio"]):
            missing_files.append(row["audio"])

    if missing_files:
        logger.warning("Found %s missing audio files (showing first 5):", len(missing_files))
        for f in missing_files[:5]:
            logger.warning("  - %s", f)

    # Remove rows with missing files
    df = df[df["audio"].apply(os.path.exists)]
    logger.info("Loaded %s samples with valid audio files", len(df))

    return df


def filter_by_duration(
    df: pd.DataFrame,
    max_duration_seconds: Optional[float] = None,
    max_duration_std: Optional[float] = None,
) -> pd.DataFrame:
    """Filter dataset by audio duration."""
    if max_duration_seconds is None and max_duration_std is None:
        return df

    logger.info("Calculating audio durations...")

    durations = []
    for audio_path in df["audio"]:
        try:
            duration = librosa.get_duration(path=audio_path)
            durations.append(duration)
        except Exception as e:
            logger.warning("Could not get duration for %s: %s", audio_path, e)
            durations.append(0)

    df["duration"] = durations
    original_len = len(df)

    if max_duration_std is not None:
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        max_duration = mean_duration + (max_duration_std * std_duration)
        logger.info("Mean duration: %.2fs, Std: %.2fs", mean_duration, std_duration)
        logger.info("Filtering to max %s std: %.2fs", max_duration_std, max_duration)
        df = df[df["duration"] <= max_duration]
    elif max_duration_seconds is not None:
        logger.info("Filtering to max duration: %ss", max_duration_seconds)
        df = df[df["duration"] <= max_duration_seconds]

    logger.info("Filtered %s samples, %s remaining", original_len - len(df), len(df))
    return df.drop(columns=["duration"])


def create_vocab_from_data(texts: List[str]) -> Dict[str, int]:
    """Create vocabulary from text data."""
    vocab_set = set()
    for text in texts:
        vocab_set.update(text)

    vocab_list = sorted(list(vocab_set))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # Add special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    logger.info("Created vocabulary with %s tokens", len(vocab_dict))
    logger.info("Sample characters: %s", list(vocab_dict.keys())[:20])

    return vocab_dict


def prepare_dataset(
    df: pd.DataFrame,
    processor: Union[Wav2Vec2BertProcessor, Wav2Vec2Processor],
    text_normalizer: TextNormalizer,
    use_wav2vec2_base: bool,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Prepare dataset for training."""
    # Normalize text
    logger.info("Normalizing text...")
    df["text"] = df["text"].apply(text_normalizer.normalize)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[["audio", "text"]])

    # Cast audio column to Audio feature
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split dataset
    logger.info("Splitting dataset: train=%s, val=%s, test=%s", train_split, val_split, test_split)

    train_test = dataset.train_test_split(test_size=(val_split + test_split), seed=seed)

    if test_split > 0:
        val_test_split = val_split / (val_split + test_split)
        val_test = train_test["test"].train_test_split(test_size=(1 - val_test_split), seed=seed)

        dataset_dict = DatasetDict({
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
    else:
        dataset_dict = DatasetDict({
            "train": train_test["train"],
            "validation": train_test["test"],
        })

    logger.info("Dataset splits: %s", dataset_dict)

    # Prepare features
    def prepare_dataset_features(batch):
        audio = batch["audio"]
        processed = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        )

        # Traditional Wav2Vec2 uses "input_values", Wav2Vec2-BERT uses "input_features"
        if use_wav2vec2_base:
            batch["input_values"] = processed.input_values[0]
        else:
            batch["input_features"] = processed.input_features[0]

        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    logger.info("Preparing dataset features...")
    dataset_dict = dataset_dict.map(
        prepare_dataset_features,
        remove_columns=dataset_dict["train"].column_names,
        num_proc=4,
    )

    return dataset_dict


def compute_metrics(pred, wer_metric, cer_metric, processor):
    """Compute WER and CER metrics."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


class BestCheckpointCallback(TrainerCallback):
    """Callback to save best checkpoint separately."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_wer = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_wer" in metrics:
            wer = metrics["eval_wer"]
            if wer < self.best_wer:
                self.best_wer = wer
                logger.info("New best WER: %.4f", wer)


class HeartbeatCallback(TrainerCallback):
    """Callback to invoke heartbeat during training."""

    def __init__(self, heartbeat_callback: Optional[Callable[[int], None]] = None):
        self.heartbeat_callback = heartbeat_callback
        self.last_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        current_epoch = int(state.epoch) if state.epoch else 0
        if current_epoch != self.last_epoch:
            self.last_epoch = current_epoch
            if self.heartbeat_callback:
                try:
                    self.heartbeat_callback(current_epoch)
                except Exception as e:
                    logger.warning("Heartbeat callback failed: %s", e)
                    # Re-raise if it's a cancellation exception
                    if "cancel" in str(e).lower():
                        raise


def train_w2v2bert_asr_api(
    csv_path: str,
    output_dir: str,
    model_name: str = "facebook/w2v-bert-2.0",
    audio_column: str = "file_name",
    text_column: str = "transcription",
    audio_base_path: Optional[str] = None,
    use_wav2vec2_base: Optional[bool] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_duration_seconds: Optional[float] = None,
    max_duration_std: Optional[float] = None,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    learning_rate: float = 3e-4,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 5,
    warmup_steps: int = 500,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    save_total_limit: int = 3,
    fp16: bool = True,
    bf16: bool = False,
    use_8bit_optimizer: bool = False,
    seed: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    heartbeat_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Train W2V2-BERT ASR model via Python API.

    This function provides programmatic access to ASR training without
    requiring command-line arguments or config files.

    Args:
        csv_path: Path to CSV file with audio paths and transcriptions
        output_dir: Directory to save model and outputs
        model_name: Pretrained model name (default: facebook/w2v-bert-2.0)
        audio_column: Column name for audio file paths
        text_column: Column name for transcriptions
        audio_base_path: Optional base path for audio files
        use_wav2vec2_base: Use traditional Wav2Vec2 (auto-detected if None)
        train_split: Training data split ratio
        val_split: Validation data split ratio
        test_split: Test data split ratio
        max_duration_seconds: Maximum audio duration in seconds
        max_duration_std: Maximum duration in standard deviations
        lowercase: Whether to lowercase text
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        learning_rate: Learning rate for training
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        use_8bit_optimizer: Use 8-bit Adam optimizer
        seed: Random seed
        resume_from_checkpoint: Path to checkpoint to resume from
        heartbeat_callback: Optional callback invoked each epoch with epoch number.
                           Can raise exception to cancel training.

    Returns:
        Dict with keys:
            - success: bool indicating if training completed
            - epochs_completed: Number of epochs completed
            - final_model_path: Path to final saved model
            - test_results: Test set evaluation results (if test_split > 0)
            - train_metrics: Training metrics from HuggingFace Trainer including:
                - train_loss: Final training loss
                - train_runtime: Total training time in seconds
                - train_samples_per_second: Training throughput
                - train_steps_per_second: Training steps throughput
                - global_step: Total training steps completed
            - error_message: Error message if success is False
    """
    result = {
        "success": False,
        "epochs_completed": 0,
        "final_model_path": None,
        "test_results": None,
        "error_message": None,
    }

    try:
        # Auto-detect model architecture if not explicitly specified
        if use_wav2vec2_base is None:
            use_wav2vec2_base = "bert" not in model_name.lower()
            if use_wav2vec2_base:
                logger.info("Auto-detected traditional Wav2Vec2 model from name: %s", model_name)

        model_type = "Wav2Vec2" if use_wav2vec2_base else "Wav2Vec2-BERT"
        logger.info("Using %s architecture with model: %s", model_type, model_name)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        logger.info("Loading dataset...")
        df = load_csv_dataset(csv_path, audio_column, text_column, audio_base_path)

        # Filter by duration
        if max_duration_seconds or max_duration_std:
            df = filter_by_duration(df, max_duration_seconds, max_duration_std)

        # Create text normalizer
        text_normalizer = TextNormalizer(
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            remove_numbers=remove_numbers,
        )

        # Normalize text for vocab
        normalized_texts = df["text"].apply(text_normalizer.normalize).tolist()

        # Debug: Show some samples
        logger.info("Total texts: %s", len(normalized_texts))
        logger.info("Sample original texts: %s", df['text'].head(3).tolist())
        logger.info("Sample normalized texts: %s", normalized_texts[:3])

        # Create vocabulary
        vocab_dict = create_vocab_from_data(normalized_texts)

        # Save vocabulary
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        logger.info("Saved vocabulary to %s", vocab_path)

        # Create tokenizer with custom vocabulary
        logger.info("Creating tokenizer with custom vocabulary")
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )

        # Load feature extractor
        logger.info("Loading feature extractor from: %s", model_name)
        if use_wav2vec2_base:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        else:
            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)

        # Create processor
        if use_wav2vec2_base:
            processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer
            )
        else:
            processor = Wav2Vec2BertProcessor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer
            )

        # Save processor
        processor.save_pretrained(output_dir)

        # Load model
        config = AutoConfig.from_pretrained(model_name)
        config.vocab_size = len(vocab_dict)

        if use_wav2vec2_base:
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            model = Wav2Vec2BertForCTC.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )

        # Freeze feature encoder
        if use_wav2vec2_base:
            for param in model.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Froze Wav2Vec2 feature extractor parameters")
        else:
            for param in model.wav2vec2_bert.feature_projection.parameters():
                param.requires_grad = False
            logger.info("Froze Wav2Vec2-BERT feature projection parameters")

        # Prepare dataset
        dataset_dict = prepare_dataset(
            df, processor, text_normalizer, use_wav2vec2_base,
            train_split, val_split, test_split, seed
        )

        # Data collator
        data_collator = DataCollatorCTCWithPadding(
            processor=processor,
            use_wav2vec2_base=use_wav2vec2_base
        )

        # Load metrics
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            fp16=fp16 and torch.cuda.is_available(),
            bf16=bf16,
            optim="adamw_bnb_8bit" if use_8bit_optimizer else "adamw_torch",
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(output_dir, "logs"),
            seed=seed,
        )

        # Log optimizer choice
        if use_8bit_optimizer:
            logger.info("Using 8-bit Adam optimizer (bitsandbytes) for reduced memory usage")
        else:
            logger.info("Using standard AdamW optimizer")

        # Setup callbacks
        callbacks = [BestCheckpointCallback(output_dir)]
        if heartbeat_callback:
            callbacks.append(HeartbeatCallback(heartbeat_callback))

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            tokenizer=processor.feature_extractor, # pylint: disable=no-member
            compute_metrics=lambda pred: compute_metrics(pred, wer_metric, cer_metric, processor),
            callbacks=callbacks,
        )

        # Resume from checkpoint
        checkpoint = resume_from_checkpoint
        if checkpoint is None and os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint = os.path.join(output_dir, sorted(checkpoints)[-1])
                logger.info("Auto-resuming from %s", checkpoint)

        # Train
        logger.info("Starting training...")
        train_output = trainer.train(resume_from_checkpoint=checkpoint)

        # Save final model
        final_model_path = os.path.join(output_dir, "final_model")
        logger.info("Saving final model to %s...", final_model_path)
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)

        result["epochs_completed"] = num_train_epochs
        result["final_model_path"] = final_model_path

        # Include training metrics from HuggingFace Trainer
        result["train_metrics"] = {
            "train_loss": train_output.training_loss,
            "train_runtime": train_output.metrics.get("train_runtime"),
            "train_samples_per_second": train_output.metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
            "global_step": train_output.global_step,
        }

        # Evaluate on test set
        if "test" in dataset_dict:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(dataset_dict["test"])
            logger.info("Test results: %s", test_results)

            test_results_path = os.path.join(output_dir, "test_results.json")
            with open(test_results_path, "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2)

            result["test_results"] = test_results

        logger.info("Training complete!")
        result["success"] = True

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["error_message"] = error_msg

    return result


def main():
    parser = argparse.ArgumentParser(description="Train W2V2-BERT ASR model")

    # Data arguments
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--audio_column", type=str, default="file_name", help="Audio column name")
    parser.add_argument("--text_column", type=str, default="transcription", help="Text column name")
    parser.add_argument("--audio_base_path", type=str, default=None, help="Base path for audio files")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/w2v-bert-2.0", help="Pretrained model")
    parser.add_argument("--use_wav2vec2_base", action="store_true", default=False,
                       help="Use traditional Wav2Vec2 architecture (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Data preprocessing
    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--max_duration_seconds", type=float, default=None, help="Max audio duration")
    parser.add_argument("--max_duration_std", type=float, default=None, help="Max duration in std devs")

    # Text normalization
    parser.add_argument("--lowercase", action="store_true", default=True)
    parser.add_argument("--no_lowercase", action="store_false", dest="lowercase")
    parser.add_argument("--remove_punctuation", action="store_true", default=True)
    parser.add_argument("--no_remove_punctuation", action="store_false", dest="remove_punctuation")
    parser.add_argument("--remove_numbers", action="store_true", default=False)

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--bf16", action="store_true", default=False)

    # Memory optimization
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=False,
                       help="Use 8-bit Adam optimizer from bitsandbytes to reduce memory usage")

    # Hub
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Auto-detect model architecture if not explicitly specified
    if not args.use_wav2vec2_base and "bert" not in args.model_name.lower():
        logger.info("Auto-detected traditional Wav2Vec2 model from name: %s", args.model_name)
        args.use_wav2vec2_base = True

    model_type = "Wav2Vec2" if args.use_wav2vec2_base else "Wav2Vec2-BERT"
    logger.info("Using %s architecture with model: %s", model_type, args.model_name)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    logger.info("Loading dataset...")
    df = load_csv_dataset(args.csv_path, args.audio_column, args.text_column, args.audio_base_path)

    # Filter by duration
    if args.max_duration_seconds or args.max_duration_std:
        df = filter_by_duration(df, args.max_duration_seconds, args.max_duration_std)

    # Create text normalizer
    text_normalizer = TextNormalizer(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_numbers=args.remove_numbers,
    )

    # Normalize text for vocab
    normalized_texts = df["text"].apply(text_normalizer.normalize).tolist()

    # Debug: Show some samples
    logger.info("Total texts: %s", len(normalized_texts))
    logger.info("Sample original texts: %s", df['text'].head(3).tolist())
    logger.info("Sample normalized texts: %s", normalized_texts[:3])
    logger.info("Sample text lengths: %s", [len(t) for t in normalized_texts[:10]])

    # Create vocabulary
    vocab_dict = create_vocab_from_data(normalized_texts)

    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    logger.info("Saved vocabulary to %s", vocab_path)

    # Create tokenizer with custom vocabulary
    logger.info("Creating tokenizer with custom vocabulary")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    # Load feature extractor
    logger.info("Loading feature extractor from: %s", args.model_name)
    if args.use_wav2vec2_base:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    else:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_name)

    # Create processor
    if args.use_wav2vec2_base:
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    else:
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )

    # Save processor
    processor.save_pretrained(args.output_dir)

    # Load model
    config = AutoConfig.from_pretrained(args.model_name)
    config.vocab_size = len(vocab_dict)

    if args.use_wav2vec2_base:
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = Wav2Vec2BertForCTC.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

    # Freeze feature encoder
    if args.use_wav2vec2_base:
        # Traditional Wav2Vec2 uses feature_extractor
        for param in model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("Froze Wav2Vec2 feature extractor parameters")
    else:
        # Wav2Vec2-BERT uses feature_projection
        for param in model.wav2vec2_bert.feature_projection.parameters():
            param.requires_grad = False
        logger.info("Froze Wav2Vec2-BERT feature projection parameters")

    # Prepare dataset
    dataset_dict = prepare_dataset(
        df, processor, text_normalizer, args.use_wav2vec2_base,
        args.train_split, args.val_split, args.test_split, args.seed
    )

    # Data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        use_wav2vec2_base=args.use_wav2vec2_base
    )

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16,
        optim="adamw_bnb_8bit" if args.use_8bit_optimizer else "adamw_torch",
        save_total_limit=args.save_total_limit,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        seed=args.seed,
    )

    # Log optimizer choice
    if args.use_8bit_optimizer:
        logger.info("Using 8-bit Adam optimizer (bitsandbytes) for reduced memory usage")
    else:
        logger.info("Using standard AdamW optimizer")

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor, # pylint: disable=no-member
        compute_metrics=lambda pred: compute_metrics(pred, wer_metric, cer_metric, processor),
        callbacks=[BestCheckpointCallback(args.output_dir)],
    )

    # Resume from checkpoint
    checkpoint = args.resume_from_checkpoint
    if checkpoint is None and os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint = os.path.join(args.output_dir, sorted(checkpoints)[-1])
            logger.info("Auto-resuming from %s", checkpoint)

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))

    # Evaluate on test set
    if "test" in dataset_dict:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(dataset_dict["test"])
        logger.info("Test results: %s", test_results)

        with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()