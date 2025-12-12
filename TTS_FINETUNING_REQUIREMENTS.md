# MMS-TTS Fine-tuning Project Requirements

## Project Overview
Fine-tune a Text-to-Speech (TTS) model for config language using the MMS-TTS architecture.

---

## Dataset Information

### Location
- **Dataset Path**: `/path/to/output/`
- **Format**: HuggingFace dataset format (local)
- **Duration**: 7.67 hours of audio
- **Language**: Language

### Current State
- Audio files are primarily in **webm** format (mixed formats)
- **No speaker IDs assigned yet** - requires speaker diarization
- Contains audio files with corresponding text transcriptions

---

## Model Configuration

### Base Model
- **Model**: `facebook/mms-tts-cmn` (Mandarin Chinese checkpoint)
- **Rationale**: Closest suitable base for language fine-tuning

### Target Audio Specifications
- **Sample Rate**: 16kHz
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Format**: WAV
- **Preprocessing**: Normalization only (no noise reduction)

---

## Training Configuration

### Hardware
- **Primary**: Vast.ai GPU with 12GB VRAM (target configuration)
- **Secondary**: Local GPU with 8GB VRAM (for quick tests only)
- **Note**: Optimize for 12GB GPU, don't prioritize 8GB compatibility

### Training Parameters
- **Epochs**: 200-300 (starting point, with validation monitoring)
- **Learning Rate**: 1e-4
- **Batch Size**: Auto-determine based on 12GB GPU
- **Gradient Accumulation**: Yes, 2-4 steps (to simulate larger batches of 32-64)
- **Mixed Precision**: fp16 (for memory efficiency and speed)
- **Validation Split**: 10% (90% training, 10% validation)
- **Checkpoint Frequency**: Every 25 epochs

### Output Configuration
- **Checkpoint Directory**: `/home/lansford/vast_ai_tts/checkpoints/`
- **Model Distribution**: Keep local only (no HuggingFace Hub upload)

---

## Pipeline Requirements

### 1. Audio Preprocessing
- Convert all audio formats (webm, mp3, wav, flac, etc.) to standardized format
- Target: 16kHz, 16-bit, mono WAV
- Apply normalization to consistent dB level
- Support batch processing of entire dataset

### 2. Speaker Diarization
- Perform speaker clustering/diarization on audio files
- Assign unique speaker IDs to each audio sample
- Method: Use pyannote.audio or resemblyzer for speaker embeddings
- Output: Speaker ID metadata for each audio file

### 3. Dataset Preparation
- Load HuggingFace dataset
- Integrate speaker IDs from diarization
- Split into train (90%) and validation (10%) sets
- Ensure proper format for MMS-TTS training

### 4. Model Training
- Fine-tune MMS-TTS model with specified parameters
- Monitor validation loss and audio quality
- Save checkpoints every 25 epochs
- Support resuming from checkpoints

---

## Technical Stack

### Core Libraries
- **TTS Framework**: Transformers (HuggingFace)
- **Audio Processing**: librosa, soundfile, pydub
- **Speaker Diarization**: pyannote.audio or resemblyzer
- **Dataset**: datasets (HuggingFace)
- **Training**: PyTorch, accelerate
- **Monitoring**: tensorboard or wandb (optional)

### System Requirements
- Python 3.8+
- CUDA-capable GPU (12GB VRAM recommended)
- FFmpeg (for audio format conversion)
- Sufficient disk space for checkpoints and processed audio

---

## Workflow Considerations

### Option A: Sequential Pipeline
1. Preprocess all audio → standardized WAV files
2. Run speaker diarization on processed files
3. Prepare dataset with speaker IDs
4. Train model

**Pros**: Clear separation of concerns, easier debugging
**Cons**: Multiple passes over data, more disk space needed

### Option B: Integrated Pipeline
1. Load audio → preprocess on-the-fly → extract speaker embeddings → cache
2. Prepare dataset with cached embeddings and speaker IDs
3. Train model

**Pros**: Single pass, less disk space
**Cons**: More complex, harder to debug individual steps

### Option C: Hybrid Approach (Recommended)
1. Preprocess audio during dataset copy/preparation (integrate with existing script)
2. Run speaker diarization as separate step on preprocessed audio
3. Merge speaker IDs into dataset metadata
4. Train model

**Pros**: Leverages existing workflow, modular, debuggable
**Cons**: Requires coordination between scripts

---

## Questions to Resolve

1. **Existing Script Integration**: How does your current data copying script work? Can we add preprocessing hooks?

2. **Speaker Diarization Timing**: 
   - Should we run diarization before or after audio preprocessing?
   - Answer: After preprocessing (consistent format makes diarization more reliable)

3. **Dataset Structure**: What fields does your HuggingFace dataset currently have?
   - Need to know: audio path, transcription text, any existing metadata

4. **Speaker ID Storage**: Where should speaker IDs be stored?
   - Options: 
     - Add column to HuggingFace dataset
     - Separate JSON/CSV mapping file
     - Embed in filename

5. **Checkpoint Management**: Do you want to keep all checkpoints or only best N?

---

## Next Steps

1. ✅ Document requirements (this file)
2. ⏳ Examine existing data copying script
3. ⏳ Design integrated preprocessing workflow
4. ⏳ Implement speaker diarization
5. ⏳ Create training script
6. ⏳ Test end-to-end pipeline

---

## Notes

- The dataset is relatively small (7.67 hours), so careful validation monitoring is crucial to prevent overfitting
- Mixed precision training (fp16) should provide ~2-3x speedup on modern GPUs
- Gradient accumulation allows effective batch size of 32-64 even with memory constraints

---

**Last Updated**: 2025-11-28
**Project Location**: `/home/lansford/work2/Mission_Mutual/audio_text_tests`