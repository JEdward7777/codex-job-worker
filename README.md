# Audio Text Tests - GitLab to HuggingFace Dataset Converter

This project downloads audio files and transcriptions from a GitLab repository containing Bible recordings and prepares them in HuggingFace AudioFolder format.

## Features

- ✅ Authenticates with GitLab using personal access token
- ✅ Explores repository structure and finds CODEX files (transcription data)
- ✅ Maps audio files to transcriptions using audio IDs
- ✅ Downloads Git LFS files using HTTP Basic authentication
- ✅ Downloads only verses with completed transcriptions
- ✅ Creates HuggingFace-compatible dataset with CSV metadata
- ✅ Supports limiting records for testing
- ✅ Properly handles CSV quoting for transcriptions
- ✅ **Fully working end-to-end pipeline**

## Setup

1. Install dependencies using uv:
```bash
uv sync
```

2. Configure your GitLab credentials in `config.yaml`:
   - Add your GitLab access token (see instructions below)
   - Verify the project_id is correct
   - Adjust `max_records` for testing (set to `0` or `null` for all records)

## How to Create a GitLab Access Token

1. Go to your GitLab instance: https://git.genesisrnd.com
2. Click on your profile picture (top right) → **Edit Profile**
3. In the left sidebar, click **Access Tokens**
4. Click **Add new token**
5. Fill in the details:
   - **Token name**: Something descriptive like "HuggingFace Dataset Converter"
   - **Expiration date**: Set as needed
   - **Select scopes**: Check `read_api` and `read_repository`
6. Click **Create personal access token**
7. **Important**: Copy the token immediately (you won't be able to see it again)
8. Paste the token into `config.yaml` under `gitlab.access_token`

## Usage

The script uses Google Fire for a clean CLI interface. All commands support the `--config_path` parameter to specify a custom configuration file (defaults to `config.yaml`).

### Getting Help

```bash
# Show all available commands
uv run python gitlab_to_hf_dataset.py

# Or use --help
uv run python gitlab_to_hf_dataset.py --help

# Get help for a specific command
uv run python gitlab_to_hf_dataset.py process --help
uv run python gitlab_to_hf_dataset.py examine --help
uv run python gitlab_to_hf_dataset.py debug --help
uv run python gitlab_to_hf_dataset.py explore --help
```

### Basic Command Structure

```bash
# Using default config.yaml
uv run python gitlab_to_hf_dataset.py <command>

# Using custom config file
uv run python gitlab_to_hf_dataset.py <command> --config_path=my_config.yaml
```

### Available Commands

#### 1. Explore Repository Structure
```bash
uv run python gitlab_to_hf_dataset.py explore
```

Shows:
- Total files in the repository
- Number of CODEX files (transcription files)
- Number of audio files
- Sample file paths

#### 2. Examine CODEX File Schema
```bash
# Examine first CODEX file found
uv run python gitlab_to_hf_dataset.py examine

# Examine specific file
uv run python gitlab_to_hf_dataset.py examine --json_path="files/target/GEN.codex"
```

Downloads and displays the structure of a CODEX file to understand the data format.

#### 3. Debug Mode (Detailed Analysis)
```bash
uv run python gitlab_to_hf_dataset.py debug
```

Provides detailed information about:
- Audio ID mapping
- Transcription availability
- File matching status

#### 4. Process and Download Dataset
```bash
uv run python gitlab_to_hf_dataset.py process
```

This will:
1. Scan all 66 CODEX files (Bible books)
2. Build audio file map (1,361 audio files)
3. Extract audio-transcription pairs (only verses with completed transcriptions)
4. Download audio files via Git LFS (using HTTP Basic authentication)
5. Create `metadata.csv` in HuggingFace format

**Note**: The script respects the `max_records` setting in `config.yaml`. Set it to a small number (e.g., 10) for testing, then set to `0` or `null` for full processing.

### Using Custom Config Files

You can specify a different configuration file for any command:

```bash
# Process with production config
uv run python gitlab_to_hf_dataset.py process --config_path=config.prod.yaml

# Explore with test config
uv run python gitlab_to_hf_dataset.py explore --config_path=config.test.yaml

# Debug with custom config
uv run python gitlab_to_hf_dataset.py debug --config_path=my_config.yaml
```

This is useful for:
- Testing with different GitLab repositories
- Using different output directories
- Switching between development and production settings
- Managing multiple projects

**Example output:**
```
Total audio-transcription pairs found: 1
Downloading audio files and creating CSV...
[1/1] Downloading audio-1759292562790-fh3hv8f4g.webm...
Dataset created successfully!
```

## Output Format

The script creates a HuggingFace AudioFolder dataset:

```
huggingface_dataset/
├── audio/
│   ├── audio-1759292562790-fh3hv8f4g.webm
│   ├── audio-1759293405884-32rj85az8.webm
│   └── ...
└── metadata.csv
```

**metadata.csv** format:
```csv
"file_name","transcription"
"audio-1759292562790-fh3hv8f4g.webm","de khai doo da hui muc diam m sia doi ce can"
```

- All fields are properly quoted
- Quotes within transcriptions are escaped (double-quoted)
- Compatible with HuggingFace `datasets` library

## Configuration

Edit `config.yaml` to customize:

```yaml
gitlab:
  server_url: "https://git.genesisrnd.com"
  access_token: "your-token-here"
  project_id: "eten/lingao/lingao-fxrmlox2kyrrr3fwupgsq"

dataset:
  output_dir: "huggingface_dataset"
  audio_dir: "audio"
  csv_filename: "metadata.csv"
  max_records: 10  # Set to 0 or null for all records
```

## Loading the Dataset in HuggingFace

Once created, you can load the dataset using:

```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="huggingface_dataset")
```

## Project Structure

- `gitlab_to_hf_dataset.py` - Main script with Google Fire CLI
- `config.yaml` - Default configuration (not committed to git - all `.yaml` files are ignored)
- `pyproject.toml` - UV project configuration
- `.gitignore` - Excludes all YAML config files and output files
- `README.md` - This file

## Notes

- The script only downloads verses that have both audio files and transcriptions
- Audio files are stored in Git LFS (Large File Storage)
- CODEX files contain metadata about Bible verses, audio attachments, and transcriptions
- The script handles multiple audio versions per verse and selects the current one