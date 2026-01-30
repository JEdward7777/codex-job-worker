#!/usr/bin/env python3
"""
Script to download audio files and transcriptions from GitLab and prepare
a HuggingFace AudioFolder dataset.

Also provides upload functionality for files to GitLab with Git LFS support.
"""

import shutil
import re
import os
import csv
import json
from typing import Dict, List, Optional, Callable, ContextManager, BinaryIO
import hashlib
from pathlib import Path
import base64
from urllib.parse import quote
from io import BytesIO
from contextlib import contextmanager
import yaml
import requests
import fire

from uroman import Uroman
from gitlab_jobs import DEFAULT_GITLAB_URL


def make_file_opener(local_path: str) -> Callable[[], ContextManager[BinaryIO]]:
    """Create a context manager function that opens a file from disk.

    Args:
        local_path: Path to the local file

    Returns:
        A callable that returns a context manager yielding the open file
    """
    @contextmanager
    def opener():
        with open(local_path, 'rb') as f:
            yield f
    return opener


def make_content_opener(content: bytes) -> Callable[[], ContextManager[BinaryIO]]:
    """Create a context manager function that yields a BytesIO for inline content.

    Args:
        content: The bytes content to wrap

    Returns:
        A callable that returns a context manager yielding a BytesIO
    """
    @contextmanager
    def opener():
        yield BytesIO(content)
    return opener



class GitLabDatasetDownloader:
    """Downloads audio files and transcriptions from GitLab.

    Also provides upload functionality for files to GitLab with Git LFS support.
    """

    def __init__(
        self,
        config_path: Optional[str] = "config.yaml",
        config_overrides: dict = None,
        # Direct parameters (override config file if provided)
        gitlab_url: Optional[str] = None,
        access_token: Optional[str] = None,
        project_id: Optional[str] = None,
        project_path: Optional[str] = None,
    ):
        """Initialize with configuration from YAML file and/or direct parameters.

        Args:
            config_path: Path to configuration YAML file. Set to None to skip config file.
            config_overrides: Dictionary of config overrides using dot notation keys
                            (e.g., {'dataset.output_dir': '/path'})
            gitlab_url: GitLab server URL (overrides config if provided)
            access_token: GitLab access token (overrides config if provided)
            project_id: Numeric project ID (overrides config if provided)
            project_path: Project path like "namespace/project" (overrides config if provided)
        """
        # Initialize config - either from file or empty
        if config_path and os.path.exists(config_path):
            print(f"Reading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}
            if config_path:
                print(f"Config file not found: {config_path}, using direct parameters only")

        # Apply config overrides using dot notation
        if config_overrides:
            print(f"Applying {len(config_overrides)} config override(s)...")
            for key, value in config_overrides.items():
                self._set_nested_config(key, value)
                print(f"  Override: {key} = {value}")

        # Source mode: gitlab or local
        self.source_mode = self.config.get('source_mode', 'gitlab')
        self.local_repo_path = Path(self.config.get('local_repo_path', '')) if self.source_mode == 'local' else None

        # GitLab configuration - direct parameters override config file
        gitlab_config = self.config.get('gitlab', {})

        # Server URL: direct param > config > default
        self.server_url = (gitlab_url or gitlab_config.get('server_url', DEFAULT_GITLAB_URL)).rstrip('/')

        # Access token: direct param > config
        self.access_token = access_token or gitlab_config.get('access_token')

        # Project identification - keep numeric ID and path separate
        # project_id_number: numeric ID (e.g., "454") - used for API calls
        # project_path: path with namespace (e.g., "namespace/project") - used for LFS
        # project_id_for_api: URL-encoded version for API calls (works with either)
        self.project_id_number = None
        self.project_path = None

        if project_id:
            # User provided numeric ID
            self.project_id_number = str(project_id)
            self.project_id_for_api = quote(str(project_id), safe='')
        elif project_path:
            # User provided path
            self.project_path = project_path
            self.project_id_for_api = quote(project_path, safe='')
        elif gitlab_config.get('project_id'):
            # From config - could be either
            config_project = gitlab_config['project_id']
            if str(config_project).isdigit():
                self.project_id_number = str(config_project)
            else:
                self.project_path = config_project
            self.project_id_for_api = quote(str(config_project), safe='')
        else:
            self.project_id_for_api = None

        # Set up headers for API calls
        if self.access_token:
            self.headers = {
                'PRIVATE-TOKEN': self.access_token
            }
        else:
            self.headers = {}

        # Dataset configuration (optional - only needed for download operations)
        dataset_config = self.config.get('dataset', {})
        if dataset_config.get('output_dir'):
            self.output_dir = Path(dataset_config['output_dir'])
            self.audio_dir = self.output_dir / dataset_config.get('audio_dir', 'audio')
            self.csv_filename = dataset_config.get('csv_filename', 'metadata.csv')
            self.max_records = dataset_config.get('max_records', 0)
            self.text_source = dataset_config.get('text_source', 'transcription')
            self.edit_history_selection = dataset_config.get('edit_history_selection', 'initial_import')

            # Create output directories (including parent directories)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        else:
            # No dataset config - upload-only mode
            self.output_dir = None
            self.audio_dir = None
            self.csv_filename = None
            self.max_records = 0
            self.text_source = 'transcription'
            self.edit_history_selection = 'initial_import'

        # Uroman configuration
        uroman_config = self.config.get('uroman', {})
        self.uroman_enabled = uroman_config.get('enabled', False)
        self.uroman_language = uroman_config.get('language', None)

        # Validate uroman availability if enabled
        if self.uroman_enabled:
            raise ImportError(
                "uroman is enabled in config but not installed. "
                "Install it with: uv add uroman"
            )

        # Initialize uroman instance if enabled
        self.uroman_instance = Uroman() if self.uroman_enabled else None

    def _set_nested_config(self, dot_path: str, value):
        """Set nested config value using dot notation.

        Args:
            dot_path: Dot-separated path to config value (e.g., 'dataset.output_dir')
            value: Value to set
        """
        keys = dot_path.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the final value
        config[keys[-1]] = value

    def list_repository_tree_local(self, path: str = "", recursive: bool = True) -> List[Dict]:
        """List files in the local repository."""

        all_items = []
        search_path = self.local_repo_path / path if path else self.local_repo_path

        if recursive:
            # Recursively walk the directory
            for root, _dirs, files in os.walk(search_path):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.local_repo_path)
                    all_items.append({
                        'name': file,
                        'path': str(rel_path),
                        'type': 'blob'
                    })
        else:
            # List only top-level items
            if search_path.exists():
                for item in search_path.iterdir():
                    rel_path = item.relative_to(self.local_repo_path)
                    all_items.append({
                        'name': item.name,
                        'path': str(rel_path),
                        'type': 'tree' if item.is_dir() else 'blob'
                    })

        return all_items

    def list_repository_tree_gitlab(self, path: str = "", recursive: bool = True) -> List[Dict]:
        """List files in the GitLab repository."""
        url = f"{self.server_url}/api/v4/projects/{self.project_id_for_api}/repository/tree"
        params = {
            'path': path,
            'recursive': recursive,
            'per_page': 100
        }

        all_items = []
        page = 1

        while True:
            params['page'] = page
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            items = response.json()
            if not items:
                break

            all_items.extend(items)

            # Check if there are more pages
            if 'x-next-page' not in response.headers or not response.headers['x-next-page']:
                break

            page += 1

        return all_items

    def list_repository_tree(self, path: str = "", recursive: bool = True) -> List[Dict]:
        """List files in the repository (supports both gitlab and local modes)."""
        if self.source_mode == 'local':
            return self.list_repository_tree_local(path, recursive)
        else:
            return self.list_repository_tree_gitlab(path, recursive)

    def download_file_local(self, file_path: str, output_path: Path) -> bool:
        """Copy a file from local repository.

        Args:
            file_path: Path to file in repository (relative)
            output_path: Local path to save file
        """
        try:
            source_path = self.local_repo_path / file_path
            if not source_path.exists():
                print(f"File not found: {source_path}")
                return False

            shutil.copy2(source_path, output_path)
            return True
        except Exception as e:
            print(f"Error copying {file_path}: {e}")
            return False

    def download_file_gitlab(self, file_path: str, output_path: Path) -> bool:
        """Download a file from GitLab repository.

        Args:
            file_path: Path to file in repository
            output_path: Local path to save file
        """
        url = f"{self.server_url}/api/v4/projects/{self.project_id_for_api}/repository/files/{quote(file_path, safe='')}/raw"

        try:
            # First attempt: regular download
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()

            # Check if this is an LFS pointer file
            content_start = response.content[:100]
            if b'version https://git-lfs.github.com/spec/v1' in content_start:
                # This is an LFS pointer, we need to download via LFS
                # Parse the LFS pointer to get the OID and size
                content_text = response.content.decode('utf-8')
                oid_line = [line for line in content_text.split('\n') if line.startswith('oid sha256:')]
                size_line = [line for line in content_text.split('\n') if line.startswith('size ')]

                if oid_line and size_line:
                    oid = oid_line[0].split(':', 1)[1].strip()
                    size = int(size_line[0].split(' ', 1)[1].strip())

                    # Use GitLab LFS batch API to get download URL
                    lfs_batch_url = f"{self.server_url}/{self.config['gitlab']['project_id']}.git/info/lfs/objects/batch"

                    lfs_batch_payload = {
                        "operation": "download",
                        "transfers": ["basic"],
                        "objects": [
                            {
                                "oid": oid,
                                "size": size
                            }
                        ]
                    }

                    # LFS batch API requires different headers and Basic auth
                    lfs_headers = {
                        'Accept': 'application/vnd.git-lfs+json',
                        'Content-Type': 'application/vnd.git-lfs+json'
                    }

                    # Use HTTP Basic auth with token as password
                    batch_response = requests.post(
                        lfs_batch_url,
                        json=lfs_batch_payload,
                        headers=lfs_headers,
                        auth=('oauth2', self.access_token)
                    )
                    batch_response.raise_for_status()
                    batch_data = batch_response.json()

                    # Get the download URL from the batch response
                    if 'objects' in batch_data and len(batch_data['objects']) > 0:
                        obj = batch_data['objects'][0]
                        if 'actions' in obj and 'download' in obj['actions']:
                            download_url = obj['actions']['download']['href']
                            download_headers = obj['actions']['download'].get('header', {})

                            # Download the actual LFS file
                            lfs_response = requests.get(download_url, headers=download_headers, stream=True)
                            lfs_response.raise_for_status()

                            with open(output_path, 'wb') as f:
                                for chunk in lfs_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            return True
                        else:
                            print(f"No download action in LFS batch response for {file_path}")
                            return False
                    else:
                        print(f"No objects in LFS batch response for {file_path}")
                        return False
                else:
                    print(f"Could not parse LFS OID/size from {file_path}")
                    return False
            else:
                # Regular file, save it
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                return True

        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            return False

    def download_file(self, file_path: str, output_path: Path) -> bool:
        """Download/copy a file (supports both gitlab and local modes).

        Args:
            file_path: Path to file in repository
            output_path: Local path to save file
        """
        if self.source_mode == 'local':
            return self.download_file_local(file_path, output_path)
        else:
            return self.download_file_gitlab(file_path, output_path)

    def explore(self):
        """Explore the repository structure to understand the layout."""
        print("Exploring repository structure...")
        print(f"Source mode: {self.source_mode}")
        if self.source_mode == 'local':
            print(f"Local path: {self.local_repo_path}")
        else:
            print(f"Server: {self.server_url}")
            print(f"Project: {self.config['gitlab']['project_id']}")
        print()

        try:
            items = self.list_repository_tree()

            # Categorize files
            json_files = [item for item in items if item['name'].endswith('.json')]
            codex_files = [item for item in items if item['name'].endswith('.codex')]
            audio_files = [item for item in items if item['name'].endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a'))]

            print(f"Found {len(items)} total items")
            print(f"Found {len(json_files)} JSON files")
            print(f"Found {len(codex_files)} CODEX files (transcription files)")
            print(f"Found {len(audio_files)} audio files")
            print()

            if codex_files:
                print("Sample CODEX files (transcriptions):")
                for item in codex_files[:10]:
                    print(f"  - {item['path']}")
                print()

            if json_files:
                print("Sample JSON files:")
                for item in json_files[:5]:
                    print(f"  - {item['path']}")
                print()

            if audio_files:
                print("Sample audio files:")
                for item in audio_files[:5]:
                    print(f"  - {item['path']}")
                print()

            return {
                'all_items': items,
                'json_files': json_files,
                'codex_files': codex_files,
                'audio_files': audio_files
            }

        except Exception as e:
            print(f"Error exploring repository: {e}")
            return None

    def download_json_file(self, json_path: str) -> Optional[Dict]:
        """Download and parse a JSON file."""
        temp_path = self.output_dir / "temp.json"

        if self.download_file(json_path, temp_path):
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            temp_path.unlink()  # Delete temp file
            return data

        return None

    def romanize_text(self, text: str) -> str:
        """Romanize text using uroman if enabled.

        Args:
            text: The text to romanize

        Returns:
            Romanized text if uroman is enabled, otherwise original text
        """
        if not self.uroman_enabled or not text or not self.uroman_instance:
            return text

        try:
            # Use uroman to romanize the text
            romanized = self.uroman_instance.romanize_string(text, lcode=self.uroman_language)
            return romanized
        except Exception as e:
            print(f"Warning: uroman failed to romanize text: {e}")
            print("Returning original text")
            return text

    def filter_text(self, text: str, suppress_uroman: bool = False) -> Optional[str]:
        """Apply all text filtering operations.

        This method consolidates all text filtering operations including:
        - Removing HTML tags
        - Replacing &nbsp; with spaces
        - Stripping whitespace
        - Replacing duplicate spaces with single spaces
        - Romanizing text (if enabled and not suppressed)

        Args:
            text: The text to filter
            suppress_uroman: If True, skip uroman romanization even if enabled in config

        Returns:
            Filtered text or None if the text is empty after filtering
        """
        if not text:
            return None

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Replace &nbsp; with actual space
        text = text.replace('&nbsp;', ' ')

        # Strip leading/trailing whitespace
        text = text.strip()

        # Replace duplicate spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Return None if text is empty after filtering
        if not text:
            return None

        # Apply romanization if enabled and not suppressed
        if not suppress_uroman:
            text = self.romanize_text(text)

        return text

    def extract_text_from_cell(self, cell: Dict, suppress_uroman: bool = False) -> Optional[str]:
        """Extract text from a cell based on configured text_source.

        Args:
            cell: The cell data from CODEX file
            suppress_uroman: If True, skip uroman romanization (keeps text in original script)

        Returns:
            Extracted text or None if not found
        """
        if self.text_source == 'value':
            # Use current value field
            value = cell.get('value', '')
            return self.filter_text(value, suppress_uroman=suppress_uroman)

        elif self.text_source == 'edit_history':
            # Extract from edit history
            metadata = cell.get('metadata', {})
            edits = metadata.get('edits', [])

            if not edits:
                return None

            if self.edit_history_selection == 'initial_import':
                # Try to find initial-import edit first
                for edit in edits:
                    if edit.get('type') == 'initial-import':
                        value = edit.get('value', '')
                        if value and not value.strip().startswith('<'):
                            filtered = self.filter_text(value, suppress_uroman=suppress_uroman)
                            if filtered:
                                return filtered
                # Fallback to first plain text edit
                for edit in edits:
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value, suppress_uroman=suppress_uroman)
                        if filtered:
                            return filtered

            elif self.edit_history_selection == 'first':
                # Use first edit with plain text
                for edit in edits:
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value, suppress_uroman=suppress_uroman)
                        if filtered:
                            return filtered

            elif self.edit_history_selection == 'last':
                # Use last edit with plain text
                for edit in reversed(edits):
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value, suppress_uroman=suppress_uroman)
                        if filtered:
                            return filtered

            return None

        elif self.text_source == 'transcription':
            # Extract transcription from audio attachment
            metadata = cell.get('metadata', {})
            attachments = metadata.get('attachments', {})
            selected_audio_id = metadata.get('selectedAudioId')

            if selected_audio_id and selected_audio_id in attachments:
                audio_info = attachments[selected_audio_id]
                transcription_data = audio_info.get('transcription', {})
                text = transcription_data.get('content', '')
                # Apply filtering (including romanization if enabled and not suppressed)
                return self.filter_text(text, suppress_uroman=suppress_uroman)

            return None

        return None

    def extract_audio_transcriptions(self, codex_data: Dict, audio_files_map: Dict[str, str]) -> List[Dict]:
        """Extract audio file paths and transcriptions from a CODEX file.

        Args:
            codex_data: The parsed CODEX JSON data
            audio_files_map: Map of audio IDs to actual file paths in the repository
        """
        results = []

        if 'cells' not in codex_data:
            return results

        for cell in codex_data['cells']:
            metadata = cell.get('metadata', {})
            attachments = metadata.get('attachments', {})
            selected_audio_id = metadata.get('selectedAudioId')
            verse_id = metadata.get('id', 'unknown')

            # Get the selected audio attachment
            if selected_audio_id and selected_audio_id in attachments:
                audio_info = attachments[selected_audio_id]

                # Skip deleted or missing audio
                if audio_info.get('isDeleted', False) or audio_info.get('isMissing', False):
                    continue

                # Get text using extract_text_from_cell for all text sources
                text = self.extract_text_from_cell(cell)

                # Get language from transcription if using transcription source
                language = None
                if self.text_source == 'transcription':
                    transcription_data = audio_info.get('transcription', {})
                    language = transcription_data.get('language')

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
                            'language': language
                        }
                    })

        return results

    def build_audio_files_map(self, items: List[Dict]) -> Dict[str, str]:
        """Build a map of audio IDs to their actual file paths in the repository."""
        audio_map = {}

        # Get all audio files
        audio_files = [item for item in items if item['name'].endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.webm'))]

        print(f"Building audio file map from {len(audio_files)} audio files...")

        for audio_file in audio_files:
            # The audio ID is simply the filename without the extension
            # e.g., "audio-oba-1-1-1763655321982-7y3i8jupz.m4a" -> "audio-oba-1-1-1763655321982-7y3i8jupz"
            filename = audio_file['name']
            audio_id = filename.rsplit('.', 1)[0]  # Remove extension
            audio_map[audio_id] = audio_file['path']

        print(f"Mapped {len(audio_map)} audio IDs to file paths")
        return audio_map

    def process(self):
        """Process all CODEX files and create HuggingFace dataset."""
        print("Processing all CODEX files and downloading audio...")
        print()

        # Extract all audio-transcription pairs
        pairs = self._process_all_codex_files()

        if pairs:
            print(f"\nTotal audio-transcription pairs found: {len(pairs)}")

            # Download audio files and create CSV
            self.download_audio_and_create_csv(pairs)
        else:
            print("No audio-transcription pairs found")

    def _process_all_codex_files(self) -> List[Dict]:
        """Process all CODEX files and extract audio-transcription pairs.

        Continues processing files until max_records complete pairs are found,
        or all files are processed.
        """
        print("Listing all files in repository...")
        items = self.list_repository_tree()
        codex_files = [item for item in items if item['name'].endswith('.codex')]

        print(f"Found {len(codex_files)} CODEX files")

        # Build audio file map
        audio_files_map = self.build_audio_files_map(items)

        all_results = []
        max_records = self.max_records if self.max_records else float('inf')

        for idx, codex_file in enumerate(codex_files, 1):
            codex_path = codex_file['path']
            print(f"\n[{idx}/{len(codex_files)}] Processing {codex_path}...")

            codex_data = self.download_json_file(codex_path)
            if codex_data:
                results = self.extract_audio_transcriptions(codex_data, audio_files_map)
                print(f"  Found {len(results)} audio-transcription pairs")

                # Add results, but only up to max_records
                if max_records == float('inf'):
                    # No limit, add all results
                    all_results.extend(results)
                elif len(all_results) < max_records:
                    remaining = int(max_records - len(all_results))
                    all_results.extend(results[:remaining])

                    # Check if we've reached the limit
                    if len(all_results) >= max_records:
                        print(f"\n✓ Reached configured max_records limit of {self.max_records} complete pairs")
                        break
            else:
                print(f"  Failed to download {codex_path}")

        return all_results

    def download_audio_and_create_csv(self, audio_transcription_pairs: List[Dict]):
        """Download audio files and create the HuggingFace metadata CSV."""
        csv_path = self.output_dir / self.csv_filename

        print("\nDownloading audio files and creating CSV...")
        print(f"Output directory: {self.output_dir}")
        print(f"Audio directory: {self.audio_dir}")

        # Get audio_dir path from config and normalize to use forward slashes
        audio_dir_prefix = self.config['dataset']['audio_dir'].replace('\\', '/')

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['file_name', 'transcription'])

            for idx, pair in enumerate(audio_transcription_pairs, 1):
                audio_url = pair['audio_url']
                transcription = pair['transcription']

                # Extract filename from URL
                audio_filename = Path(audio_url).name
                output_audio_path = self.audio_dir / audio_filename

                print(f"[{idx}/{len(audio_transcription_pairs)}] Downloading {audio_filename}...")

                # Download audio file
                if self.download_file(audio_url, output_audio_path):
                    # Write to CSV with audio_dir prefix and forward slashes
                    relative_path = f"{audio_dir_prefix}/{audio_filename}"
                    writer.writerow([relative_path, transcription])
                else:
                    print(f"  Failed to download {audio_url}")

        print("\nDataset created successfully!")
        print(f"CSV file: {csv_path}")
        print(f"Audio files: {self.audio_dir}")
        print(f"Total records: {len(audio_transcription_pairs)}")

    def debug(self):
        """Debug mode: Examine LEV.codex in detail."""
        print("Debug mode: Examining LEV.codex in detail...")
        print()

        # Get all items and build audio map
        items = self.list_repository_tree()
        audio_map = self.build_audio_files_map(items)

        # Download LEV.codex
        codex_data = self.download_json_file("files/target/LEV.codex")

        if codex_data and 'cells' in codex_data:
            print(f"\nTotal cells in LEV.codex: {len(codex_data['cells'])}")

            # Check first few cells with attachments
            cells_with_audio = 0
            cells_with_transcription = 0
            cells_with_selected = 0

            for idx, cell in enumerate(codex_data['cells'][:10]):
                metadata = cell.get('metadata', {})
                attachments = metadata.get('attachments', {})
                selected_audio_id = metadata.get('selectedAudioId')
                verse_id = metadata.get('id', 'unknown')

                if attachments:
                    cells_with_audio += 1
                    print(f"\nCell {idx} ({verse_id}):")
                    print(f"  Attachments: {len(attachments)}")
                    print(f"  Selected audio ID: {selected_audio_id}")

                    if selected_audio_id:
                        cells_with_selected += 1
                        if selected_audio_id in attachments:
                            audio_info = attachments[selected_audio_id]
                            print(f"  Is deleted: {audio_info.get('isDeleted', False)}")
                            print(f"  Is missing: {audio_info.get('isMissing', False)}")

                            transcription = audio_info.get('transcription', {})
                            if transcription.get('content'):
                                cells_with_transcription += 1
                                print("  Has transcription: Yes")
                                print(f"  Transcription: {transcription.get('content')[:50]}...")
                            else:
                                print("  Has transcription: No")

                            # Check if audio ID is in map
                            if selected_audio_id in audio_map:
                                print(f"  ✓ Found in audio map: {audio_map[selected_audio_id]}")
                            else:
                                print("  ✗ NOT found in audio map")
                                print(f"  Looking for: {selected_audio_id}")

            print("\nSummary (first 10 cells):")
            print(f"  Cells with attachments: {cells_with_audio}")
            print(f"  Cells with selected audio: {cells_with_selected}")
            print(f"  Cells with transcription: {cells_with_transcription}")

            # Show some audio IDs from the map
            print("\nSample audio IDs in map:")
            for audio_id in list(audio_map.keys())[:5]:
                print(f"  {audio_id} -> {audio_map[audio_id]}")

    def examine(self, json_path: Optional[str] = None):
        """Examine a specific JSON/CODEX file.

        Args:
            json_path: Path to the JSON file to examine. If not provided, uses first CODEX file found.
        """
        if not json_path:
            # Default to examining a codex file from files/targets
            repo_info = self.explore()
            if repo_info and repo_info['codex_files']:
                json_path = repo_info['codex_files'][0]['path']
            else:
                print("No .codex files found in repository")
                return

        print(f"Downloading and examining: {json_path}")
        print()

        data = self.download_json_file(json_path)
        if data:
            print("JSON Structure:")
            print("-" * 60)
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])  # First 2000 chars
            print()
            if len(json.dumps(data)) > 2000:
                print("... (truncated, showing first 2000 characters)")
                print()
            print(f"Total JSON size: {len(json.dumps(data))} characters")
            print()

            # Try to understand the structure
            if isinstance(data, dict):
                print(f"Top-level keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"JSON is a list with {len(data)} items")
                if len(data) > 0:
                    print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
        else:
            print("Failed to download JSON file")

    def get_project_path(self) -> Optional[str]:
        """Get the project path (namespace/project).

        Returns the project path if known, or fetches it from the API if only
        the numeric ID was provided.
        """
        if self.project_path:
            return self.project_path
        # Fetch from API if we only have numeric ID
        if self.project_id_number:
            project_info = self._get_project_info()
            self.project_path = project_info.get('path_with_namespace')
            return self.project_path
        return None

    def get_project_id_number(self) -> Optional[str]:
        """Get the numeric project ID.

        Returns the numeric ID if known, or fetches it from the API if only
        the path was provided.
        """
        if self.project_id_number:
            return self.project_id_number
        # Fetch from API if we only have path
        if self.project_path:
            project_info = self._get_project_info()
            self.project_id_number = str(project_info.get('id'))
            return self.project_id_number
        return None

    # =========================================================================
    # Upload Methods
    # =========================================================================

    def _get_project_info(self) -> Dict[str, any]:
        """Get project information from GitLab API.

        Returns:
            Dictionary with project info including 'path_with_namespace' and 'default_branch'
        """
        if not hasattr(self, '_cached_project_info'):
            url = f"{self.server_url}/api/v4/projects/{self.project_id_for_api}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            self._cached_project_info = response.json()
        return self._cached_project_info

    def _get_project_path_for_lfs(self) -> str:
        """Get the project path (namespace/project) for LFS operations.

        The LFS batch API requires the project path, not the numeric ID.
        """
        # Use cached project_path if available
        if self.project_path:
            return self.project_path
        # Otherwise fetch from API
        project_info = self._get_project_info()
        self.project_path = project_info.get('path_with_namespace')
        return self.project_path

    def _get_default_branch(self) -> str:
        """Get the default branch of the repository."""
        project_info = self._get_project_info()
        return project_info.get('default_branch', 'main')

    def _compute_lfs_pointer_from_opener(
        self,
        content_opener: Callable[[], ContextManager[BinaryIO]]
    ) -> Dict[str, any]:
        """Compute LFS pointer data using a content opener callback.

        Args:
            content_opener: A callable that returns a context manager yielding
                           a file-like object with the content

        Returns:
            Dictionary with oid (sha256 hash) and size
        """
        with content_opener() as f:
            content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
        return {
            'oid': sha256_hash,
            'size': len(content)
        }

    def _create_lfs_pointer_content(self, oid: str, size: int) -> str:
        """Create the content of an LFS pointer file.

        Args:
            oid: SHA256 hash of the file content
            size: Size of the file in bytes

        Returns:
            LFS pointer file content as string
        """
        return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"

    def _upload_lfs_objects(self, objects: List[Dict]) -> Dict[str, Dict]:
        """Upload objects to LFS storage using the batch API.

        Args:
            objects: List of dicts with 'oid', 'size', and 'content_opener' keys.
                    content_opener is a callable that returns a context manager
                    yielding a file-like object.

        Returns:
            Dictionary mapping oid to upload result (success/error)
        """
        if not objects:
            return {}

        # Get project path for LFS URL (LFS requires path, not numeric ID)
        project_path = self._get_project_path_for_lfs()

        # Prepare batch request
        lfs_batch_url = f"{self.server_url}/{project_path}.git/info/lfs/objects/batch"

        batch_objects = [{'oid': obj['oid'], 'size': obj['size']} for obj in objects]

        lfs_batch_payload = {
            "operation": "upload",
            "transfers": ["basic"],
            "objects": batch_objects
        }

        lfs_headers = {
            'Accept': 'application/vnd.git-lfs+json',
            'Content-Type': 'application/vnd.git-lfs+json'
        }

        # Request upload URLs
        batch_response = requests.post(
            lfs_batch_url,
            json=lfs_batch_payload,
            headers=lfs_headers,
            auth=('oauth2', self.access_token)
        )
        batch_response.raise_for_status()
        batch_data = batch_response.json()

        # Build a map of oid -> content_opener for easy lookup
        opener_map = {obj['oid']: obj['content_opener'] for obj in objects}
        size_map = {obj['oid']: obj['size'] for obj in objects}

        # Process each object in the response
        results = {}
        for obj in batch_data.get('objects', []):
            oid = obj['oid']

            # Check for errors in the batch response
            if 'error' in obj:
                results[oid] = {
                    'success': False,
                    'error': f"LFS batch error: {obj['error'].get('message', 'Unknown error')}"
                }
                continue

            # Check if upload action is present (object may already exist)
            if 'actions' not in obj or 'upload' not in obj['actions']:
                # Object already exists in LFS storage
                results[oid] = {'success': True, 'already_exists': True}
                continue

            # Upload the object
            upload_action = obj['actions']['upload']
            upload_url = upload_action['href']
            content_opener = opener_map[oid]
            file_size = size_map[oid]

            # Start with headers from the batch response
            upload_headers = dict(upload_action.get('header', {}))
            # Add headers required by LFS spec for basic transfer
            upload_headers['Content-Type'] = 'application/octet-stream'
            # Remove Transfer-Encoding if present - nginx rejects chunked encoding
            upload_headers.pop('Transfer-Encoding', None)

            try:
                # Use the content opener to get a file-like object
                # For path-based files, this opens the file directly
                # For inline content, this returns a BytesIO wrapper
                with content_opener() as file_obj:
                    upload_response = requests.put(
                        upload_url,
                        data=file_obj,
                        headers={**upload_headers, 'Content-Length': str(file_size)}
                    )
                    upload_response.raise_for_status()
                results[oid] = {'success': True}
            except Exception as e:
                results[oid] = {'success': False, 'error': str(e)}

        return results

    def _create_commit_with_files(
        self,
        files: List[Dict],
        commit_message: str,
        branch: str
    ) -> Dict[str, any]:
        """Create a commit with multiple file actions.

        Args:
            files: List of dicts with 'path', 'content', and 'action' keys
                   action can be 'create' or 'update'
            commit_message: Commit message
            branch: Branch to commit to

        Returns:
            Commit result from GitLab API
        """
        url = f"{self.server_url}/api/v4/projects/{self.project_id_for_api}/repository/commits"

        actions = []
        for file_info in files:
            action = {
                'action': file_info.get('action', 'create'),
                'file_path': file_info['path'],
                'content': file_info['content']
            }
            # For binary content, we need to base64 encode it
            if file_info.get('encoding') == 'base64':
                action['encoding'] = 'base64'
            actions.append(action)

        payload = {
            'branch': branch,
            'commit_message': commit_message,
            'actions': actions
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _check_file_exists(self, file_path: str, branch: str) -> bool:
        """Check if a file exists in the repository.

        Args:
            file_path: Path to check
            branch: Branch to check on

        Returns:
            True if file exists, False otherwise
        """
        url = f"{self.server_url}/api/v4/projects/{self.project_id_for_api}/repository/files/{quote(file_path, safe='')}"
        params = {'ref': branch}

        response = requests.head(url, headers=self.headers, params=params)
        return response.status_code == 200

    def upload_batch(
        self,
        files: List[Dict[str, any]],
        commit_message: Optional[str] = None
    ) -> Dict[str, any]:
        """Upload multiple files to GitLab in a single commit.

        Each file dict must have:
            - remote_path: Path in the repository
            - lfs: Boolean indicating whether to use LFS (required, no default)

        And one of:
            - local_path: Path to local file to upload
            - content: Content to upload (str or bytes)

        Args:
            files: List of file specifications
            commit_message: Optional commit message (auto-generated if not provided)

        Returns:
            Dictionary with:
                - success: True if all files uploaded successfully
                - commit_sha: Git commit SHA (if any files succeeded)
                - files_uploaded: List of successfully uploaded files
                - files_failed: List of failed files with error messages
                - error_message: Overall error message if complete failure
        """
        if not files:
            return {
                'success': True,
                'commit_sha': None,
                'files_uploaded': [],
                'files_failed': [],
                'error_message': None
            }

        # Validate required fields
        if not self.access_token:
            return {
                'success': False,
                'commit_sha': None,
                'files_uploaded': [],
                'files_failed': [],
                'error_message': "No access token provided"
            }

        if not self.project_id_for_api:
            return {
                'success': False,
                'commit_sha': None,
                'files_uploaded': [],
                'files_failed': [],
                'error_message': "No project ID or path provided"
            }

        # Get default branch
        try:
            branch = self._get_default_branch()
        except Exception as e:
            return {
                'success': False,
                'commit_sha': None,
                'files_uploaded': [],
                'files_failed': [],
                'error_message': f"Failed to get default branch: {e}"
            }

        files_uploaded = []
        files_failed = []
        lfs_objects = []  # Objects to upload to LFS
        commit_files = []  # Files to include in the commit

        # Process each file
        for file_spec in files:
            remote_path = file_spec.get('remote_path')
            use_lfs = file_spec.get('lfs')
            local_path = file_spec.get('local_path')
            content = file_spec.get('content')

            # Validate required fields
            if not remote_path:
                files_failed.append({
                    'remote_path': remote_path,
                    'error': "Missing 'remote_path' field"
                })
                continue

            if use_lfs is None:
                files_failed.append({
                    'remote_path': remote_path,
                    'error': "Missing 'lfs' field - must explicitly specify True or False"
                })
                continue

            if local_path is None and content is None:
                files_failed.append({
                    'remote_path': remote_path,
                    'error': "Must provide either 'local_path' or 'content'"
                })
                continue

            # Create content opener - this defers reading the file until needed
            # For path-based files, content is read from disk when the opener is called
            # For inline content, the bytes are wrapped in BytesIO
            try:
                if local_path:
                    if not os.path.exists(local_path):
                        files_failed.append({
                            'remote_path': remote_path,
                            'error': f"File not found: {local_path}"
                        })
                        continue
                    content_opener = make_file_opener(local_path)
                else:
                    # Content provided directly - convert to bytes if needed
                    if isinstance(content, str):
                        content_bytes = content.encode('utf-8')
                    else:
                        content_bytes = content
                    content_opener = make_content_opener(content_bytes)
            except Exception as e:
                files_failed.append({
                    'remote_path': remote_path,
                    'error': f"Failed to prepare content: {e}"
                })
                continue

            # Check if file exists to determine action
            file_exists = self._check_file_exists(remote_path, branch)
            action = 'update' if file_exists else 'create'

            if use_lfs:
                # Compute LFS pointer using the opener (reads file for hash)
                pointer_data = self._compute_lfs_pointer_from_opener(content_opener)
                pointer_content = self._create_lfs_pointer_content(
                    pointer_data['oid'],
                    pointer_data['size']
                )

                # Add to LFS upload queue with opener (file will be read again for upload)
                lfs_objects.append({
                    'oid': pointer_data['oid'],
                    'size': pointer_data['size'],
                    'content_opener': content_opener,
                    'remote_path': remote_path
                })

                # Add pointer file to commit
                commit_files.append({
                    'path': remote_path,
                    'content': pointer_content,
                    'action': action,
                    'lfs': True,
                    'size': pointer_data['size'],
                    'oid': pointer_data['oid']
                })
            else:
                # Regular file - need to read content for commit
                # (non-LFS files are included directly in the commit payload)
                with content_opener() as f:
                    file_content = f.read()

                # For text content, use string; for binary, base64 encode
                try:
                    # Try to decode as UTF-8 text
                    text_content = file_content.decode('utf-8')
                    commit_files.append({
                        'path': remote_path,
                        'content': text_content,
                        'action': action,
                        'lfs': False,
                        'size': len(file_content)
                    })
                except UnicodeDecodeError:
                    # Binary content - base64 encode
                    commit_files.append({
                        'path': remote_path,
                        'content': base64.b64encode(file_content).decode('ascii'),
                        'action': action,
                        'encoding': 'base64',
                        'lfs': False,
                        'size': len(file_content)
                    })

        # Upload LFS objects first
        if lfs_objects:
            try:
                lfs_results = self._upload_lfs_objects(lfs_objects)

                # Check for LFS upload failures
                for obj in lfs_objects:
                    oid = obj['oid']
                    result = lfs_results.get(oid, {'success': False, 'error': 'No response from LFS'})

                    if not result.get('success'):
                        # Remove from commit files and add to failed
                        commit_files = [f for f in commit_files if f.get('oid') != oid]
                        files_failed.append({
                            'remote_path': obj['remote_path'],
                            'error': f"LFS upload failed: {result.get('error', 'Unknown error')}"
                        })
            except Exception as e:
                # All LFS uploads failed
                for obj in lfs_objects:
                    files_failed.append({
                        'remote_path': obj['remote_path'],
                        'error': f"LFS batch upload failed: {e}"
                    })
                # Remove all LFS files from commit
                commit_files = [f for f in commit_files if not f.get('lfs')]

        # Create commit with all files
        if commit_files:
            # Generate commit message if not provided
            if not commit_message:
                if len(commit_files) == 1:
                    commit_message = f"Upload {commit_files[0]['path']}"
                else:
                    # Find common folder
                    paths = [f['path'] for f in commit_files]
                    common_prefix = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
                    if common_prefix:
                        commit_message = f"Upload {len(commit_files)} files to {common_prefix}/"
                    else:
                        commit_message = f"Upload {len(commit_files)} files"

            try:
                commit_result = self._create_commit_with_files(
                    commit_files,
                    commit_message,
                    branch
                )

                # Mark all committed files as uploaded
                for file_info in commit_files:
                    files_uploaded.append({
                        'remote_path': file_info['path'],
                        'lfs': file_info.get('lfs', False),
                        'size': file_info.get('size', 0)
                    })

                return {
                    'success': len(files_failed) == 0,
                    'commit_sha': commit_result.get('id'),
                    'files_uploaded': files_uploaded,
                    'files_failed': files_failed,
                    'error_message': None if len(files_failed) == 0 else f"{len(files_failed)} file(s) failed to upload"
                }

            except Exception as e:
                # Commit failed - all files failed
                for file_info in commit_files:
                    files_failed.append({
                        'remote_path': file_info['path'],
                        'error': f"Commit failed: {e}"
                    })

                return {
                    'success': False,
                    'commit_sha': None,
                    'files_uploaded': [],
                    'files_failed': files_failed,
                    'error_message': f"Commit failed: {e}"
                }
        else:
            # No files to commit (all failed earlier)
            return {
                'success': False,
                'commit_sha': None,
                'files_uploaded': [],
                'files_failed': files_failed,
                'error_message': "No files to commit - all files failed validation or LFS upload"
            }


class CLI:
    """GitLab to HuggingFace Dataset Converter CLI.

    Available commands:
    - explore: Explore the repository structure
    - examine: Examine a specific JSON/CODEX file
    - debug: Debug mode with detailed analysis
    - process: Process all CODEX files and create HuggingFace dataset
    - upload: Upload files to GitLab with optional LFS support

    All commands support:
    - --config_path parameter (default: config.yaml)
    - Config overrides using dot notation (e.g., --dataset.output_dir="/some/path")

    Examples:
        # Use default config
        python gitlab_to_hf_dataset.py process

        # Override output directory
        python gitlab_to_hf_dataset.py process --dataset.output_dir="/tmp/output"

        # Override multiple config values
        python gitlab_to_hf_dataset.py process --dataset.output_dir="/tmp/output" --dataset.max_records=100

        # Use different config file with overrides
        python gitlab_to_hf_dataset.py process --config_path="custom.yaml" --dataset.output_dir="/tmp/output"

        # Upload a single file to LFS
        python gitlab_to_hf_dataset.py upload --token=TOKEN --project-id=454 \\
            --lfs --path=gpu_jobs/test/model.pt --file=./checkpoint.pt

        # Upload inline content (no LFS)
        python gitlab_to_hf_dataset.py upload --token=TOKEN --project-id=454 \\
            --path=gpu_jobs/job_123/response.yaml --content="state: completed"

        # For complex uploads with multiple file groups, use upload_to_gitlab.py
    """

    def explore(self, config_path: str = "config.yaml", **config_overrides):
        """Explore the repository structure to understand the layout.

        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
            **config_overrides: Config overrides using dot notation
                              (e.g., dataset.output_dir="/path")
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path, config_overrides=config_overrides)
        downloader.explore()

    def examine(self, json_path: Optional[str] = None, config_path: str = "config.yaml", **config_overrides):
        """Examine a specific JSON/CODEX file.

        Args:
            json_path: Path to the JSON file to examine. If not provided, uses first CODEX file found.
            config_path: Path to configuration YAML file (default: config.yaml)
            **config_overrides: Config overrides using dot notation
                              (e.g., dataset.output_dir="/path")
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path, config_overrides=config_overrides)
        downloader.examine(json_path=json_path)

    def debug(self, config_path: str = "config.yaml", **config_overrides):
        """Debug mode: Examine LEV.codex in detail.

        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
            **config_overrides: Config overrides using dot notation
                              (e.g., dataset.output_dir="/path")
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path, config_overrides=config_overrides)
        downloader.debug()

    def process(self, config_path: str = "config.yaml", **config_overrides):
        """Process all CODEX files and create HuggingFace dataset.

        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
            **config_overrides: Config overrides using dot notation
                              (e.g., dataset.output_dir="/path")
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path, config_overrides=config_overrides)
        downloader.process()

    def upload(
        self,
        token: Optional[str] = None,
        gitlab_url: Optional[str] = None,
        project_id: Optional[str] = None,
        project_path: Optional[str] = None,
        config_path: Optional[str] = None,
        commit_message: Optional[str] = None,
        # File specification parameters
        lfs: bool = False,
        folder: Optional[str] = None,
        files: Optional[str] = None,
        path: Optional[str] = None,
        file: Optional[str] = None,
        content: Optional[str] = None,
    ):
        """Upload files to GitLab with optional LFS support.

        This is a simplified interface for single file group uploads.
        For more complex uploads with multiple file groups, use upload_to_gitlab.py.

        Args:
            token: GitLab access token (or use GITLAB_TOKEN env var)
            gitlab_url: GitLab server URL (defaults to https://git.genesisrnd.com)
            project_id: Numeric project ID
            project_path: Project path like "namespace/project"
            config_path: Optional path to config.yaml for credentials
            commit_message: Optional commit message (auto-generated if not provided)
            lfs: Use LFS for file uploads (default: False)
            folder: Remote folder for --files uploads
            files: Comma-separated list of local files to upload to folder
            path: Remote path for single file upload
            file: Local file to upload to --path location
            content: Inline content to upload to --path location

        Examples:
            # Upload a single file to LFS
            python gitlab_to_hf_dataset.py upload --token=TOKEN --project-id=454 \\
                --lfs --path=gpu_jobs/test/model.pt --file=./checkpoint.pt

            # Upload inline content (no LFS)
            python gitlab_to_hf_dataset.py upload --token=TOKEN --project-id=454 \\
                --path=gpu_jobs/job_123/response.yaml --content="state: completed"

            # Upload multiple files to a folder
            python gitlab_to_hf_dataset.py upload --token=TOKEN --project-id=454 \\
                --lfs --folder=gpu_jobs/job_123/models/ --files=model.pt,checkpoint.pt

        For complex uploads with multiple file groups, use upload_to_gitlab.py:
            python upload_to_gitlab.py --token=TOKEN --project-id=454 \\
                --lfs --folder gpu_jobs/audio/ --files a.wav b.wav \\
                --no-lfs --path gpu_jobs/response.yaml --content "done"
        """
        print("=" * 60)
        print("GitLab LFS Upload")
        print("=" * 60)
        print()

        # Get token from parameter or environment
        access_token = token or os.environ.get('GITLAB_TOKEN')
        if not access_token:
            print("ERROR: No GitLab token provided. Use --token or set GITLAB_TOKEN env var.")
            return {'success': False, 'error_message': 'No token provided'}

        # Require project identification
        if not project_id and not project_path:
            print("ERROR: Must provide either --project-id or --project-path")
            return {'success': False, 'error_message': 'No project specified'}

        # Build file specifications from parameters
        file_specs = []

        # Handle --folder + --files pattern
        if folder and files:
            folder_path = folder.rstrip('/')
            file_list = [f.strip() for f in files.split(',')]
            for local_path in file_list:
                filename = os.path.basename(local_path)
                remote_path = f"{folder_path}/{filename}"
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': lfs
                })
        elif folder:
            print("ERROR: --folder requires --files to be specified")
            return {'success': False, 'error_message': '--folder requires --files'}

        # Handle --path + --file or --path + --content pattern
        if path:
            if file:
                file_specs.append({
                    'local_path': file,
                    'remote_path': path,
                    'lfs': lfs
                })
            elif content:
                file_specs.append({
                    'content': content,
                    'remote_path': path,
                    'lfs': lfs
                })
            else:
                print("ERROR: --path requires either --file or --content")
                return {'success': False, 'error_message': '--path requires --file or --content'}

        if not file_specs:
            print("ERROR: No files specified for upload")
            print("Use --lfs --path=<remote> --file=<local> or --path=<remote> --content=<text>")
            print("Or use --lfs --folder=<dir> --files=<file1>,<file2>")
            return {'success': False, 'error_message': 'No files specified'}

        print(f"Files to upload: {len(file_specs)}")
        for spec in file_specs:
            lfs_str = "LFS" if spec.get('lfs') else "git"
            if 'local_path' in spec:
                print(f"  [{lfs_str}] {spec['local_path']} -> {spec['remote_path']}")
            else:
                content_preview = spec.get('content', '')[:50]
                if len(spec.get('content', '')) > 50:
                    content_preview += '...'
                print(f"  [{lfs_str}] <content> -> {spec['remote_path']}")
        print()

        # Create uploader instance
        uploader = GitLabDatasetDownloader(
            config_path=config_path,
            gitlab_url=gitlab_url,
            access_token=access_token,
            project_id=project_id,
            project_path=project_path,
        )

        print(f"Server: {uploader.server_url}")
        print(f"Project: {uploader.project_path or uploader.project_id_number}")
        print()

        # Perform upload
        result = uploader.upload_batch(file_specs, commit_message=commit_message)

        # Print results
        if result['success']:
            print("✓ Upload successful!")
            print(f"  Commit SHA: {result['commit_sha']}")
            print(f"  Files uploaded: {len(result['files_uploaded'])}")
            for f in result['files_uploaded']:
                lfs_str = "LFS" if f.get('lfs') else "git"
                print(f"    [{lfs_str}] {f['remote_path']} ({f.get('size', 0)} bytes)")
        else:
            print("✗ Upload failed!")
            if result.get('error_message'):
                print(f"  Error: {result['error_message']}")
            if result.get('files_failed'):
                print(f"  Failed files: {len(result['files_failed'])}")
                for f in result['files_failed']:
                    print(f"    {f['remote_path']}: {f['error']}")
            if result.get('files_uploaded'):
                print(f"  Partially uploaded: {len(result['files_uploaded'])} files")

        return result


def main():
    """Main entry point using Google Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()