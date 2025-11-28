#!/usr/bin/env python3
"""
Script to download audio files and transcriptions from GitLab and prepare
a HuggingFace AudioFolder dataset.
"""

import os
import csv
import json
import yaml
import requests
import fire
from pathlib import Path
from urllib.parse import quote
from typing import Dict, List, Optional

try:
    from uroman import Uroman
    UROMAN_AVAILABLE = True
except ImportError:
    UROMAN_AVAILABLE = False
    Uroman = None


class GitLabDatasetDownloader:
    """Downloads audio files and transcriptions from GitLab."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration from YAML file."""
        print(f"Reading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Source mode: gitlab or local
        self.source_mode = self.config.get('source_mode', 'gitlab')
        self.local_repo_path = Path(self.config.get('local_repo_path', '')) if self.source_mode == 'local' else None
        
        # GitLab configuration (only needed for gitlab mode)
        if self.source_mode == 'gitlab':
            self.server_url = self.config['gitlab']['server_url'].rstrip('/')
            self.access_token = self.config['gitlab']['access_token']
            self.project_id = quote(self.config['gitlab']['project_id'], safe='')
            self.headers = {
                'PRIVATE-TOKEN': self.access_token
            }
        
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.audio_dir = self.output_dir / self.config['dataset']['audio_dir']
        self.csv_filename = self.config['dataset']['csv_filename']
        self.max_records = self.config['dataset'].get('max_records', 0)
        self.text_source = self.config['dataset'].get('text_source', 'transcription')
        self.edit_history_selection = self.config['dataset'].get('edit_history_selection', 'initial_import')
        
        # Uroman configuration
        uroman_config = self.config.get('uroman', {})
        self.uroman_enabled = uroman_config.get('enabled', False)
        self.uroman_language = uroman_config.get('language', None)
        
        # Validate uroman availability if enabled
        if self.uroman_enabled and not UROMAN_AVAILABLE:
            raise ImportError(
                "uroman is enabled in config but not installed. "
                "Install it with: uv add uroman"
            )
        
        # Initialize uroman instance if enabled
        self.uroman_instance = Uroman() if self.uroman_enabled else None
        
        # Create output directories (including parent directories)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    def list_repository_tree_local(self, path: str = "", recursive: bool = True) -> List[Dict]:
        """List files in the local repository."""
        import os
        
        all_items = []
        search_path = self.local_repo_path / path if path else self.local_repo_path
        
        if recursive:
            # Recursively walk the directory
            for root, dirs, files in os.walk(search_path):
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
        url = f"{self.server_url}/api/v4/projects/{self.project_id}/repository/tree"
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
            
            import shutil
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
        url = f"{self.server_url}/api/v4/projects/{self.project_id}/repository/files/{quote(file_path, safe='')}/raw"
        
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
            print(f"Returning original text")
            return text
    
    def filter_text(self, text: str) -> Optional[str]:
        """Apply all text filtering operations.
        
        This method consolidates all text filtering operations including:
        - Removing HTML tags
        - Replacing &nbsp; with spaces
        - Stripping whitespace
        - Replacing duplicate spaces with single spaces
        - Romanizing text (if enabled)
        
        Args:
            text: The text to filter
            
        Returns:
            Filtered text or None if the text is empty after filtering
        """
        if not text:
            return None
        
        import re
        
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
        
        # Apply romanization if enabled
        text = self.romanize_text(text)
        
        return text
    
    def extract_text_from_cell(self, cell: Dict) -> Optional[str]:
        """Extract text from a cell based on configured text_source.
        
        Args:
            cell: The cell data from CODEX file
            
        Returns:
            Extracted text or None if not found
        """
        if self.text_source == 'value':
            # Use current value field
            value = cell.get('value', '')
            return self.filter_text(value)
            
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
                            filtered = self.filter_text(value)
                            if filtered:
                                return filtered
                # Fallback to first plain text edit
                for edit in edits:
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value)
                        if filtered:
                            return filtered
                        
            elif self.edit_history_selection == 'first':
                # Use first edit with plain text
                for edit in edits:
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value)
                        if filtered:
                            return filtered
                        
            elif self.edit_history_selection == 'last':
                # Use last edit with plain text
                for edit in reversed(edits):
                    value = edit.get('value', '')
                    if value and not value.strip().startswith('<'):
                        filtered = self.filter_text(value)
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
                # Apply filtering (including romanization if enabled)
                return self.filter_text(text)
            
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
        
        print(f"\nDownloading audio files and creating CSV...")
        print(f"Output directory: {self.output_dir}")
        print(f"Audio directory: {self.audio_dir}")
        
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
                    # Write to CSV with proper quoting
                    # The transcription is already quoted by csv.QUOTE_ALL
                    writer.writerow([audio_filename, transcription])
                else:
                    print(f"  Failed to download {audio_url}")
        
        print(f"\nDataset created successfully!")
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
                                print(f"  Has transcription: Yes")
                                print(f"  Transcription: {transcription.get('content')[:50]}...")
                            else:
                                print(f"  Has transcription: No")
                            
                            # Check if audio ID is in map
                            if selected_audio_id in audio_map:
                                print(f"  ✓ Found in audio map: {audio_map[selected_audio_id]}")
                            else:
                                print(f"  ✗ NOT found in audio map")
                                print(f"  Looking for: {selected_audio_id}")
            
            print(f"\nSummary (first 10 cells):")
            print(f"  Cells with attachments: {cells_with_audio}")
            print(f"  Cells with selected audio: {cells_with_selected}")
            print(f"  Cells with transcription: {cells_with_transcription}")
            
            # Show some audio IDs from the map
            print(f"\nSample audio IDs in map:")
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


class CLI:
    """GitLab to HuggingFace Dataset Converter CLI.
    
    Available commands:
    - explore: Explore the repository structure
    - examine: Examine a specific JSON/CODEX file
    - debug: Debug mode with detailed analysis
    - process: Process all CODEX files and create HuggingFace dataset
    
    All commands support --config_path parameter (default: config.yaml)
    """
    
    def explore(self, config_path: str = "config.yaml"):
        """Explore the repository structure to understand the layout.
        
        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path)
        downloader.explore()
    
    def examine(self, json_path: Optional[str] = None, config_path: str = "config.yaml"):
        """Examine a specific JSON/CODEX file.
        
        Args:
            json_path: Path to the JSON file to examine. If not provided, uses first CODEX file found.
            config_path: Path to configuration YAML file (default: config.yaml)
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path)
        downloader.examine(json_path=json_path)
    
    def debug(self, config_path: str = "config.yaml"):
        """Debug mode: Examine LEV.codex in detail.
        
        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path)
        downloader.debug()
    
    def process(self, config_path: str = "config.yaml"):
        """Process all CODEX files and create HuggingFace dataset.
        
        Args:
            config_path: Path to configuration YAML file (default: config.yaml)
        """
        print("=" * 60)
        print("GitLab to HuggingFace Dataset Converter")
        print("=" * 60)
        print()
        downloader = GitLabDatasetDownloader(config_path=config_path)
        downloader.process()


def main():
    """Main entry point using Google Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()