"""
Base utilities for job handlers.

This module provides shared functionality used by all handlers.
"""

import time
import tarfile
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# File extensions that should be uploaded via Git LFS
LFS_EXTENSIONS = {
    # Model weight files
    '.pt', '.pth', '.bin', '.safetensors', '.ckpt',
    # Audio files
    '.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac', '.aac',
    # Archive files (for model bundles)
    '.tar', '.tar.gz', '.tgz', '.pth.tar',
}

# File extensions that should always use regular git (never LFS)
REGULAR_GIT_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.txt', '.log', '.csv', '.md',
    '.codex', '.py', '.sh', '.toml',
}


def should_use_lfs(file_path: Union[str, Path]) -> bool:
    """
    Determine if a file should be uploaded via Git LFS based on its extension.

    Args:
        file_path: Path to the file (can be string or Path object)

    Returns:
        True if the file should use LFS, False for regular git
    """
    path = Path(file_path)

    # Check for compound extensions like .pth.tar
    name_lower = path.name.lower()
    for ext in LFS_EXTENSIONS:
        if name_lower.endswith(ext):
            return True

    # Check single extension
    ext = path.suffix.lower()
    if ext in LFS_EXTENSIONS:
        return True
    if ext in REGULAR_GIT_EXTENSIONS:
        return False

    # Default: use LFS for unknown binary-looking files, regular for text-looking
    # This is a fallback - ideally all extensions should be explicitly listed
    return ext not in {'.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js'}


def create_tar_archive(source_dir: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> bytes:
    """
    Create a tar archive from a directory.

    Args:
        source_dir: Directory to archive
        output_path: Optional path to write the archive to. If None, returns bytes.

    Returns:
        Archive content as bytes (if output_path is None)
    """
    source_dir = Path(source_dir)

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                tar.add(file_path, arcname=arcname)

    content = buffer.getvalue()

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(content)

    return content


def extract_tar_archive(archive_path: Union[str, Path, bytes], output_dir: Union[str, Path]) -> Path:
    """
    Extract a tar archive to a directory.

    Args:
        archive_path: Path to the archive file, or bytes content
        output_dir: Directory to extract to

    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(archive_path, bytes):
        buffer = BytesIO(archive_path)
        with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
            tar.extractall(output_dir)
    else:
        with tarfile.open(archive_path, mode='r:gz') as tar:
            tar.extractall(output_dir)

    return output_dir


def get_cell_id(cell: Dict[str, Any]) -> str:
    """
    Get cell identifier, handling both old and new formats.

    Old format: metadata.id contains the reference (e.g., "MAT 1:1")
    New format: metadata.id is a UUID, reference is in metadata.data.globalReferences[0]

    Args:
        cell: Cell data from .codex file

    Returns:
        Cell identifier (UUID)
    """
    metadata = cell.get('metadata', {})
    return metadata.get('id', '')


def get_cell_reference(cell: Dict[str, Any]) -> Optional[str]:
    """
    Get cell reference (e.g., "MAT 1:1"), handling both old and new formats.

    Args:
        cell: Cell data from .codex file

    Returns:
        Cell reference string, or None if not found
    """
    metadata = cell.get('metadata', {})

    # New format: globalReferences
    data = metadata.get('data', {})
    global_refs = data.get('globalReferences', [])
    if global_refs:
        return global_refs[0]

    # Old format: id field might contain reference (if it has a colon)
    cell_id = metadata.get('id', '')
    if ':' in cell_id:
        return cell_id

    return None


def is_reference_format(item: str) -> bool:
    """
    Check if filter item is a Bible reference (contains colon).

    Args:
        item: Filter item string

    Returns:
        True if item looks like a Bible reference
    """
    return ':' in item


def cell_matches_filter(cell: Dict[str, Any], filter_item: str) -> bool:
    """
    Check if cell matches a filter item.

    Args:
        cell: Cell data from .codex file
        filter_item: Filter item (reference or UUID)

    Returns:
        True if cell matches the filter item
    """
    if is_reference_format(filter_item):
        # Match against reference
        cell_ref = get_cell_reference(cell)
        return cell_ref == filter_item
    else:
        # Match against UUID
        return get_cell_id(cell) == filter_item


def filter_cells(
    cells: List[Dict[str, Any]],
    include_verses: Optional[List[str]] = None,
    exclude_verses: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter cells based on include/exclude lists.

    Args:
        cells: List of cell data from .codex file
        include_verses: Optional list of verses to include (if provided, only these are included)
        exclude_verses: Optional list of verses to exclude

    Returns:
        Filtered list of cells
    """
    if include_verses:
        # Include only specified verses
        return [
            cell for cell in cells
            if any(cell_matches_filter(cell, v) for v in include_verses)
        ]
    elif exclude_verses:
        # Exclude specified verses
        return [
            cell for cell in cells
            if not any(cell_matches_filter(cell, v) for v in exclude_verses)
        ]
    else:
        # No filter, return all
        return cells


def cell_has_audio(cell: Dict[str, Any]) -> bool:
    """
    Check if cell has audio attached.

    Args:
        cell: Cell data from .codex file

    Returns:
        True if cell has a selected audio attachment
    """
    metadata = cell.get('metadata', {})
    selected_audio_id = metadata.get('selectedAudioId')

    if not selected_audio_id:
        return False

    attachments = metadata.get('attachments', {})
    if selected_audio_id not in attachments:
        return False

    audio_info = attachments[selected_audio_id]

    # Check if audio is deleted or missing
    if audio_info.get('isDeleted', False) or audio_info.get('isMissing', False):
        return False

    return True


def cell_has_text(cell: Dict[str, Any]) -> bool:
    """
    Check if cell has text content.

    Args:
        cell: Cell data from .codex file

    Returns:
        True if cell has non-empty text value
    """
    value = cell.get('value', '')
    return bool(value and value.strip())


def get_cells_needing_audio(
    cells: List[Dict[str, Any]],
    include_verses: Optional[List[str]] = None,
    exclude_verses: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get cells that have text but no audio (for TTS inference).

    If include_verses is specified, those verses are always included
    even if they already have audio.

    Args:
        cells: List of cell data from .codex file
        include_verses: Optional list of verses to include (overrides audio check)
        exclude_verses: Optional list of verses to exclude

    Returns:
        List of cells needing audio generation
    """
    result = []

    for cell in cells:
        # Check exclusion first
        if exclude_verses and any(cell_matches_filter(cell, v) for v in exclude_verses):
            continue

        # If explicitly included, add regardless of audio status
        if include_verses and any(cell_matches_filter(cell, v) for v in include_verses):
            if cell_has_text(cell):
                result.append(cell)
            continue

        # Default: include if has text but no audio
        if include_verses is None:
            if cell_has_text(cell) and not cell_has_audio(cell):
                result.append(cell)

    return result


def get_cells_needing_text(
    cells: List[Dict[str, Any]],
    include_verses: Optional[List[str]] = None,
    exclude_verses: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get cells that have audio but no text (for ASR inference).

    If include_verses is specified, those verses are always included
    even if they already have text.

    Args:
        cells: List of cell data from .codex file
        include_verses: Optional list of verses to include (overrides text check)
        exclude_verses: Optional list of verses to exclude

    Returns:
        List of cells needing text generation
    """
    result = []

    for cell in cells:
        # Check exclusion first
        if exclude_verses and any(cell_matches_filter(cell, v) for v in exclude_verses):
            continue

        # If explicitly included, add regardless of text status
        if include_verses and any(cell_matches_filter(cell, v) for v in include_verses):
            if cell_has_audio(cell):
                result.append(cell)
            continue

        # Default: include if has audio but no text
        if include_verses is None:
            if cell_has_audio(cell) and not cell_has_text(cell):
                result.append(cell)

    return result


def get_cells_with_audio_and_text(
    cells: List[Dict[str, Any]],
    include_verses: Optional[List[str]] = None,
    exclude_verses: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get cells that have both audio and text (for training).

    Args:
        cells: List of cell data from .codex file
        include_verses: Optional list of verses to include
        exclude_verses: Optional list of verses to exclude

    Returns:
        List of cells with both audio and text
    """
    # First apply include/exclude filter
    filtered = filter_cells(cells, include_verses, exclude_verses)

    # Then filter for cells with both audio and text
    return [
        cell for cell in filtered
        if cell_has_audio(cell) and cell_has_text(cell)
    ]


def detect_use_uroman(text: str, threshold: float = 0.5) -> bool:
    """
    Auto-detect whether uroman romanization is needed based on text content.

    Examines the alphabetic characters in ``text`` and returns ``True`` when
    more than ``threshold`` (default 50 %) of them are non-Latin script.

    This is intended to be called with the first non-empty verse/transcription
    from a dataset so that ``use_uroman`` can be inferred when the manifest
    does not specify it explicitly.

    Args:
        text: Sample text to analyse (e.g. the first non-empty verse).
        threshold: Fraction of non-Latin alphabetic characters above which
            the function returns ``True``.  Defaults to ``0.5``.

    Returns:
        ``True`` if uroman romanization is likely needed, ``False`` otherwise.
    """
    latin_count = 0
    non_latin_count = 0

    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch, '')
        except ValueError:
            name = ''
        if 'LATIN' in name:
            latin_count += 1
        else:
            non_latin_count += 1

    total = latin_count + non_latin_count
    if total == 0:
        return False
    return (non_latin_count / total) > threshold


def resolve_use_uroman(
    model_config: Dict[str, Any],
    sample_texts: List[str],
) -> bool:
    """
    Determine the effective ``use_uroman`` value.

    If the manifest explicitly sets ``model.use_uroman`` to ``True`` or
    ``False``, that value is honoured.  When the key is absent (``None``),
    the function auto-detects by inspecting the first non-empty entry in
    *sample_texts* via :func:`detect_use_uroman`.

    Args:
        model_config: The ``model`` section of the job manifest.
        sample_texts: Iterable of transcription / verse strings from the
            downloaded dataset.  Only the first non-empty entry is used.

    Returns:
        The resolved boolean value for ``use_uroman``.
    """
    explicit = model_config.get('use_uroman')
    if explicit is not None:
        return bool(explicit)

    # Auto-detect from the first non-empty text sample
    for text in sample_texts:
        stripped = text.strip() if text else ''
        if stripped:
            result = detect_use_uroman(stripped)
            print(f"  Auto-detected use_uroman={result} from sample text: "
                  f"{stripped[:80]}{'…' if len(stripped) > 80 else ''}")
            return result

    # No text available — default to False
    return False


# Default pretrained model coordinates for HuggingFace Hub
STABLETTS_DEFAULT_REPO_ID = "KdaiP/StableTTS1.1"
STABLETTS_DEFAULT_FILENAME = "StableTTS/checkpoint_0.pt"


def download_pretrained_model(
    repo_id: str = STABLETTS_DEFAULT_REPO_ID,
    filename: str = STABLETTS_DEFAULT_FILENAME,
    max_retries: int = 3,
) -> str:
    """
    Download a pretrained model from HuggingFace Hub with caching and retry logic.

    Uses huggingface_hub's built-in caching mechanism (~/.cache/huggingface/hub/)
    so the model is only downloaded once and shared across multiple jobs.

    Args:
        repo_id: HuggingFace repository ID (e.g., "KdaiP/StableTTS1.1")
        filename: File path within the repository (e.g., "StableTTS/checkpoint_0.pt")
        max_retries: Number of retry attempts with exponential backoff

    Returns:
        Local file path to the cached pretrained model

    Raises:
        RuntimeError: If download fails after all retries
    """
    from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Downloading pretrained model from HuggingFace Hub "
                  f"(repo={repo_id}, file={filename}, attempt {attempt}/{max_retries})...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"  Pretrained model cached at: {local_path}")
            return local_path
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                print(f"  Download attempt {attempt} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Download attempt {attempt} failed: {e}")

    raise RuntimeError(
        f"Failed to download pretrained model from HuggingFace Hub after {max_retries} attempts. "
        f"repo_id={repo_id}, filename={filename}. Last error: {last_error}"
    )
