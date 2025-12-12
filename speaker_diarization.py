#!/usr/bin/env python3
"""
Speaker Diarization Script for TTS Fine-tuning
Extracts speaker embeddings and clusters them to assign speaker IDs.
Supports resemblyzer and pyannote.audio methods.
"""

import os
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Handles speaker diarization and clustering."""
    
    def __init__(
        self,
        method: str = "resemblyzer",
        clustering_method: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int = 3,
        n_clusters: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        """
        Initialize the speaker diarizer.
        
        Args:
            method: "resemblyzer" or "pyannote"
            clustering_method: "hdbscan" or "kmeans"
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN core points
            n_clusters: Number of clusters for K-means (if None, auto-detect)
            min_speakers: Minimum number of speakers (for validation)
            max_speakers: Maximum number of speakers (for validation)
        """
        self.method = method
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Initialize embedding model
        if method == "resemblyzer":
            try:
                from resemblyzer import VoiceEncoder, preprocess_wav
                self.encoder = VoiceEncoder()
                self.preprocess_wav = preprocess_wav
                logger.info("Loaded Resemblyzer voice encoder")
            except ImportError:
                raise ImportError(
                    "resemblyzer not installed. Install with: pip install resemblyzer"
                )
        elif method == "pyannote":
            try:
                from pyannote.audio import Model
                from pyannote.audio.pipelines import SpeakerDiarization
                # Note: pyannote requires authentication token
                logger.warning(
                    "pyannote.audio requires HuggingFace authentication. "
                    "Make sure you have accepted the user agreement and set HF_TOKEN."
                )
                # This will be initialized when needed
                self.pipeline = None
            except ImportError:
                raise ImportError(
                    "pyannote.audio not installed. Install with: pip install pyannote.audio"
                )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'resemblyzer' or 'pyannote'")
    
    def extract_embedding_resemblyzer(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        Extract speaker embedding using Resemblyzer.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding vector or None if failed
        """
        try:
            import librosa
            
            # Load audio
            wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            
            # Preprocess for resemblyzer
            wav = self.preprocess_wav(wav)
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            return None
    
    def extract_embeddings(
        self,
        audio_files: List[Path],
        cache_path: Optional[Path] = None
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract embeddings from all audio files.
        
        Args:
            audio_files: List of audio file paths
            cache_path: Optional path to cache embeddings
            
        Returns:
            Tuple of (embeddings list, filenames list)
        """
        # Check if cached embeddings exist
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True).item()
            return data['embeddings'], data['filenames']
        
        logger.info(f"Extracting embeddings from {len(audio_files)} audio files")
        
        embeddings = []
        filenames = []
        
        for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
            if self.method == "resemblyzer":
                embedding = self.extract_embedding_resemblyzer(audio_path)
            else:
                # pyannote method would go here
                logger.error("pyannote method not yet implemented")
                embedding = None
            
            if embedding is not None:
                embeddings.append(embedding)
                filenames.append(audio_path.name)
        
        logger.info(f"Successfully extracted {len(embeddings)} embeddings")
        
        # Cache embeddings if path provided
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, {
                'embeddings': embeddings,
                'filenames': filenames
            })
            logger.info(f"Cached embeddings to {cache_path}")
        
        return embeddings, filenames
    
    def cluster_embeddings(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Cluster speaker embeddings to assign speaker IDs.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Array of cluster labels (speaker IDs)
        """
        # Convert to numpy array
        X = np.array(embeddings)
        
        logger.info(f"Clustering {len(X)} embeddings using {self.clustering_method}")
        
        if self.clustering_method == "hdbscan":
            try:
                import hdbscan
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric='euclidean'
                )
                
                labels = clusterer.fit_predict(X)
                
                # Count clusters (excluding noise points labeled as -1)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                logger.info(f"HDBSCAN found {n_clusters} clusters")
                logger.info(f"Noise points: {n_noise}")
                
                # Assign noise points to nearest cluster
                if n_noise > 0:
                    logger.info("Assigning noise points to nearest clusters")
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Get non-noise points
                    non_noise_mask = labels != -1
                    non_noise_X = X[non_noise_mask]
                    non_noise_labels = labels[non_noise_mask]
                    
                    # Find nearest neighbors for noise points
                    noise_mask = labels == -1
                    noise_X = X[noise_mask]
                    
                    if len(non_noise_X) > 0:
                        nn = NearestNeighbors(n_neighbors=1)
                        nn.fit(non_noise_X)
                        _, indices = nn.kneighbors(noise_X)
                        
                        # Assign noise points to nearest cluster
                        noise_labels = non_noise_labels[indices.flatten()]
                        labels[noise_mask] = noise_labels
                
                return labels
                
            except ImportError:
                raise ImportError(
                    "hdbscan not installed. Install with: pip install hdbscan"
                )
        
        elif self.clustering_method == "kmeans":
            from sklearn.cluster import KMeans
            
            # Determine number of clusters
            if self.n_clusters is None:
                # Auto-determine using elbow method or silhouette score
                from sklearn.metrics import silhouette_score
                
                best_score = -1
                best_k = 2
                
                # Try different numbers of clusters
                min_k = self.min_speakers or 2
                max_k = min(self.max_speakers or 20, len(X) // 2)
                
                logger.info(f"Auto-determining number of clusters (trying {min_k} to {max_k})")
                
                for k in range(min_k, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_temp = kmeans.fit_predict(X)
                    score = silhouette_score(X, labels_temp)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                self.n_clusters = best_k
                logger.info(f"Selected {self.n_clusters} clusters (silhouette score: {best_score:.3f})")
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            logger.info(f"K-means clustering complete with {self.n_clusters} clusters")
            
            return labels
        
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def save_speaker_mapping(
        self,
        filenames: List[str],
        speaker_ids: np.ndarray,
        output_path: Path
    ):
        """
        Save speaker ID mapping to JSON file.
        
        Args:
            filenames: List of audio filenames
            speaker_ids: Array of speaker IDs
            output_path: Path to output JSON file
        """
        # Create mapping dictionary
        mapping = {}
        for filename, speaker_id in zip(filenames, speaker_ids):
            mapping[filename] = int(speaker_id)
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved speaker mapping to {output_path}")
        
        # Print statistics
        unique_speakers = len(set(speaker_ids))
        logger.info(f"Total unique speakers: {unique_speakers}")
        
        # Count samples per speaker
        from collections import Counter
        speaker_counts = Counter(speaker_ids)
        logger.info("Samples per speaker:")
        for speaker_id, count in sorted(speaker_counts.items()):
            logger.info(f"  Speaker {speaker_id}: {count} samples")
    
    def generate_cluster_samples(
        self,
        audio_dir: Path,
        filenames: List[str],
        speaker_ids: np.ndarray,
        output_dir: Path,
        samples_per_cluster: int = 3
    ):
        """
        Generate sample audio files for each cluster for verification.
        
        Args:
            audio_dir: Directory containing audio files
            filenames: List of audio filenames
            speaker_ids: Array of speaker IDs
            output_dir: Output directory for sample files
            samples_per_cluster: Number of samples per cluster
        """
        import shutil
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating cluster samples in {output_dir}")
        
        # Group files by speaker
        from collections import defaultdict
        speaker_files = defaultdict(list)
        for filename, speaker_id in zip(filenames, speaker_ids):
            speaker_files[speaker_id].append(filename)
        
        # Copy sample files for each speaker
        for speaker_id, files in speaker_files.items():
            # Take first N samples
            samples = files[:samples_per_cluster]
            
            for idx, filename in enumerate(samples):
                src_path = audio_dir / filename
                if src_path.exists():
                    dst_filename = f"speaker_{speaker_id}_sample_{idx + 1}{src_path.suffix}"
                    dst_path = output_dir / dst_filename
                    shutil.copy2(src_path, dst_path)
        
        logger.info(f"Generated samples for {len(speaker_files)} speakers")


def main():
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization on audio files"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (e.g., config.yaml)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing preprocessed audio files"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata.csv file (optional, for filtering)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for speaker analysis results"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["resemblyzer", "pyannote"],
        default="resemblyzer",
        help="Diarization method (default: resemblyzer)"
    )
    parser.add_argument(
        "--clustering",
        type=str,
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="Clustering method (default: hdbscan)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        help="Number of clusters for K-means (auto-detect if not specified)"
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        default=True,
        help="Generate sample audio for each cluster (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load config if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get parameters from config or args
    if config:
        tts_config = config.get('tts', {})
        diarization_config = tts_config.get('speaker_diarization', {})
        method = args.method or diarization_config.get('method', 'resemblyzer')
        clustering_method = args.clustering or diarization_config.get('clustering_method', 'hdbscan')
        min_cluster_size = diarization_config.get('min_cluster_size', 5)
        min_samples = diarization_config.get('min_samples', 3)
        n_clusters = args.n_clusters or diarization_config.get('n_clusters')
        min_speakers = diarization_config.get('min_speakers')
        max_speakers = diarization_config.get('max_speakers')
        generate_samples = args.generate_samples or diarization_config.get('generate_samples', True)
        samples_per_cluster = diarization_config.get('samples_per_cluster', 3)

        # Get paths from config
        paths = tts_config.get('paths', {})
        audio_dir = Path(args.audio_dir or paths.get('preprocessed_audio', ''))
        output_dir = Path(args.output_dir or paths.get('speaker_analysis', ''))
        metadata_path = Path(args.metadata or paths.get('preprocessed_metadata', '')) if (args.metadata or paths.get('preprocessed_metadata')) else None
    else:
        # Use command line arguments
        method = args.method
        clustering_method = args.clustering
        min_cluster_size = 5
        min_samples = 3
        n_clusters = args.n_clusters
        min_speakers = None
        max_speakers = None
        generate_samples = args.generate_samples
        samples_per_cluster = 3
        
        if not args.audio_dir or not args.output_dir:
            logger.error("Either --config or both --audio_dir and --output_dir must be provided")
            return 1
        
        audio_dir = Path(args.audio_dir)
        output_dir = Path(args.output_dir)
        metadata_path = Path(args.metadata) if args.metadata else None
    
    # Validate audio directory
    if not audio_dir.exists():
        logger.error(f"Audio directory does not exist: {audio_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files
    if metadata_path and metadata_path.exists():
        logger.info(f"Reading audio files from metadata: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            audio_files = [audio_dir / row['file_name'] for row in reader]
    else:
        logger.info(f"Scanning audio directory: {audio_dir}")
        audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return 1
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(
        method=method,
        clustering_method=clustering_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_clusters=n_clusters,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # Extract embeddings
    embeddings_cache = output_dir / "embeddings.npy"
    embeddings, filenames = diarizer.extract_embeddings(
        audio_files,
        cache_path=embeddings_cache
    )
    
    if not embeddings:
        logger.error("No embeddings extracted")
        return 1
    
    # Cluster embeddings
    speaker_ids = diarizer.cluster_embeddings(embeddings)
    
    # Save speaker mapping
    speaker_ids_path = output_dir / "speaker_ids.json"
    diarizer.save_speaker_mapping(filenames, speaker_ids, speaker_ids_path)
    
    # Generate cluster samples if requested
    if generate_samples:
        samples_dir = output_dir / "cluster_samples"
        diarizer.generate_cluster_samples(
            audio_dir,
            filenames,
            speaker_ids,
            samples_dir,
            samples_per_cluster=samples_per_cluster
        )
    
    logger.info("\nSpeaker diarization complete!")
    logger.info(f"Speaker IDs saved to: {speaker_ids_path}")
    if generate_samples:
        logger.info(f"Cluster samples saved to: {samples_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())