"""
Sound Bank Retrieval API (provider.py)
======================================
High-speed access to audio slices from the Master WAV container.

Features:
- Intensity-based retrieval (normalized 0.0-1.0 scale)
- Lazy-loading with sample-accurate seeking (never loads full file)
- Spectral verification to confirm vocal pocket notch
- Closest-match algorithm for energy matching
"""

import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
from dataclasses import dataclass
from .database import SoundBankDB, SoundAsset


@dataclass
class IntensityRange:
    """Cached intensity range for normalization."""
    min_rms: float
    max_rms: float
    range_span: float
    
    def normalize(self, raw_rms: float) -> float:
        """Convert raw RMS to 0.0-1.0 scale."""
        if self.range_span == 0:
            return 0.5
        return (raw_rms - self.min_rms) / self.range_span
    
    def denormalize(self, normalized: float) -> float:
        """Convert 0.0-1.0 scale back to raw RMS."""
        return self.min_rms + (normalized * self.range_span)


@dataclass
class SpectralAnalysis:
    """Result of spectral notch verification."""
    has_notch: bool
    notch_depth_db: float
    notch_center_freq: float
    notch_bandwidth: float
    full_band_energy: float
    notch_band_energy: float
    confidence: float  # 0.0 to 1.0


class SoundBankProvider:
    """
    Retrieval API for the Sound Bank.
    
    Provides intensity-based audio retrieval using high-speed seeking
    to pull specific samples without loading the entire Master WAV.
    
    The provider holds a persistent connection to the database and
    uses lazy-loading for all audio access.
    
    Usage:
        provider = SoundBankProvider('output/master_bank.wav', 'output/bank.db')
        
        # Get audio by normalized intensity (0.0 = quietest, 1.0 = loudest)
        audio = provider.get_by_normalized_intensity(0.8, category='808')
        
        # Get the closest match to a specific energy level
        asset_id = provider.get_id_by_intensity(target_energy=0.15)
        audio = provider.get_audio_by_id(asset_id)
        
        # Verify spectral notch is present
        result = provider.verify_spectral_notch(asset_id=1)
        print(f"Notch present: {result.has_notch}, Depth: {result.notch_depth_db}dB")
    """
    
    # Vocal pocket frequency range
    NOTCH_LOW_FREQ = 1000   # Hz
    NOTCH_HIGH_FREQ = 3000  # Hz
    NOTCH_THRESHOLD_DB = -6.0  # Minimum attenuation to consider notch present
    
    def __init__(
        self,
        master_wav_path: str = "master_bank.wav",
        database_path: str = "bank.db"
    ):
        """
        Initialize the Sound Bank Provider.
        
        Opens the database connection and caches intensity range for normalization.
        
        Args:
            master_wav_path: Path to the Master WAV container
            database_path: Path to the SQLite database index
        """
        self.master_wav_path = Path(master_wav_path)
        self.database_path = Path(database_path)
        
        # Validate files exist
        if not self.master_wav_path.exists():
            raise FileNotFoundError(f"Master WAV not found: {master_wav_path}")
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        # Initialize database connection (held open for performance)
        self.db = SoundBankDB(str(database_path))
        self.db.connect()
        
        # Cache master WAV info (metadata only, no audio loaded)
        self._wav_info = sf.info(str(self.master_wav_path))
        self.sample_rate = self._wav_info.samplerate
        self.channels = self._wav_info.channels
        self.total_samples = self._wav_info.frames
        
        # Cache intensity range for normalization
        self._intensity_range: Optional[IntensityRange] = None
        self._cache_intensity_range()
        
        # Cache all assets for fast lookup
        self._asset_cache: Dict[int, SoundAsset] = {}
        self._intensity_sorted_ids: List[int] = []
        self._build_asset_cache()
    
    def __del__(self):
        """Cleanup: close database connection."""
        if hasattr(self, 'db') and self.db.conn:
            self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db.conn:
            self.db.close()
    
    def _cache_intensity_range(self) -> None:
        """Cache min/max intensity for normalization scaling."""
        stats = self.db.get_statistics()
        intensity = stats.get('intensity_range', {})
        
        min_rms = intensity.get('min', 0.0) or 0.0
        max_rms = intensity.get('max', 1.0) or 1.0
        
        self._intensity_range = IntensityRange(
            min_rms=min_rms,
            max_rms=max_rms,
            range_span=max_rms - min_rms
        )
    
    def _build_asset_cache(self) -> None:
        """Build in-memory cache of all assets sorted by intensity."""
        assets = self.db.get_all_assets()
        
        # Build ID -> Asset map
        self._asset_cache = {a.id: a for a in assets}
        
        # Build intensity-sorted list for fast closest-match lookup
        sorted_assets = sorted(assets, key=lambda a: a.intensity_score)
        self._intensity_sorted_ids = [a.id for a in sorted_assets]
        self._intensity_sorted_values = [a.intensity_score for a in sorted_assets]
    
    @property
    def duration_seconds(self) -> float:
        """Total duration of the master bank in seconds."""
        return self.total_samples / self.sample_rate
    
    @property
    def intensity_range(self) -> IntensityRange:
        """Get the intensity range for normalization."""
        return self._intensity_range
    
    # =========================================================================
    # CORE RETRIEVAL METHODS
    # =========================================================================
    
    def get_id_by_intensity(
        self,
        target_energy: float,
        category: Optional[str] = None
    ) -> int:
        """
        Find the asset ID whose RMS energy is closest to the target.
        
        Uses binary search for O(log n) performance on large banks.
        
        Args:
            target_energy: Target RMS energy value (raw, not normalized)
            category: Optional category filter
            
        Returns:
            Asset ID of the closest match
            
        Raises:
            ValueError: If no assets match the criteria
        """
        if category:
            # Filter by category first
            candidates = [
                (aid, self._asset_cache[aid].intensity_score)
                for aid in self._intensity_sorted_ids
                if self._asset_cache[aid].category == category
            ]
            if not candidates:
                raise ValueError(f"No assets in category '{category}'")
            
            # Find closest in filtered list
            closest_id = min(candidates, key=lambda x: abs(x[1] - target_energy))[0]
            return closest_id
        
        # Use binary search on full sorted list
        if not self._intensity_sorted_ids:
            raise ValueError("Sound bank is empty")
        
        # Binary search for closest value
        idx = np.searchsorted(self._intensity_sorted_values, target_energy)
        
        # Check neighbors to find actual closest
        if idx == 0:
            return self._intensity_sorted_ids[0]
        elif idx >= len(self._intensity_sorted_ids):
            return self._intensity_sorted_ids[-1]
        else:
            # Compare with left and right neighbors
            left_diff = abs(self._intensity_sorted_values[idx - 1] - target_energy)
            right_diff = abs(self._intensity_sorted_values[idx] - target_energy)
            
            if left_diff <= right_diff:
                return self._intensity_sorted_ids[idx - 1]
            else:
                return self._intensity_sorted_ids[idx]
    
    def get_audio_by_id(self, asset_id: int) -> np.ndarray:
        """
        Get audio data for a specific asset using lazy-loading.
        
        Uses seek() to jump directly to the sample offset without
        loading the entire master file into memory.
        
        Args:
            asset_id: The asset ID to retrieve
            
        Returns:
            numpy array of audio samples
            
        Raises:
            ValueError: If asset ID not found
        """
        if asset_id not in self._asset_cache:
            raise ValueError(f"Asset ID {asset_id} not found")
        
        asset = self._asset_cache[asset_id]
        return self._read_slice_lazy(asset.start_sample, asset.end_sample)
    
    def _read_slice_lazy(self, start: int, end: int) -> np.ndarray:
        """
        Lazy-load a slice from the Master WAV using efficient seeking.
        
        NEVER loads the entire file - uses soundfile's seek() to jump
        directly to the requested position.
        
        Args:
            start: Start sample offset
            end: End sample offset
            
        Returns:
            numpy array of audio samples
        """
        # Validate bounds
        start = max(0, start)
        end = min(self.total_samples, end)
        
        if start >= end:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        
        num_samples = end - start
        
        # Use soundfile for efficient seeking - O(1) seek operation
        with sf.SoundFile(str(self.master_wav_path)) as wav:
            wav.seek(start)  # Jump directly to position (no loading)
            audio = wav.read(num_samples, dtype='float32')
        
        return audio
    
    # =========================================================================
    # NORMALIZED INTENSITY METHODS (0.0 - 1.0 SCALE)
    # =========================================================================
    
    def get_by_normalized_intensity(
        self,
        intensity: float,
        category: Optional[str] = None
    ) -> np.ndarray:
        """
        Get audio by normalized intensity (0.0 = quietest, 1.0 = loudest).
        
        This is the primary method for external systems to request audio
        based on a simple intensity scale.
        
        Args:
            intensity: Normalized intensity (0.0 to 1.0)
                - 0.0 = quietest asset in the bank
                - 0.5 = medium intensity
                - 1.0 = loudest asset in the bank
            category: Optional category filter
            
        Returns:
            numpy array of audio samples
            
        Example:
            # Get a loud drum loop
            audio = provider.get_by_normalized_intensity(0.9, category='loops')
            
            # Get a soft atmosphere
            audio = provider.get_by_normalized_intensity(0.2, category='atmospheres')
        """
        # Clamp to valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Convert normalized intensity to raw RMS
        target_rms = self._intensity_range.denormalize(intensity)
        
        # Find closest match
        asset_id = self.get_id_by_intensity(target_rms, category=category)
        
        return self.get_audio_by_id(asset_id)
    
    def get_by_normalized_intensity_with_info(
        self,
        intensity: float,
        category: Optional[str] = None
    ) -> Tuple[np.ndarray, SoundAsset, float]:
        """
        Get audio with metadata and actual intensity value.
        
        Args:
            intensity: Normalized intensity (0.0 to 1.0)
            category: Optional category filter
            
        Returns:
            Tuple of (audio, asset_metadata, actual_normalized_intensity)
        """
        intensity = max(0.0, min(1.0, intensity))
        target_rms = self._intensity_range.denormalize(intensity)
        
        asset_id = self.get_id_by_intensity(target_rms, category=category)
        asset = self._asset_cache[asset_id]
        audio = self.get_audio_by_id(asset_id)
        
        actual_normalized = self._intensity_range.normalize(asset.intensity_score)
        
        return audio, asset, actual_normalized
    
    def get_loudest(self, category: Optional[str] = None) -> Tuple[np.ndarray, SoundAsset]:
        """Get the loudest asset (intensity = 1.0)."""
        audio, asset, _ = self.get_by_normalized_intensity_with_info(1.0, category)
        return audio, asset
    
    def get_quietest(self, category: Optional[str] = None) -> Tuple[np.ndarray, SoundAsset]:
        """Get the quietest asset (intensity = 0.0)."""
        audio, asset, _ = self.get_by_normalized_intensity_with_info(0.0, category)
        return audio, asset
    
    # =========================================================================
    # SPECTRAL VERIFICATION
    # =========================================================================
    
    def verify_spectral_notch(
        self,
        asset_id: Optional[int] = None,
        audio: Optional[np.ndarray] = None
    ) -> SpectralAnalysis:
        """
        Verify that the 1kHz-3kHz spectral notch is present.
        
        This proves the proprietary transformation was applied correctly.
        Analyzes the frequency content and measures the energy reduction
        in the vocal pocket (1-3kHz) compared to surrounding frequencies.
        
        Args:
            asset_id: Asset ID to analyze (loads audio automatically)
            audio: Or provide audio directly
            
        Returns:
            SpectralAnalysis with notch verification results
            
        Example:
            result = provider.verify_spectral_notch(asset_id=1)
            if result.has_notch:
                print(f"Notch confirmed: {result.notch_depth_db:.1f}dB at {result.notch_center_freq}Hz")
        """
        if audio is None:
            if asset_id is None:
                raise ValueError("Must provide either asset_id or audio")
            audio = self.get_audio_by_id(asset_id)
        
        # Ensure mono for analysis
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute power spectrum
        n_fft = min(4096, len(audio))
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=n_fft)
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        # Define frequency bands
        # Below notch: 500-1000Hz
        # Notch band: 1000-3000Hz
        # Above notch: 3000-6000Hz
        
        below_mask = (freqs >= 500) & (freqs < self.NOTCH_LOW_FREQ)
        notch_mask = (freqs >= self.NOTCH_LOW_FREQ) & (freqs <= self.NOTCH_HIGH_FREQ)
        above_mask = (freqs > self.NOTCH_HIGH_FREQ) & (freqs <= 6000)
        
        # Calculate average energy in each band
        below_energy = np.mean(psd_db[below_mask]) if np.any(below_mask) else -60
        notch_energy = np.mean(psd_db[notch_mask]) if np.any(notch_mask) else -60
        above_energy = np.mean(psd_db[above_mask]) if np.any(above_mask) else -60
        
        # Average of surrounding bands
        surrounding_energy = (below_energy + above_energy) / 2
        
        # Calculate notch depth (how much quieter is the notch band)
        notch_depth = notch_energy - surrounding_energy
        
        # Determine if notch is present (significant energy reduction)
        has_notch = notch_depth < self.NOTCH_THRESHOLD_DB
        
        # Calculate confidence based on consistency
        # Higher confidence if both above and below bands are similar
        band_consistency = 1.0 - min(1.0, abs(below_energy - above_energy) / 20)
        
        # Confidence also based on notch depth
        if has_notch:
            depth_confidence = min(1.0, abs(notch_depth) / 12)  # -12dB = full confidence
        else:
            depth_confidence = 0.0
        
        confidence = (band_consistency + depth_confidence) / 2
        
        return SpectralAnalysis(
            has_notch=has_notch,
            notch_depth_db=notch_depth,
            notch_center_freq=2000,  # Center of 1-3kHz band
            notch_bandwidth=2000,     # 3000 - 1000
            full_band_energy=surrounding_energy,
            notch_band_energy=notch_energy,
            confidence=confidence
        )
    
    def verify_all_assets(self) -> Dict[int, SpectralAnalysis]:
        """
        Verify spectral notch for all assets in the bank.
        
        Returns:
            Dict mapping asset_id to SpectralAnalysis results
        """
        results = {}
        for asset_id in self._asset_cache:
            results[asset_id] = self.verify_spectral_notch(asset_id=asset_id)
        return results
    
    # =========================================================================
    # LEGACY COMPATIBILITY METHODS
    # =========================================================================
    
    def get_by_intensity(
        self,
        target_rms: float,
        tolerance: float = 0.1,
        category: Optional[str] = None,
        return_all_matches: bool = False
    ) -> Union[np.ndarray, List[Tuple[SoundAsset, np.ndarray]]]:
        """
        Get audio that matches a target RMS energy level.
        
        Legacy method - prefer get_by_normalized_intensity() for new code.
        """
        if return_all_matches:
            # Get all within tolerance
            results = []
            for asset_id, asset in self._asset_cache.items():
                if category and asset.category != category:
                    continue
                if abs(asset.intensity_score - target_rms) <= tolerance:
                    audio = self.get_audio_by_id(asset_id)
                    results.append((asset, audio))
            return sorted(results, key=lambda x: abs(x[0].intensity_score - target_rms))
        else:
            asset_id = self.get_id_by_intensity(target_rms, category)
            return self.get_audio_by_id(asset_id)
    
    def request_slice(
        self,
        asset_id: Optional[int] = None,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None
    ) -> np.ndarray:
        """
        Request a specific audio slice from the Master WAV.
        
        Uses lazy-loading - never loads the entire file.
        """
        if asset_id is not None:
            return self.get_audio_by_id(asset_id)
        
        if start_sample is None or end_sample is None:
            raise ValueError("Must provide either asset_id or start_sample/end_sample")
        
        return self._read_slice_lazy(start_sample, end_sample)
    
    def get_by_category(
        self,
        category: str,
        limit: int = 10
    ) -> List[Tuple[SoundAsset, np.ndarray]]:
        """Get all audio assets in a category."""
        results = []
        count = 0
        
        for asset_id, asset in self._asset_cache.items():
            if asset.category == category:
                audio = self.get_audio_by_id(asset_id)
                results.append((asset, audio))
                count += 1
                if count >= limit:
                    break
        
        return results
    
    def get_random(
        self,
        category: Optional[str] = None,
        min_intensity: float = 0.0,
        max_intensity: float = 1.0
    ) -> Tuple[SoundAsset, np.ndarray]:
        """Get a random audio asset."""
        import random
        
        # Filter candidates
        candidates = [
            aid for aid, asset in self._asset_cache.items()
            if (category is None or asset.category == category)
            and min_intensity <= asset.intensity_score <= max_intensity
        ]
        
        if not candidates:
            raise ValueError("No assets match the criteria")
        
        asset_id = random.choice(candidates)
        asset = self._asset_cache[asset_id]
        audio = self.get_audio_by_id(asset_id)
        
        return asset, audio
    
    # =========================================================================
    # TAG-BASED RETRIEVAL METHODS
    # =========================================================================
    
    def get_by_tag(
        self,
        tag: str,
        intensity_target: Optional[float] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> Tuple[SoundAsset, np.ndarray]:
        """
        Get audio by tag, optionally filtered by intensity and category.
        
        Args:
            tag: Tag name to search for
            intensity_target: Optional target intensity (0.0-1.0) for filtering
            category: Optional category filter
            limit: Maximum results to consider
            
        Returns:
            Tuple of (SoundAsset, audio_array)
        """
        # Get assets with the tag
        candidates = self.db.get_assets_by_tag(tag, limit=limit)
        
        if not candidates:
            raise ValueError(f"No assets found with tag '{tag}'")
        
        # Filter by category if specified
        if category:
            candidates = [a for a in candidates if a.category == category]
            if not candidates:
                raise ValueError(f"No assets with tag '{tag}' in category '{category}'")
        
        # If intensity target specified, find closest match
        if intensity_target is not None:
            normalized_target = self._intensity_range.denormalize(intensity_target)
            closest = min(candidates, key=lambda a: abs(a.intensity_score - normalized_target))
            asset = closest
        else:
            # Return first match
            asset = candidates[0]
        
        audio = self.get_audio_by_id(asset.id)
        return asset, audio
    
    def get_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        intensity_target: Optional[float] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[SoundAsset, np.ndarray]]:
        """
        Get audio by multiple tags.
        
        Args:
            tags: List of tag names
            match_all: If True, asset must have ALL tags; if False, ANY tag
            intensity_target: Optional target intensity for sorting results
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of (SoundAsset, audio_array) tuples
        """
        candidates = self.db.get_assets_by_tags(tags, match_all=match_all, limit=limit)
        
        if not candidates:
            raise ValueError(f"No assets found with tags {tags}")
        
        # Filter by category if specified
        if category:
            candidates = [a for a in candidates if a.category == category]
        
        # Sort by intensity if target specified
        if intensity_target is not None:
            normalized_target = self._intensity_range.denormalize(intensity_target)
            candidates.sort(key=lambda a: abs(a.intensity_score - normalized_target))
        
        # Load audio for results
        results = []
        for asset in candidates[:limit]:
            audio = self.get_audio_by_id(asset.id)
            results.append((asset, audio))
        
        return results
    
    def get_asset_tags(self, asset_id: int) -> List[str]:
        """Get all tags for a specific asset."""
        return self.db.get_asset_tags(asset_id)
    
    def search_by_characteristics(
        self,
        characteristics: List[str],
        intensity_target: Optional[float] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[SoundAsset, np.ndarray]]:
        """
        Search for assets matching sonic characteristics.
        
        Example characteristics: 'punchy', 'warm', 'bright', 'metallic', 'organic'
        
        Args:
            characteristics: List of characteristic tags
            intensity_target: Optional intensity filter
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of (SoundAsset, audio_array) tuples
        """
        return self.get_by_tags(
            characteristics,
            match_all=False,
            intensity_target=intensity_target,
            category=category,
            limit=limit
        )
    
    def get_asset_info(self, asset_id: int) -> Optional[SoundAsset]:
        """Get asset metadata without loading audio."""
        return self._asset_cache.get(asset_id)
    
    def list_assets(self, category: Optional[str] = None) -> List[SoundAsset]:
        """List all assets without loading audio."""
        if category:
            return [a for a in self._asset_cache.values() if a.category == category]
        return list(self._asset_cache.values())
    
    def get_statistics(self) -> dict:
        """Get sound bank statistics."""
        db_stats = self.db.get_statistics()
        
        return {
            **db_stats,
            'master_wav_path': str(self.master_wav_path),
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'total_samples': self.total_samples,
            'master_duration_seconds': self.duration_seconds,
            'intensity_range': {
                'min': self._intensity_range.min_rms,
                'max': self._intensity_range.max_rms,
                'span': self._intensity_range.range_span
            }
        }
    
    def export_asset(
        self,
        asset_id: int,
        output_path: str,
        format: str = 'WAV'
    ) -> str:
        """Export a single asset to a file."""
        audio = self.get_audio_by_id(asset_id)
        output_path = Path(output_path)
        
        sf.write(str(output_path), audio, self.sample_rate, format=format)
        
        return str(output_path)
