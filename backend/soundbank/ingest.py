#!/usr/bin/env python3
"""
Sound Bank Ingestion Engine (ingest.py)
=======================================
CLI tool that processes raw audio samples and creates:
- master_bank.wav: Single concatenated WAV container
- bank.db: SQLite index with metadata

Usage:
    python -m soundbank.ingest /path/to/samples --output ./output --category 808
    python -m soundbank.ingest /path/to/samples -o ./output -c loops --normalize-rms 0.1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf
from datetime import datetime

from .database import SoundBankDB
from .transformations import Transformations
from .classifier import AudioClassifier


class IngestionEngine:
    """
    Processes raw audio samples into a Master WAV container with database index.
    
    Features:
    - Spectral notch filter (1kHz-3kHz vocal pocket)
    - Peak/RMS normalization
    - Sample rate conversion to 44.1kHz
    - Metadata extraction (RMS energy, mid-range density)
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif'}
    TARGET_SAMPLE_RATE = 44100
    
    def __init__(
        self,
        output_dir: str = "output",
        target_rms: float = 0.1,
        target_peak: float = 0.95,
        notch_attenuation_db: float = -12.0
    ):
        """
        Initialize the Ingestion Engine.
        
        Args:
            output_dir: Directory for output files
            target_rms: Target RMS level for normalization
            target_peak: Target peak level for normalization
            notch_attenuation_db: Attenuation for vocal pocket notch (dB)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_rms = target_rms
        self.target_peak = target_peak
        self.notch_attenuation_db = notch_attenuation_db
        
        self.master_wav_path = self.output_dir / "master_bank.wav"
        self.database_path = self.output_dir / "bank.db"
        
        # Processing statistics
        self.processed_count = 0
        self.failed_count = 0
        self.total_samples_written = 0
        
    def process_directory(
        self,
        input_dir: str,
        category: str = "loops",
        apply_notch: bool = True,
        normalize_mode: str = "rms",
        recursive: bool = True
    ) -> Tuple[int, int]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Input directory containing audio files
            category: Category for all files ('808', 'snare', 'loops', 'atmospheres')
            apply_notch: Apply spectral notch filter
            normalize_mode: 'rms', 'peak', or 'none'
            recursive: Search subdirectories
            
        Returns:
            Tuple of (processed_count, failed_count)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all audio files
        audio_files = self._find_audio_files(input_path, recursive)
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return 0, 0
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Initialize database
        db = SoundBankDB(str(self.database_path))
        with db:
            db.create_schema()
            db.set_metadata('created_at', datetime.now().isoformat())
            db.set_metadata('sample_rate', str(self.TARGET_SAMPLE_RATE))
            db.set_metadata('notch_attenuation_db', str(self.notch_attenuation_db))
        
        # Initialize master WAV writer
        # First pass: process all files and collect audio
        all_processed_audio = []
        all_metadata = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"Processing [{i+1}/{len(audio_files)}]: {audio_file.name}")
            
            try:
                processed, metadata = self._process_file(
                    audio_file, category, apply_notch, normalize_mode
                )
                all_processed_audio.append(processed)
                all_metadata.append({
                    'filename': audio_file.name,
                    **metadata
                })
                self.processed_count += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                self.failed_count += 1
        
        if not all_processed_audio:
            print("No files were successfully processed")
            return 0, self.failed_count
        
        # Write concatenated master WAV
        print("\nWriting master_bank.wav...")
        self._write_master_wav(all_processed_audio, all_metadata, db, category)
        
        # Final statistics
        with db:
            stats = db.get_statistics()
        
        print(f"\n{'='*50}")
        print("INGESTION COMPLETE")
        print(f"{'='*50}")
        print(f"Processed: {self.processed_count} files")
        print(f"Failed: {self.failed_count} files")
        print(f"Total duration: {stats['total_duration_seconds']:.2f} seconds")
        print(f"Output: {self.master_wav_path}")
        print(f"Index: {self.database_path}")
        
        return self.processed_count, self.failed_count
    
    def _find_audio_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all supported audio files in directory."""
        files = []
        pattern = '**/*' if recursive else '*'
        
        for ext in self.SUPPORTED_FORMATS:
            files.extend(directory.glob(f"{pattern}{ext}"))
            files.extend(directory.glob(f"{pattern}{ext.upper()}"))
        
        return sorted(files)
    
    def _process_file(
        self,
        file_path: Path,
        category: str,
        apply_notch: bool,
        normalize_mode: str
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a single audio file.
        
        Returns:
            Tuple of (processed_audio, metadata_dict)
        """
        # Load audio
        audio, sample_rate = sf.read(str(file_path))
        
        # Convert to mono for consistent processing
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to target rate
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio = Transformations.resample(
                audio, sample_rate, self.TARGET_SAMPLE_RATE
            )
            print(f"  Resampled: {sample_rate}Hz -> {self.TARGET_SAMPLE_RATE}Hz")
        
        # Calculate pre-processing metrics
        original_rms = Transformations.calculate_rms_energy(audio)
        
        # Apply spectral notch filter (vocal pocket carving)
        spectral_hole = 0.0
        if apply_notch:
            audio, spectral_hole = Transformations.apply_spectral_notch(
                audio,
                sample_rate=self.TARGET_SAMPLE_RATE,
                attenuation_db=self.notch_attenuation_db
            )
            print(f"  Applied notch filter: {self.notch_attenuation_db}dB @ 1-3kHz")
        
        # Normalize
        if normalize_mode == 'rms':
            audio = Transformations.normalize_rms(audio, self.target_rms)
            print(f"  Normalized RMS: {original_rms:.4f} -> {self.target_rms:.4f}")
        elif normalize_mode == 'peak':
            audio = Transformations.normalize_peak(audio, self.target_peak)
            print(f"  Normalized peak to {self.target_peak}")
        
        # Calculate final metrics
        rms_energy = Transformations.calculate_rms_energy(audio)
        mid_range_density = Transformations.calculate_mid_range_density(
            audio, self.TARGET_SAMPLE_RATE
        )
        
        metadata = {
            'rms_energy': rms_energy,
            'intensity_score': rms_energy,  # Using RMS as intensity score
            'mid_range_density': mid_range_density,
            'spectral_hole': spectral_hole,
            'duration_samples': len(audio)
        }
        
        return audio, metadata
    
    def _write_master_wav(
        self,
        audio_segments: List[np.ndarray],
        metadata_list: List[dict],
        db: SoundBankDB,
        category: str
    ) -> None:
        """Write all processed audio to master WAV and update database."""
        
        # Concatenate all audio
        master_audio = np.concatenate(audio_segments)
        
        # Write master WAV
        sf.write(
            str(self.master_wav_path),
            master_audio,
            self.TARGET_SAMPLE_RATE,
            subtype='PCM_24'
        )
        
        # Update database with sample offsets
        current_offset = 0
        classifier = AudioClassifier(sr=self.TARGET_SAMPLE_RATE)
        
        with db:
            for i, (audio, meta) in enumerate(zip(audio_segments, metadata_list)):
                start_sample = current_offset
                end_sample = current_offset + len(audio)
                
                # Insert asset
                asset_id = db.insert_asset(
                    original_filename=meta['filename'],
                    start_sample=start_sample,
                    end_sample=end_sample,
                    category=category,
                    intensity_score=meta['intensity_score'],
                    spectral_hole=meta['spectral_hole'],
                    rms_energy=meta['rms_energy'],
                    mid_range_density=meta['mid_range_density'],
                    sample_rate=self.TARGET_SAMPLE_RATE
                )
                
                # Classify and tag the asset
                try:
                    tags = classifier.classify(audio, sample_rate=self.TARGET_SAMPLE_RATE)
                    db.add_tags_batch(asset_id, tags)
                    print(f"  Tags: {', '.join(tags[:5])}" + ("..." if len(tags) > 5 else ""))
                except Exception as e:
                    print(f"  Warning: Could not classify - {e}")
                
                current_offset = end_sample
        
        self.total_samples_written = current_offset


def ingest_samples(
    input_dir: str,
    output_dir: str = "output",
    category: str = "loops",
    normalize_mode: str = "rms",
    target_rms: float = 0.1,
    notch_attenuation_db: float = -12.0,
    apply_notch: bool = True,
    recursive: bool = True
) -> Tuple[int, int]:
    """
    Ingest audio samples from a directory into a sound bank.
    
    This is the main API function used by the FastAPI server for background ingestion tasks.
    
    Args:
        input_dir: Path to directory containing audio files
        output_dir: Path to output directory for master_bank.wav and bank.db
        category: Sample category (808, snare, loops, atmospheres)
        normalize_mode: Normalization mode (rms, peak, or none)
        target_rms: Target RMS level (default: 0.1)
        notch_attenuation_db: Notch filter attenuation (default: -12.0)
        apply_notch: Whether to apply spectral notch filter (default: True)
        recursive: Whether to search subdirectories (default: True)
        
    Returns:
        Tuple of (processed_count, failed_count)
    """
    engine = IngestionEngine(
        output_dir=output_dir,
        target_rms=target_rms,
        notch_attenuation_db=notch_attenuation_db
    )
    
    processed, failed = engine.process_directory(
        input_dir=input_dir,
        category=category,
        apply_notch=apply_notch,
        normalize_mode=normalize_mode,
        recursive=recursive
    )
    
    return processed, failed


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sound Bank Ingestion Engine - Process audio samples into Master WAV container',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 808 samples
  python -m soundbank.ingest ./samples/808s -o ./output -c 808
  
  # Process loops with peak normalization
  python -m soundbank.ingest ./samples/loops -o ./output -c loops --normalize peak
  
  # Process without notch filter
  python -m soundbank.ingest ./samples -o ./output -c atmospheres --no-notch
  
Categories: 808, snare, loops, atmospheres
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Input directory containing audio samples'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '-c', '--category',
        default='loops',
        choices=['808', 'snare', 'loops', 'atmospheres'],
        help='Category for the samples (default: loops)'
    )
    
    parser.add_argument(
        '--normalize',
        choices=['rms', 'peak', 'none'],
        default='rms',
        help='Normalization mode (default: rms)'
    )
    
    parser.add_argument(
        '--target-rms',
        type=float,
        default=0.1,
        help='Target RMS level for normalization (default: 0.1)'
    )
    
    parser.add_argument(
        '--notch-db',
        type=float,
        default=-12.0,
        help='Notch filter attenuation in dB (default: -12.0)'
    )
    
    parser.add_argument(
        '--no-notch',
        action='store_true',
        help='Skip spectral notch filter'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )
    
    args = parser.parse_args()
    
    # Create engine
    engine = IngestionEngine(
        output_dir=args.output,
        target_rms=args.target_rms,
        notch_attenuation_db=args.notch_db
    )
    
    # Process
    try:
        processed, failed = engine.process_directory(
            input_dir=args.input_dir,
            category=args.category,
            apply_notch=not args.no_notch,
            normalize_mode=args.normalize,
            recursive=not args.no_recursive
        )
        
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
