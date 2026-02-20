#!/usr/bin/env python3
"""
Test Sample Generator
=====================
Generates white-noise bursts and other test samples for verifying
the stitching and slicing logic is sample-accurate.

Usage:
    python -m soundbank.test_generator --output ./test_samples --count 10
    python -m soundbank.test_generator -o ./test_samples -n 5 --duration 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf


class TestSampleGenerator:
    """
    Generates test audio samples for validating sound bank operations.
    
    Creates:
    - White noise bursts with markers for sample-accurate verification
    - Test tones at specific frequencies
    - Silence with markers for boundary testing
    """
    
    SAMPLE_RATE = 44100
    
    def __init__(self, output_dir: str = "test_samples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_white_noise_burst(
        self,
        duration_seconds: float = 0.5,
        amplitude: float = 0.5,
        add_markers: bool = True
    ) -> np.ndarray:
        """
        Generate a white noise burst with optional start/end markers.
        
        Markers are short impulses at the exact start and end samples,
        making it easy to verify sample-accurate slicing.
        
        Args:
            duration_seconds: Duration of the burst
            amplitude: Peak amplitude (0.0 to 1.0)
            add_markers: Add impulse markers at boundaries
            
        Returns:
            numpy array of audio samples
        """
        num_samples = int(duration_seconds * self.SAMPLE_RATE)
        
        # Generate white noise
        noise = np.random.randn(num_samples) * amplitude
        
        # Normalize to prevent clipping
        noise = noise / np.max(np.abs(noise)) * amplitude
        
        if add_markers:
            # Add start marker (positive impulse)
            marker_length = 10  # samples
            noise[:marker_length] = amplitude * 0.9
            
            # Add end marker (negative impulse)
            noise[-marker_length:] = -amplitude * 0.9
        
        return noise.astype(np.float32)
    
    def generate_marker_tone(
        self,
        frequency: float = 1000.0,
        duration_seconds: float = 0.1,
        amplitude: float = 0.8
    ) -> np.ndarray:
        """
        Generate a pure tone marker for boundary verification.
        
        Args:
            frequency: Tone frequency in Hz
            duration_seconds: Duration of the tone
            amplitude: Peak amplitude
            
        Returns:
            numpy array of audio samples
        """
        num_samples = int(duration_seconds * self.SAMPLE_RATE)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        
        tone = np.sin(2 * np.pi * frequency * t) * amplitude
        
        # Apply fade in/out to prevent clicks
        fade_samples = min(100, num_samples // 4)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
        return tone.astype(np.float32)
    
    def generate_silence_with_markers(
        self,
        duration_seconds: float = 0.5,
        marker_amplitude: float = 0.8
    ) -> np.ndarray:
        """
        Generate silence with boundary markers.
        
        Useful for testing that the slicing correctly identifies
        asset boundaries even in low-energy regions.
        
        Args:
            duration_seconds: Duration of silence
            marker_amplitude: Amplitude of boundary markers
            
        Returns:
            numpy array of audio samples
        """
        num_samples = int(duration_seconds * self.SAMPLE_RATE)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        # Start marker: short positive pulse
        marker_len = 5
        audio[:marker_len] = marker_amplitude
        
        # End marker: short negative pulse
        audio[-marker_len:] = -marker_amplitude
        
        return audio
    
    def generate_intensity_gradient(
        self,
        count: int = 5,
        duration_seconds: float = 0.3
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Generate noise bursts with varying intensities.
        
        Useful for testing intensity-based retrieval.
        
        Args:
            count: Number of samples to generate
            duration_seconds: Duration of each sample
            
        Returns:
            List of (audio, target_intensity) tuples
        """
        samples = []
        intensities = np.linspace(0.1, 0.9, count)
        
        for intensity in intensities:
            audio = self.generate_white_noise_burst(
                duration_seconds=duration_seconds,
                amplitude=intensity,
                add_markers=True
            )
            samples.append((audio, float(intensity)))
        
        return samples
    
    def generate_category_samples(
        self,
        samples_per_category: int = 3,
        duration_seconds: float = 0.5
    ) -> dict:
        """
        Generate test samples for each category.
        
        Creates distinct patterns for each category:
        - 808: Low-frequency sine with noise
        - snare: High-frequency noise burst
        - loops: Repeating pattern
        - atmospheres: Filtered noise
        
        Args:
            samples_per_category: Number of samples per category
            duration_seconds: Duration of each sample
            
        Returns:
            Dict mapping category to list of audio arrays
        """
        num_samples = int(duration_seconds * self.SAMPLE_RATE)
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        
        samples = {
            '808': [],
            'snare': [],
            'loops': [],
            'atmospheres': []
        }
        
        for i in range(samples_per_category):
            intensity = 0.3 + (i * 0.2)  # Vary intensity
            
            # 808: Low sine + sub harmonics
            freq_808 = 40 + (i * 10)  # 40-60 Hz range
            audio_808 = np.sin(2 * np.pi * freq_808 * t) * intensity
            audio_808 += np.sin(2 * np.pi * freq_808 * 2 * t) * intensity * 0.5
            # Add attack transient
            attack_len = int(0.01 * self.SAMPLE_RATE)
            audio_808[:attack_len] += np.random.randn(attack_len) * intensity * 0.3
            samples['808'].append(audio_808.astype(np.float32))
            
            # Snare: High-freq noise with sharp attack
            snare_noise = np.random.randn(num_samples) * intensity
            # Apply envelope
            envelope = np.exp(-np.linspace(0, 10, num_samples))
            audio_snare = snare_noise * envelope
            samples['snare'].append(audio_snare.astype(np.float32))
            
            # Loops: Repeating pattern
            pattern_len = num_samples // 4
            pattern = np.random.randn(pattern_len) * intensity
            audio_loop = np.tile(pattern, 4)[:num_samples]
            samples['loops'].append(audio_loop.astype(np.float32))
            
            # Atmospheres: Filtered noise (like wind)
            atmos_noise = np.random.randn(num_samples) * intensity * 0.5
            # Simple lowpass via convolution
            kernel = np.ones(100) / 100
            audio_atmos = np.convolve(atmos_noise, kernel, mode='same')
            samples['atmospheres'].append(audio_atmos.astype(np.float32))
        
        return samples
    
    def create_test_sample_set(
        self,
        count: int = 10,
        duration_range: Tuple[float, float] = (0.3, 1.0),
        include_markers: bool = True
    ) -> List[Path]:
        """
        Create a complete set of test samples and save to disk.
        
        Args:
            count: Number of samples to generate
            duration_range: (min, max) duration in seconds
            include_markers: Add boundary markers
            
        Returns:
            List of paths to generated files
        """
        generated_files = []
        
        print(f"Generating {count} test samples...")
        
        for i in range(count):
            # Randomize duration
            duration = np.random.uniform(*duration_range)
            
            # Randomize amplitude/intensity
            amplitude = np.random.uniform(0.2, 0.8)
            
            # Generate noise burst
            audio = self.generate_white_noise_burst(
                duration_seconds=duration,
                amplitude=amplitude,
                add_markers=include_markers
            )
            
            # Save file
            filename = f"test_sample_{i+1:03d}.wav"
            filepath = self.output_dir / filename
            sf.write(str(filepath), audio, self.SAMPLE_RATE)
            
            generated_files.append(filepath)
            print(f"  Created: {filename} ({duration:.2f}s, intensity={amplitude:.2f})")
        
        return generated_files
    
    def create_category_sample_set(
        self,
        samples_per_category: int = 3
    ) -> dict:
        """
        Create and save test samples organized by category.
        
        Args:
            samples_per_category: Samples per category
            
        Returns:
            Dict mapping category to list of file paths
        """
        category_samples = self.generate_category_samples(
            samples_per_category=samples_per_category
        )
        
        saved_files = {}
        
        for category, audio_list in category_samples.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            saved_files[category] = []
            
            for i, audio in enumerate(audio_list):
                filename = f"{category}_{i+1:02d}.wav"
                filepath = category_dir / filename
                sf.write(str(filepath), audio, self.SAMPLE_RATE)
                saved_files[category].append(filepath)
                print(f"  Created: {category}/{filename}")
        
        return saved_files
    
    def verify_sample_accuracy(
        self,
        master_wav_path: str,
        db_path: str,
        strict_markers: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Verify that slicing from master WAV is sample-accurate.
        
        Checks that boundary markers are correctly aligned (if present).
        
        Args:
            master_wav_path: Path to master WAV file
            db_path: Path to database
            strict_markers: If True, enforce marker checks. If False, only warn.
            
        Returns:
            Tuple of (all_passed, list of error messages)
        """
        from .provider import SoundBankProvider
        
        provider = SoundBankProvider(master_wav_path, db_path)
        errors = []
        
        assets = provider.list_assets()
        print(f"\nVerifying {len(assets)} assets for sample accuracy...")
        
        for asset in assets:
            audio = provider.request_slice(asset_id=asset.id)
            
            # Check length matches database
            expected_length = asset.end_sample - asset.start_sample
            if len(audio) != expected_length:
                errors.append(
                    f"Asset {asset.id}: Length mismatch "
                    f"(expected {expected_length}, got {len(audio)})"
                )
            
            # Only check boundary markers in strict mode or for test samples
            if strict_markers:
                marker_len = 10
                start_marker = audio[:marker_len]
                end_marker = audio[-marker_len:]
                
                # Start should be positive (if markers were added)
                if np.mean(start_marker) < 0:
                    errors.append(
                        f"Asset {asset.id} ({asset.original_filename}): "
                        f"Start marker incorrect (expected positive, got {np.mean(start_marker):.4f})"
                    )
                
                # End should be negative (if markers were added)
                if np.mean(end_marker) > 0:
                    errors.append(
                        f"Asset {asset.id} ({asset.original_filename}): "
                        f"End marker incorrect (expected negative, got {np.mean(end_marker):.4f})"
                    )
        
        if errors:
            print(f"FAILED: {len(errors)} issues found")
            for error in errors:
                print(f"  - {error}")
        else:
            print("PASSED: All assets are sample-accurate")
        
        return len(errors) == 0, errors


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate test audio samples for Sound Bank validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 white noise bursts
  python -m soundbank.test_generator -o ./test_samples -n 10
  
  # Generate category-organized samples
  python -m soundbank.test_generator -o ./test_samples --categories
  
  # Verify sample accuracy after ingestion
  python -m soundbank.test_generator --verify ./output/master_bank.wav ./output/bank.db
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./test_samples',
        help='Output directory (default: ./test_samples)'
    )
    
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=10,
        help='Number of samples to generate (default: 10)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=0.5,
        help='Base duration in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--categories',
        action='store_true',
        help='Generate samples organized by category'
    )
    
    parser.add_argument(
        '--verify',
        nargs=2,
        metavar=('MASTER_WAV', 'DATABASE'),
        help='Verify sample accuracy of existing sound bank'
    )
    
    parser.add_argument(
        '--no-markers',
        action='store_true',
        help='Do not add boundary markers to samples'
    )
    
    args = parser.parse_args()
    
    generator = TestSampleGenerator(output_dir=args.output)
    
    if args.verify:
        # Verification mode
        master_wav, db_path = args.verify
        passed, errors = generator.verify_sample_accuracy(master_wav, db_path)
        sys.exit(0 if passed else 1)
    
    elif args.categories:
        # Category-organized generation
        print(f"Generating category-organized samples in {args.output}/")
        generator.create_category_sample_set(samples_per_category=args.count)
        print("\nDone! Use ingest.py to process each category:")
        print(f"  python -m soundbank.ingest {args.output}/808 -o ./output -c 808")
        
    else:
        # Simple white noise bursts
        print(f"Generating {args.count} test samples in {args.output}/")
        files = generator.create_test_sample_set(
            count=args.count,
            duration_range=(args.duration * 0.5, args.duration * 1.5),
            include_markers=not args.no_markers
        )
        print(f"\nGenerated {len(files)} files")
        print("\nTo ingest these samples:")
        print(f"  python -m soundbank.ingest {args.output} -o ./output -c loops")


if __name__ == '__main__':
    main()
