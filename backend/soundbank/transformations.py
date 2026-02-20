"""
Audio Transformation Modules
============================
Contains spectral notch filter, bit-crushing, and time-stretching transformations.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings


class Transformations:
    """
    Audio transformation utilities for the Sound Bank Builder.
    
    Primary transformation: Spectral notch filter (1kHz-3kHz "vocal pocket")
    Additional transformations: Bit-crushing, Time-stretching
    """
    
    # Standard sample rate for all processing
    TARGET_SAMPLE_RATE = 44100
    
    # Vocal pocket frequency range (Hz)
    VOCAL_POCKET_LOW = 1000
    VOCAL_POCKET_HIGH = 3000
    
    @staticmethod
    def apply_spectral_notch(
        audio: np.ndarray,
        sample_rate: int = 44100,
        low_freq: int = 1000,
        high_freq: int = 3000,
        attenuation_db: float = -12.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply spectral notch filter to create the "vocal pocket".
        
        This is a DESTRUCTIVE process that permanently EQ cuts the
        1kHz-3kHz range to ensure beats sit behind vocals.
        
        Args:
            audio: Input audio array (mono or stereo)
            sample_rate: Sample rate of the audio
            low_freq: Low frequency cutoff (default 1kHz)
            high_freq: High frequency cutoff (default 3kHz)
            attenuation_db: Amount of attenuation in dB (default -12dB)
            
        Returns:
            Tuple of (processed_audio, spectral_hole_depth)
        """
        # Convert attenuation to linear gain
        gain = 10 ** (attenuation_db / 20)
        
        # Design bandstop (notch) filter
        # Butterworth filter for smooth response
        nyquist = sample_rate / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        
        # Clamp normalized frequencies to valid range
        low_normalized = max(0.001, min(0.999, low_normalized))
        high_normalized = max(0.001, min(0.999, high_normalized))
        
        if low_normalized >= high_normalized:
            warnings.warn("Invalid frequency range for notch filter")
            return audio.copy(), 0.0
        
        # Design a parametric EQ-style notch using cascaded biquads
        # First, create a bandpass to isolate the vocal range
        b_bp, a_bp = signal.butter(2, [low_normalized, high_normalized], btype='band')
        
        # Apply the effect: subtract attenuated bandpass from original
        if audio.ndim == 1:
            # Mono
            bandpass_content = signal.filtfilt(b_bp, a_bp, audio)
            processed = audio - bandpass_content * (1 - gain)
        else:
            # Stereo
            processed = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                bandpass_content = signal.filtfilt(b_bp, a_bp, audio[:, ch])
                processed[:, ch] = audio[:, ch] - bandpass_content * (1 - gain)
        
        # Calculate spectral hole depth (how much energy was removed)
        original_band_energy = np.sqrt(np.mean(bandpass_content ** 2))
        spectral_hole = abs(attenuation_db) if original_band_energy > 0.001 else 0.0
        
        return processed, spectral_hole
    
    @staticmethod
    def normalize_peak(
        audio: np.ndarray,
        target_peak: float = 0.95
    ) -> np.ndarray:
        """
        Normalize audio to a target peak level.
        
        Args:
            audio: Input audio array
            target_peak: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio array
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio
    
    @staticmethod
    def normalize_rms(
        audio: np.ndarray,
        target_rms: float = 0.1
    ) -> np.ndarray:
        """
        Normalize audio to a target RMS level.
        
        Args:
            audio: Input audio array
            target_rms: Target RMS level
            
        Returns:
            Normalized audio array
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            gain = target_rms / current_rms
            normalized = audio * gain
            # Prevent clipping
            peak = np.max(np.abs(normalized))
            if peak > 0.99:
                normalized = normalized * (0.99 / peak)
            return normalized
        return audio
    
    @staticmethod
    def calculate_rms_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy of audio signal."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono for analysis
        return float(np.sqrt(np.mean(audio ** 2)))
    
    @staticmethod
    def calculate_mid_range_density(
        audio: np.ndarray,
        sample_rate: int = 44100,
        low_freq: int = 250,
        high_freq: int = 4000
    ) -> float:
        """
        Calculate the energy density in the mid-range frequencies.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            low_freq: Low frequency bound
            high_freq: High frequency bound
            
        Returns:
            Mid-range density as ratio of mid energy to total energy
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Design bandpass filter for mid-range
        nyquist = sample_rate / 2
        low_norm = max(0.001, min(0.999, low_freq / nyquist))
        high_norm = max(0.001, min(0.999, high_freq / nyquist))
        
        if low_norm >= high_norm:
            return 0.0
        
        b, a = signal.butter(2, [low_norm, high_norm], btype='band')
        mid_content = signal.filtfilt(b, a, audio)
        
        total_energy = np.mean(audio ** 2)
        mid_energy = np.mean(mid_content ** 2)
        
        if total_energy > 0:
            return float(mid_energy / total_energy)
        return 0.0
    
    @staticmethod
    def bit_crush(
        audio: np.ndarray,
        bit_depth: int = 8,
        sample_rate_reduction: int = 1
    ) -> np.ndarray:
        """
        Apply bit-crushing effect for lo-fi sound.
        
        Args:
            audio: Input audio array (float, -1 to 1)
            bit_depth: Target bit depth (1-16, default 8)
            sample_rate_reduction: Sample rate reduction factor (default 1 = no reduction)
            
        Returns:
            Bit-crushed audio array
        """
        # Clamp bit depth
        bit_depth = max(1, min(16, bit_depth))
        
        # Calculate quantization levels
        levels = 2 ** bit_depth
        
        # Quantize amplitude
        crushed = np.round(audio * (levels / 2)) / (levels / 2)
        
        # Apply sample rate reduction (sample-and-hold effect)
        if sample_rate_reduction > 1:
            # Sample-and-hold: repeat each Nth sample
            indices = np.arange(0, len(crushed), sample_rate_reduction)
            held_values = crushed[indices]
            crushed = np.repeat(held_values, sample_rate_reduction)[:len(audio)]
        
        return crushed
    
    @staticmethod
    def time_stretch(
        audio: np.ndarray,
        stretch_factor: float = 1.0,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Time-stretch audio without changing pitch using phase vocoder.
        
        Args:
            audio: Input audio array
            stretch_factor: Stretch factor (< 1 = faster, > 1 = slower)
            sample_rate: Sample rate of audio
            
        Returns:
            Time-stretched audio array
        """
        if abs(stretch_factor - 1.0) < 0.01:
            return audio.copy()
        
        # Phase vocoder parameters
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Handle stereo
        if audio.ndim == 2:
            left = Transformations._phase_vocoder_stretch(
                audio[:, 0], stretch_factor, n_fft, hop_length
            )
            right = Transformations._phase_vocoder_stretch(
                audio[:, 1], stretch_factor, n_fft, hop_length
            )
            return np.column_stack([left, right])
        else:
            return Transformations._phase_vocoder_stretch(
                audio, stretch_factor, n_fft, hop_length
            )
    
    @staticmethod
    def _phase_vocoder_stretch(
        audio: np.ndarray,
        stretch_factor: float,
        n_fft: int,
        hop_length: int
    ) -> np.ndarray:
        """
        Phase vocoder implementation for time-stretching.
        """
        # STFT
        num_frames = 1 + (len(audio) - n_fft) // hop_length
        if num_frames < 2:
            return audio.copy()
        
        # Compute STFT
        stft_matrix = np.array([
            np.fft.rfft(audio[i * hop_length:i * hop_length + n_fft] * np.hanning(n_fft))
            for i in range(num_frames)
        ])
        
        # Calculate new number of frames
        new_num_frames = int(num_frames * stretch_factor)
        if new_num_frames < 2:
            return audio.copy()
        
        # Phase accumulator
        phase_acc = np.angle(stft_matrix[0])
        expected_phase_advance = 2 * np.pi * hop_length * np.arange(n_fft // 2 + 1) / n_fft
        
        # Interpolate and reconstruct
        output_length = (new_num_frames - 1) * hop_length + n_fft
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        window = np.hanning(n_fft)
        
        for i in range(new_num_frames):
            # Find source frame position
            src_pos = i / stretch_factor
            src_idx = int(src_pos)
            frac = src_pos - src_idx
            
            if src_idx >= num_frames - 1:
                src_idx = num_frames - 2
                frac = 1.0
            
            # Interpolate magnitude
            mag0 = np.abs(stft_matrix[src_idx])
            mag1 = np.abs(stft_matrix[src_idx + 1])
            mag = mag0 * (1 - frac) + mag1 * frac
            
            # Phase advancement
            if i > 0:
                phase_diff = np.angle(stft_matrix[min(src_idx + 1, num_frames - 1)]) - \
                            np.angle(stft_matrix[src_idx])
                phase_diff = phase_diff - expected_phase_advance
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
                phase_acc += expected_phase_advance + phase_diff
            
            # Synthesize frame
            frame_fft = mag * np.exp(1j * phase_acc)
            frame = np.fft.irfft(frame_fft, n_fft) * window
            
            # Overlap-add
            start = i * hop_length
            output[start:start + n_fft] += frame
            window_sum[start:start + n_fft] += window ** 2
        
        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum
        
        return output
    
    @staticmethod
    def resample(
        audio: np.ndarray,
        original_sr: int,
        target_sr: int = 44100
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio array
            original_sr: Original sample rate
            target_sr: Target sample rate (default 44100)
            
        Returns:
            Resampled audio array
        """
        if original_sr == target_sr:
            return audio.copy()
        
        # Calculate resampling ratio
        ratio = target_sr / original_sr
        new_length = int(len(audio) * ratio)
        
        if audio.ndim == 1:
            return signal.resample(audio, new_length)
        else:
            resampled = np.zeros((new_length, audio.shape[1]))
            for ch in range(audio.shape[1]):
                resampled[:, ch] = signal.resample(audio[:, ch], new_length)
            return resampled


def add_transformation_module(
    name: str,
    transform_func,
    description: str = ""
) -> None:
    """
    Add a custom transformation to the Transformations class.
    
    Usage example:
        def my_custom_effect(audio, intensity=0.5):
            return audio * (1 - intensity) + np.sin(audio * 10) * intensity
            
        add_transformation_module(
            'custom_saturation',
            my_custom_effect,
            'Adds harmonic saturation to audio'
        )
        
        # Then use:
        Transformations.custom_saturation(audio, intensity=0.7)
    
    Args:
        name: Name of the transformation method
        transform_func: The transformation function
        description: Description of the transformation
    """
    transform_func.__doc__ = description or transform_func.__doc__
    setattr(Transformations, name, staticmethod(transform_func))
    print(f"Added transformation: {name}")
