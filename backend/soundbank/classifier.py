"""
Audio Classifier for Sound Bank Assets
======================================
Analyzes audio characteristics and automatically assigns relevant tags.

Detects:
- Instrument types (kick, snare, hi-hat, bass, synth, etc.)
- Drum specifics (808, acoustic, synth kicks)
- Sonic characteristics (punchy, warm, bright, dark, metallic, etc.)
- Frequency focus (sub-bass, bass, mid, treble, etc.)
- Envelope type (fast attack, slow attack, sustained, etc.)
- Energy level (low, medium, high energy)
- Use cases (melodic-friendly, vocal-carrier, ambient, etc.)
"""

import numpy as np
from typing import List, Tuple
import librosa
import soundfile as sf


class AudioClassifier:
    """Analyzes audio and generates descriptive tags."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def classify(self, audio: np.ndarray, sample_rate: int = 44100) -> List[str]:
        """
        Analyze audio and return list of appropriate tags.
        
        Args:
            audio: Audio waveform (mono or will be converted)
            sample_rate: Sample rate of audio
            
        Returns:
            List of tag strings
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sample_rate != self.sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sr)
        
        tags = []
        
        # Analyze different characteristics
        tags.extend(self._classify_percussion(audio))
        tags.extend(self._classify_frequency_content(audio))
        tags.extend(self._classify_envelope(audio))
        tags.extend(self._classify_spectral_character(audio))
        tags.extend(self._classify_energy_level(audio))
        tags.extend(self._classify_song_sections(audio))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags
    
    def _classify_percussion(self, audio: np.ndarray) -> List[str]:
        """Classify percussion/drum characteristics."""
        tags = []
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_rate = len(onset_frames) / (len(audio) / self.sr)
        
        # Very high onset rate = hi-hat or cymbal-like
        if onset_rate > 20:
            tags.append("hi-hat")
            tags.append("metallic")
            tags.append("crispy")
        # High onset rate = percussion or snare
        elif onset_rate > 5:
            tags.append("percussion")
            if self._has_snap(audio):
                tags.append("snare")
        # Lower onset rate could be kick or bass
        elif onset_rate < 2:
            # Check if it's a kick by looking at low freq energy
            if self._is_kick(audio):
                tags.append("kick")
                # Check if it's an 808
                if self._is_808_kick(audio):
                    tags.append("808-kick")
                else:
                    tags.append("acoustic-kick")
            elif self._is_bass(audio):
                tags.append("bass")
        
        # Clap detection (strong snap without sustained tail)
        if self._is_clap(audio):
            tags.append("clap")
        
        return tags
    
    def _classify_frequency_content(self, audio: np.ndarray) -> List[str]:
        """Classify based on frequency distribution."""
        tags = []
        
        # Compute FFT
        magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Energy in different frequency bands
        sub_bass = np.sum(magnitude[(freqs >= 20) & (freqs < 60)])
        bass = np.sum(magnitude[(freqs >= 60) & (freqs < 250)])
        low_mid = np.sum(magnitude[(freqs >= 250) & (freqs < 500)])
        mid = np.sum(magnitude[(freqs >= 500) & (freqs < 2000)])
        high_mid = np.sum(magnitude[(freqs >= 2000) & (freqs < 5000)])
        treble = np.sum(magnitude[(freqs >= 5000) & (freqs < 20000)])
        
        total_energy = sub_bass + bass + low_mid + mid + high_mid + treble
        
        if total_energy == 0:
            return tags
        
        # Normalize
        sub_bass_pct = sub_bass / total_energy
        bass_pct = bass / total_energy
        low_mid_pct = low_mid / total_energy
        mid_pct = mid / total_energy
        high_mid_pct = high_mid / total_energy
        treble_pct = treble / total_energy
        
        # Determine dominant frequency range
        if sub_bass_pct > 0.2:
            tags.append("sub-bass")
        if bass_pct > 0.25:
            tags.append("bass")
        if low_mid_pct > 0.25:
            tags.append("low-mid")
        if mid_pct > 0.25:
            tags.append("mid")
        if high_mid_pct > 0.25:
            tags.append("high-mid")
        if treble_pct > 0.25:
            tags.append("treble")
        
        # Check spectrum spread
        top_freqs = [sub_bass_pct, bass_pct, low_mid_pct, mid_pct, high_mid_pct, treble_pct]
        num_significant = sum(1 for p in top_freqs if p > 0.1)
        
        if num_significant >= 4:
            tags.append("wide-spectrum")
        elif num_significant <= 2:
            tags.append("narrow-band")
        
        # Brightness (treble-to-bass ratio)
        brightness = treble_pct / (bass_pct + 0.001)
        if brightness > 1.5:
            tags.append("bright")
        elif brightness < 0.5:
            tags.append("dark")
        
        return tags
    
    def _classify_envelope(self, audio: np.ndarray) -> List[str]:
        """Classify attack and decay characteristics."""
        tags = []
        
        # Compute envelope using STFT
        S = librosa.magphase(librosa.stft(audio))[0]
        envelope = np.mean(S, axis=0)
        
        # Normalize
        envelope = envelope / (np.max(envelope) + 1e-8)
        
        # Time to peak (attack time)
        peak_idx = np.argmax(envelope)
        attack_time = (peak_idx / len(envelope)) * (len(audio) / self.sr)
        
        # Determine attack type
        if attack_time < 0.01:
            tags.append("fast-attack")
            tags.append("percussive")
        elif attack_time < 0.05:
            tags.append("fast-attack")
        elif attack_time > 0.2:
            tags.append("slow-attack")
            tags.append("sustained")
        
        # Check decay
        if peak_idx < len(envelope):
            decay = envelope[peak_idx:] if peak_idx > 0 else envelope
            if len(decay) > 1:
                decay_rate = np.mean(np.diff(decay))
                if decay_rate < -0.01:
                    tags.append("short-decay")
                elif decay_rate > -0.001:
                    tags.append("long-decay")
        
        # Plucked characteristic
        if 0.01 < attack_time < 0.1 and peak_idx < len(envelope) * 0.3:
            tags.append("plucked")
        
        # Pad-like
        if attack_time > 0.1 and len(envelope) > 100:
            tail_energy = np.sum(envelope[-50:]) / np.sum(envelope)
            if tail_energy > 0.1:
                tags.append("pad-like")
        
        return tags
    
    def _classify_spectral_character(self, audio: np.ndarray) -> List[str]:
        """Classify perceived tonal character."""
        tags = []
        
        # RMS for loudness perception
        rms = np.sqrt(np.mean(audio ** 2))
        
        # MFCC untuk spectral texture
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=12)
        mfcc_variance = np.var(mfcc[1:6])  # Mid-range MFCCs
        
        # Zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spec_cent_mean = np.mean(spec_cent)
        
        # Resonance detection (look for spectral peaks)
        magnitude = np.abs(np.fft.rfft(audio))
        resonance = np.max(magnitude) / (np.mean(magnitude) + 1e-8)
        
        if resonance > 3:
            tags.append("resonant")
        
        # Spectral character
        if zcr_mean > 0.15:
            tags.append("crispy")
            tags.append("bright")
        elif zcr_mean < 0.05:
            tags.append("smooth")
        
        # Filtered detection
        if mfcc_variance > 50:
            tags.append("colored")
        
        # Warmth (lower spectral centroid)
        if spec_cent_mean < 3000:
            tags.append("warm")
        
        # Metallic/ring (high spectral centroid + resonance)
        if spec_cent_mean > 6000 and resonance > 2:
            tags.append("metallic")
        
        # Organic vs digital
        if zcr_mean > 0.1 and resonance < 1.5:
            tags.append("organic")
        else:
            tags.append("digital")
        
        return tags
    
    def _classify_energy_level(self, audio: np.ndarray) -> List[str]:
        """Classify based on energy/density."""
        tags = []
        
        # Spectral flux (rate of spectral change)
        S = np.abs(librosa.stft(audio))
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_mean = np.mean(spectral_flux)
        
        # Onset density
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_density = len(onset_frames) / (len(audio) / self.sr)
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Complexity score
        complexity = (flux_mean * 0.4) + (onset_density * 0.3) + (rms * 0.3)
        
        if complexity > 0.5:
            tags.append("high-energy")
            tags.append("explosive")
        elif complexity > 0.2:
            tags.append("medium-energy")
        else:
            tags.append("low-energy")
            tags.append("minimalist")
        
        # Check if good for melody/vocals
        if rms < 0.3 and flux_mean < 0.3:
            tags.append("melodic-friendly")
        
        # Ambient quality
        if onset_density < 3 and flux_mean < 0.2:
            tags.append("ambient")
            tags.append("texture")
        
        return tags
    
    def _is_kick(self, audio: np.ndarray) -> bool:
        """Detect if audio is a kick drum."""
        # Low frequency energy
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_freq_mask = freqs < 250
        low_energy = np.sum(S[low_freq_mask, :])
        high_energy = np.sum(S[~low_freq_mask, :])
        
        return low_energy > high_energy * 0.5
    
    def _is_808_kick(self, audio: np.ndarray) -> bool:
        """Detect if kick is 808-style (pitched, resonant)."""
        # Check for fundamental pitch and long decay
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        if len(onset_frames) == 0:
            return False
        
        # Get main onset region
        if onset_frames[0] < len(audio) * 0.9:
            duration = len(audio) * 0.7
        else:
            duration = len(audio)
        
        # Check for pitch (high spectral centroid in low freqs)
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_freq_mask = (freqs > 40) & (freqs < 300)
        
        if np.sum(low_freq_mask) > 0:
            spectrum = np.mean(S[low_freq_mask, :], axis=1)
            spectral_cent = np.average(freqs[low_freq_mask], weights=spectrum)
            # 808 kicks typically have fundamental between 40-150Hz
            return 40 < spectral_cent < 150
        
        return False
    
    def _is_bass(self, audio: np.ndarray) -> bool:
        """Detect if audio is a bass sound."""
        # Strong low frequency presence
        magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        bass_energy = np.sum(magnitude[(freqs >= 60) & (freqs < 250)])
        total_energy = np.sum(magnitude)
        
        return (bass_energy / total_energy) > 0.4
    
    def _has_snap(self, audio: np.ndarray) -> bool:
        """Detect if audio has a snappy transient (snare-like)."""
        # Look for high-frequency spike at attack
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        high_freq_mask = freqs > 3000
        
        if np.sum(high_freq_mask) == 0:
            return False
        
        high_energy = S[high_freq_mask, :]
        peak_time = np.argmax(np.sum(high_energy, axis=0))
        
        # Check if peak is early (snap at attack)
        return peak_time < len(high_energy[0]) * 0.3
    
    def _is_clap(self, audio: np.ndarray) -> bool:
        """Detect if audio is a clap."""
        # Multiple transients close together
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        
        if len(onset_frames) < 2:
            return False
        
        # Check spacing between onsets
        onset_diffs = np.diff(onset_frames)
        
        # Claps have close onsets (clap has multiple hits)
        close_onsets = np.sum(onset_diffs < 10)  # < 10 frames apart
        
        return close_onsets >= 1 and len(onset_frames) >= 2
    
    def _classify_song_sections(self, audio: np.ndarray) -> List[str]:
        """Classify suitability for different song sections."""
        tags = []
        
        # Compute metrics
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_density = len(onset_frames) / (len(audio) / self.sr)
        
        S = np.abs(librosa.stft(audio))
        spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_mean = np.mean(spectral_flux)
        
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Envelope analysis
        envelope = np.mean(S, axis=0)
        envelope = envelope / (np.max(envelope) + 1e-8)
        peak_idx = np.argmax(envelope)
        attack_time = (peak_idx / len(envelope)) * (len(audio) / self.sr)
        
        # INTRO: Minimalist, calm, sparse, establishes vibe
        # Characteristics: low-medium energy, slow attack, sparse onsets
        if onset_density < 5 and rms < 0.25 and attack_time > 0.05:
            tags.append("intro")
        
        # VERSE: Focused, supporting vocals, moderate energy, clear
        # Characteristics: tight, not too dense, moderate attack
        if 2 < onset_density < 10 and rms < 0.35 and 0.01 < attack_time < 0.15:
            tags.append("verse")
        
        # PRE-CHORUS: Tension building, rising energy, some movement
        # Characteristics: increasing flux, moderate-high density
        if flux_mean > 0.25 and onset_density > 5 and 0 < attack_time < 0.2:
            tags.append("pre-chorus")
        
        # CHORUS: HIGH energy, punchy, full, memorable
        # Characteristics: high density, fast attack, high RMS
        if onset_density > 8 and rms > 0.25 and attack_time < 0.05:
            tags.append("chorus")
        
        # BRIDGE: Contrasting, different from main pattern
        # Characteristics: unexpected texture - either much sparser or much denser
        if (onset_density < 2 and rms < 0.15) or (onset_density > 15 and flux_mean > 0.4):
            tags.append("bridge")
        
        # BREAKDOWN: Reduced elements, tension drop, minimalist
        # Characteristics: very sparse, low complexity, long decay
        if onset_density < 3 and flux_mean < 0.15 and rms < 0.2:
            tags.append("breakdown")
        
        # BUILD-UP: Rising energy, increasing density, momentum
        # Characteristics: medium-high flux, moderate-high density, progressive
        if 0.2 < flux_mean < 0.6 and 5 < onset_density < 15:
            tags.append("build-up")
        
        # DROP/CLIMAX: Maximum energy, released tension, explosive
        # Characteristics: very high onset density, high RMS, fast attack
        if onset_density > 12 and rms > 0.35 and attack_time < 0.01:
            tags.append("drop")
        
        # PRE-DROP: Tension before drop, anticipatory
        # Characteristics: high flux with lower density (teaser)
        if flux_mean > 0.3 and onset_density > 5 and onset_density < 10:
            tags.append("pre-drop")
        
        # FILL/TRANSITION: Short, connects sections, fills gaps
        # Characteristics: short duration (implicit via sample length check)
        # Note: This is inferred from duration in context during ingestion
        if len(audio) / self.sr < 2 and onset_density > 5:
            tags.append("fill")
        
        # INTERLUDE: Instrumental break, different texture, atmospheric
        # Characteristics: sustained elements, lower onset density, may have spectral depth
        S = np.abs(librosa.stft(audio))
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
        centroid_var = np.var(spectral_centroid)
        
        if onset_density < 8 and centroid_var > 500:
            tags.append("interlude")
        
        # OUTRO: Winding down, decreasing energy, conclusion
        # Characteristics: gradual decay, lower density toward end
        if len(envelope) > 100:
            tail_energy = np.sum(envelope[-50:]) / np.sum(envelope)
            if tail_energy < 0.15 and onset_density < 7:
                tags.append("outro")
        
        return tags


def classify_file(file_path: str) -> List[str]:
    """
    Convenience function to classify an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        List of tag strings
    """
    audio, sr = sf.read(file_path)
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    classifier = AudioClassifier(sr=44100)
    return classifier.classify(audio, sample_rate=sr)
