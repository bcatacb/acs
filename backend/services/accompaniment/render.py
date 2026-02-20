import math
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .analysis import _section_levels
from .style import _pack_profile, _resolve_palette, _style_family

SEMITONE = 2.0 ** (1.0 / 12.0)
def _add_wave(dest: np.ndarray, sample: np.ndarray, start_idx: int) -> None:
    if start_idx >= dest.size:
        return
    end_idx = min(dest.size, start_idx + sample.size)
    n = end_idx - start_idx
    if n <= 0:
        return
    dest[start_idx:end_idx] += sample[:n]

def _kick(sr: int, strength: float) -> np.ndarray:
    length = 0.22
    t = np.linspace(0.0, length, int(sr * length), endpoint=False, dtype=np.float32)
    freq = 140.0 * np.exp(-t * 14.0) + 36.0
    phase = 2.0 * math.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 12.0)
    return (np.sin(phase) * env * (0.8 * strength)).astype(np.float32)

def _snare(sr: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    length = 0.18
    t = np.linspace(0.0, length, int(sr * length), endpoint=False, dtype=np.float32)
    noise = rng.normal(0.0, 1.0, t.size).astype(np.float32)
    tone = np.sin(2.0 * math.pi * 210.0 * t).astype(np.float32)
    env = np.exp(-t * 20.0)
    return ((0.2 * noise + 0.8 * tone) * env * (0.52 * strength)).astype(np.float32)

def _hihat(sr: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    length = 0.045
    t = np.linspace(0.0, length, int(sr * length), endpoint=False, dtype=np.float32)
    tone = np.sin(2.0 * math.pi * 7200.0 * t).astype(np.float32)
    noise = rng.normal(0.0, 1.0, t.size).astype(np.float32)
    env = np.exp(-t * 75.0)
    return ((0.72 * tone + 0.28 * noise) * env * (0.09 * strength)).astype(np.float32)

def _bass(sr: int, hz: float, beat_len: float, strength: float) -> np.ndarray:
    t = np.linspace(0.0, beat_len, int(sr * beat_len), endpoint=False, dtype=np.float32)
    # Cleaner, more tonal bass: reduce grit so it stays musically glued.
    sub = np.sin(2.0 * math.pi * hz * t)
    body = 0.22 * np.sin(2.0 * math.pi * hz * 2.0 * t + 0.08)
    env = np.minimum(1.0, t * 24.0) * np.exp(-t * 2.9)
    wave = (sub + body) * env
    wave = _one_pole_lowpass(wave.astype(np.float32), sr, 280.0)
    wave = np.tanh(wave * 1.12)
    return (wave * (0.30 * strength)).astype(np.float32)

def _pad(sr: int, hz: float, sec_len: float, strength: float) -> np.ndarray:
    t = np.linspace(0.0, sec_len, int(sr * sec_len), endpoint=False, dtype=np.float32)
    triad = (
        0.48 * np.sin(2.0 * math.pi * hz * t)
        + 0.33 * np.sin(2.0 * math.pi * (hz * 1.25) * t + 0.2)
        + 0.25 * np.sin(2.0 * math.pi * (hz * 1.5) * t - 0.1)
    )
    lfo = 0.8 + 0.2 * np.sin(2.0 * math.pi * 0.18 * t)
    env = np.minimum(1.0, t * 2.8) * np.exp(-t * 0.55)
    return (triad * lfo * env * (0.12 * strength)).astype(np.float32)

def _lead(sr: int, hz: float, sec_len: float, strength: float) -> np.ndarray:
    t = np.linspace(0.0, sec_len, int(sr * sec_len), endpoint=False, dtype=np.float32)
    saw = 2.0 * ((hz * t) - np.floor(0.5 + hz * t))
    harm = 0.35 * np.sin(2.0 * math.pi * hz * 2.0 * t + 0.4)
    env = np.minimum(1.0, t * 40.0) * np.exp(-t * 7.5)
    return ((0.65 * saw + harm) * env * (0.1 * strength)).astype(np.float32)

def _voice_waveform(t: np.ndarray, hz: float, voice: str) -> np.ndarray:
    if voice == "rhodes":
        return (
            0.62 * np.sin(2.0 * math.pi * hz * t)
            + 0.28 * np.sin(2.0 * math.pi * hz * 2.0 * t + 0.22)
            + 0.1 * np.sin(2.0 * math.pi * hz * 3.0 * t - 0.11)
        ).astype(np.float32)
    if voice == "saw_pluck":
        saw = 2.0 * ((hz * t) - np.floor(0.5 + hz * t))
        return (0.8 * saw + 0.2 * np.sin(2.0 * math.pi * hz * 2.0 * t)).astype(np.float32)
    if voice == "fm_bell":
        mod = np.sin(2.0 * math.pi * hz * 2.0 * t) * 2.2
        return np.sin(2.0 * math.pi * hz * t + mod).astype(np.float32)
    if voice == "warm_pad":
        return (
            0.55 * np.sin(2.0 * math.pi * hz * t)
            + 0.3 * np.sin(2.0 * math.pi * hz * 1.5 * t + 0.1)
            + 0.2 * np.sin(2.0 * math.pi * hz * 2.0 * t - 0.15)
        ).astype(np.float32)
    if voice == "dark_pad":
        return (
            0.68 * np.sin(2.0 * math.pi * hz * t)
            + 0.24 * np.sin(2.0 * math.pi * hz * 0.5 * t)
            + 0.14 * np.sin(2.0 * math.pi * hz * 2.0 * t + 0.3)
        ).astype(np.float32)
    if voice == "analog_keys":
        sq = np.sign(np.sin(2.0 * math.pi * hz * t))
        return (0.65 * sq + 0.35 * np.sin(2.0 * math.pi * hz * t)).astype(np.float32)
    if voice == "soft_sine":
        return (
            0.8 * np.sin(2.0 * math.pi * hz * t)
            + 0.2 * np.sin(2.0 * math.pi * hz * 2.0 * t - 0.2)
        ).astype(np.float32)
    if voice == "string_ensemble":
        vibrato = 0.7 * np.sin(2.0 * math.pi * 5.0 * t)
        return (
            0.52 * np.sin(2.0 * math.pi * hz * t + vibrato * 0.01)
            + 0.28 * np.sin(2.0 * math.pi * hz * 2.0 * t)
            + 0.2 * np.sin(2.0 * math.pi * hz * 3.0 * t)
        ).astype(np.float32)
    if voice == "piano_keys":
        return (
            0.68 * np.sin(2.0 * math.pi * hz * t)
            + 0.22 * np.sin(2.0 * math.pi * hz * 2.0 * t + 0.15)
            + 0.1 * np.sin(2.0 * math.pi * hz * 4.0 * t)
        ).astype(np.float32)
    if voice == "flute_lead":
        breath = 0.04 * np.sin(2.0 * math.pi * 7.0 * t)
        return (
            0.88 * np.sin(2.0 * math.pi * hz * t + breath)
            + 0.12 * np.sin(2.0 * math.pi * hz * 2.0 * t)
        ).astype(np.float32)
    return np.sin(2.0 * math.pi * hz * t).astype(np.float32)

def _keys_voice(sr: int, hz: float, sec_len: float, strength: float, voice: str, long_env: bool) -> np.ndarray:
    t = np.linspace(0.0, sec_len, int(sr * sec_len), endpoint=False, dtype=np.float32)
    base = _voice_waveform(t, hz, voice)
    if long_env:
        env = _adsr_envelope(t, attack=0.08, decay=0.35, sustain=0.62, release=0.45)
    else:
        env = _adsr_envelope(t, attack=0.01, decay=0.12, sustain=0.3, release=0.12)
    shaped = base * env
    cutoff = 2600.0 if long_env else 4200.0
    if voice in ("rhodes", "warm_pad", "dark_pad", "string_ensemble"):
        shaped = _one_pole_lowpass(shaped, sr, cutoff)
    return (shaped * strength).astype(np.float32)

def _adsr_envelope(t: np.ndarray, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    total = float(max(1e-4, t[-1] if t.size > 1 else attack + decay + release))
    env = np.zeros_like(t, dtype=np.float32)
    a_end = attack
    d_end = a_end + decay
    r_start = max(d_end, total - release)

    # Attack
    mask_a = t <= a_end
    if np.any(mask_a):
        env[mask_a] = np.clip(t[mask_a] / max(1e-4, attack), 0.0, 1.0)

    # Decay to sustain
    mask_d = (t > a_end) & (t <= d_end)
    if np.any(mask_d):
        td = (t[mask_d] - a_end) / max(1e-4, decay)
        env[mask_d] = 1.0 + (sustain - 1.0) * np.clip(td, 0.0, 1.0)

    # Sustain
    mask_s = (t > d_end) & (t < r_start)
    if np.any(mask_s):
        env[mask_s] = sustain

    # Release
    mask_r = t >= r_start
    if np.any(mask_r):
        tr = (t[mask_r] - r_start) / max(1e-4, release)
        env[mask_r] = sustain * (1.0 - np.clip(tr, 0.0, 1.0))

    return env.astype(np.float32)

def _one_pole_lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff = max(30.0, min(float(sr) * 0.45, float(cutoff_hz)))
    alpha = math.exp(-2.0 * math.pi * cutoff / float(sr))
    y = np.zeros_like(x, dtype=np.float32)
    prev = 0.0
    for i in range(x.size):
        prev = (1.0 - alpha) * float(x[i]) + alpha * prev
        y[i] = prev
    return y

def _simple_reverb(x: np.ndarray, sr: int, mix: float = 0.15) -> np.ndarray:
    if x.size == 0 or mix <= 0.0:
        return x
    # Light Schroeder-like tail using short delays.
    delays = [int(sr * 0.029), int(sr * 0.037), int(sr * 0.041)]
    gains = [0.35, 0.28, 0.22]
    wet = np.zeros_like(x, dtype=np.float32)
    for d, g in zip(delays, gains):
        if d < x.size:
            wet[d:] += x[:-d] * g
    return (x * (1.0 - mix) + wet * mix).astype(np.float32)

def _one_pole_highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    return (x - _one_pole_lowpass(x, sr, cutoff_hz)).astype(np.float32)

def _soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * max(0.1, float(drive))).astype(np.float32)

def _bus_compress(x: np.ndarray, amount: float = 0.25) -> np.ndarray:
    if x.size == 0 or amount <= 0.0:
        return x.astype(np.float32, copy=False)
    env = np.convolve(np.abs(x), np.ones(1024, dtype=np.float32) / 1024.0, mode="same")
    gain = 1.0 / (1.0 + amount * env * 3.0)
    return (x * gain).astype(np.float32)

def _minor_chord_freqs(root_hz: float, degree: int, inversion: int = 0) -> List[float]:
    # Natural minor scale degrees in semitones from root.
    degree_offsets = [0, 2, 3, 5, 7, 8, 10]
    triad_intervals = [0, 3, 7]  # keep harmony cleaner and less crowded
    semitone_shift = degree_offsets[degree % len(degree_offsets)]
    chord_root = root_hz * (SEMITONE ** semitone_shift) * 2.0
    notes = [chord_root * (SEMITONE ** i) for i in triad_intervals]
    inv = inversion % len(notes)
    for i in range(inv):
        notes[i] *= 2.0
    return notes

def _section_profile(bar: int, bars: int) -> Dict[str, float]:
    if bars <= 8:
        return {"name": "main", "energy": 1.0, "music": 1.0, "drums": 1.0}
    q = bar / float(max(1, bars - 1))
    if q < 0.12:
        return {"name": "intro", "energy": 0.72, "music": 0.85, "drums": 0.78}
    if q < 0.52:
        return {"name": "verse", "energy": 0.96, "music": 0.94, "drums": 1.0}
    if q < 0.82:
        return {"name": "hook", "energy": 1.18, "music": 1.15, "drums": 1.12}
    return {"name": "outro", "energy": 0.84, "music": 0.9, "drums": 0.86}

def _shaker(sr: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    length = 0.045
    t = np.linspace(0.0, length, int(sr * length), endpoint=False, dtype=np.float32)
    noise = rng.normal(0.0, 1.0, t.size).astype(np.float32)
    env = np.exp(-t * 52.0)
    hp = np.concatenate(([0.0], np.diff(noise))).astype(np.float32)
    return (hp * env * (0.13 * strength)).astype(np.float32)

def _build_vocal_mix(beat: np.ndarray, vocal: np.ndarray) -> np.ndarray:
    n = beat.size
    out = beat.copy()
    vv = np.zeros(n, dtype=np.float32)
    m = min(n, vocal.size)
    vv[:m] = vocal[:m]
    # Light sidechain feel: duck beat where vocals are present.
    env = np.convolve(np.abs(vv), np.ones(2048, dtype=np.float32) / 2048.0, mode="same")
    duck = 1.0 - np.clip(env * 0.35, 0.0, 0.35)
    out *= duck.astype(np.float32)
    out += vv * 0.78
    return out.astype(np.float32)

def _write_wav_stereo(path: Path, mono: np.ndarray, sr: int) -> None:
    mono = _bus_compress(mono, amount=0.1)
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak * 0.92
    mono = _soft_clip(mono, drive=0.92)

    # Avoid phase artifacts: export stable dual-mono.
    left = mono.astype(np.float32, copy=False)
    right = mono.astype(np.float32, copy=False)
    stereo = np.stack([left, right], axis=1)
    int16_audio = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())

def _compute_repetition_score(density: np.ndarray) -> float:
    """
    Returns 0..1 where higher means more repetitive bar-to-bar structure.
    """
    if density.size < 8:
        return 1.0

    # Compare first half against second half pattern.
    half = density.size // 2
    a = density[:half]
    b = density[half : half + half]
    if a.size != b.size or a.size == 0:
        return 1.0

    mean_diff = float(np.mean(np.abs(a - b)))
    # Low diff => repetitive, high diff => varied.
    rep = 1.0 - min(1.0, mean_diff / 0.4)
    return max(0.0, min(1.0, rep))

def _render_mix(
    bars: int,
    bar_len: float,
    sec_per_beat: float,
    synth_sr: int,
    onsets: np.ndarray,
    density: np.ndarray,
    root: float,
    seed: int,
    genre: str,
    instrument_pack: str,
    drum_paths: Optional[Dict[str, List[Path]]] = None,
    loop_layer: Optional[np.ndarray] = None,
    flow_grid: Optional[List[Dict]] = None,
    section_labels: Optional[List[str]] = None,
    progression: Optional[List[int]] = None,
    midi_patterns: Optional[Dict[str, Dict]] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_duration = bars * bar_len
    beat_len_samples = int(total_duration * synth_sr)
    drums = np.zeros(beat_len_samples, dtype=np.float32)
    bass_layer = np.zeros(beat_len_samples, dtype=np.float32)
    music = np.zeros(beat_len_samples, dtype=np.float32)
    palette = _resolve_palette(genre, instrument_pack)
    profile = _pack_profile(instrument_pack)
    style_family = _style_family(genre)
    trap_mode = style_family == "trap"
    southern_mode = style_family == "southern"
    melodic_mode = style_family == "melodic"
    eastcoast_mode = style_family == "east_coast"
    boombap_mode = style_family == "boom_bap"
    lofi_mode = style_family == "lo_fi"
    westcoast_mode = style_family == "west_coast"
    allow_key_stabs = not (trap_mode or southern_mode or melodic_mode or eastcoast_mode or boombap_mode or lofi_mode or westcoast_mode)
    # Safety-first arranging: prioritize legibility over density/complexity.
    safe_mode = not trap_mode and not southern_mode and not melodic_mode and not eastcoast_mode and not boombap_mode and not lofi_mode and not westcoast_mode
    if safe_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.9,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.82,
            "hat_density": min(0.72, float(profile.get("hat_density", 0.72))),
            "shaker_density": 0.0,
            "stab_density": min(0.62, float(profile.get("stab_density", 0.62))),
            "arp_density": 0.0,
            "lead_density": min(0.36, float(profile.get("lead_density", 0.36))),
            "music_level": float(profile.get("music_level", 1.0)) * 1.22,
        }
    elif trap_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.74,
            "hat_density": min(0.62, float(profile.get("hat_density", 0.62))),
            "shaker_density": 0.0,
            "stab_density": min(0.44, float(profile.get("stab_density", 0.44))),
            "arp_density": 0.0,
            "lead_density": min(0.14, float(profile.get("lead_density", 0.14))),
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.58,
            "music_level": float(profile.get("music_level", 1.0)) * 1.62,
        }
    elif southern_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.84,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.7,
            "music_level": float(profile.get("music_level", 1.0)) * 1.46,
            "hat_density": min(0.58, float(profile.get("hat_density", 0.58))),
            "shaker_density": 0.1,
            "stab_density": 0.12,
            "arp_density": 0.0,
            "lead_density": 0.08,
            "swing": 0.04,
        }
    elif melodic_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.8,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.62,
            "music_level": float(profile.get("music_level", 1.0)) * 1.58,
            "hat_density": min(0.52, float(profile.get("hat_density", 0.52))),
            "shaker_density": 0.04,
            "stab_density": 0.0,
            "arp_density": 0.18,
            "lead_density": 0.26,
            "swing": 0.04,
        }
    elif eastcoast_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.96,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.62,
            "music_level": float(profile.get("music_level", 1.0)) * 1.34,
            "hat_density": min(0.38, float(profile.get("hat_density", 0.38))),
            "shaker_density": 0.0,
            "stab_density": 0.0,
            "arp_density": 0.0,
            "lead_density": 0.0,
            "swing": 0.08,
        }
    elif boombap_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.82,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.58,
            "music_level": float(profile.get("music_level", 1.0)) * 1.52,
            "hat_density": min(0.34, float(profile.get("hat_density", 0.34))),
            "shaker_density": 0.0,
            "stab_density": min(0.28, float(profile.get("stab_density", 0.28))),
            "arp_density": 0.0,
            "lead_density": 0.0,
            "swing": 0.07,
        }
    elif lofi_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.68,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.46,
            "music_level": float(profile.get("music_level", 1.0)) * 1.62,
            "hat_density": min(0.42, float(profile.get("hat_density", 0.42))),
            "shaker_density": 0.0,
            "stab_density": 0.0,
            "arp_density": 0.0,
            "lead_density": 0.0,
            "swing": 0.09,
        }
    elif westcoast_mode:
        profile = {
            **profile,
            "drum_level": float(profile.get("drum_level", 1.0)) * 0.86,
            "bass_level": float(profile.get("bass_level", 1.0)) * 0.72,
            "music_level": float(profile.get("music_level", 1.0)) * 1.44,
            "hat_density": min(0.36, float(profile.get("hat_density", 0.36))),
            "shaker_density": 0.08,
            "stab_density": 0.0,
            "arp_density": 0.0,
            "lead_density": 0.0,
            "swing": 0.06,
        }
    # Restore intentional drum tone for rap styles by using consistent synth drums.
    use_drum_samples = not (trap_mode or southern_mode or melodic_mode or eastcoast_mode or boombap_mode or lofi_mode or westcoast_mode)
    if progression:
        progression_use = progression
    elif trap_mode:
        progression_use = [0, 5, 0, 4]
    elif southern_mode:
        progression_use = [0, 5, 4, 3]
    elif melodic_mode:
        progression_use = [0, 4, 5, 3]
    elif eastcoast_mode:
        progression_use = [0, 3, 5, 4]
    elif boombap_mode:
        progression_use = [0, 3, 5, 4]
    elif lofi_mode:
        progression_use = [0, 3, 5, 4]
    elif westcoast_mode:
        progression_use = [0, 4, 5, 3]
    elif (instrument_pack or "auto").lower() == "dark":
        progression_use = [0, 5, 0, 4]
    elif (instrument_pack or "auto").lower() == "warm":
        progression_use = [0, 3, 5, 4]
    elif (instrument_pack or "auto").lower() == "orchestral":
        progression_use = [0, 4, 5, 3]
    else:
        progression_use = [0, 5, 3, 4]

    scale = np.array([1.0, 1.122, 1.335, 1.498], dtype=np.float32)  # root, 2, 4, 5-ish
    phrase_len = 8
    kick_motifs = [
        [0.0, 1.5, 2.0, 2.75],
        [0.0, 1.75, 2.5, 3.25],
        [0.0, 1.0, 2.0, 3.0],
    ]
    hat_motifs = [
        [i * 0.5 for i in range(8)],
        [i * 0.25 for i in range(16)],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75],
    ]
    bass_motifs = [
        [0.0, 1.5, 2.0, 3.0],
        [0.0, 2.0, 2.75],
        [0.0, 1.0, 2.0, 2.5, 3.25],
    ]
    motif_seed = int(rng.integers(0, 3))
    bar_debug: List[Dict] = []
    intro_guard_bars = 2

    for bar in range(bars):
        bar_start = bar * bar_len
        section_name = (
            section_labels[bar]
            if section_labels is not None and bar < len(section_labels)
            else _section_profile(bar, bars)["name"]
        )
        section = _section_levels(section_name)
        in_hook = section_name == "hook"
        in_intro_outro = section_name in ("intro", "outro", "break")
        phrase_bar = bar % phrase_len
        is_turnaround = phrase_bar in (3, 7)
        next_label = section_name
        if section_labels is not None and (bar + 1) < len(section_labels):
            next_label = section_labels[bar + 1]
        pre_hook = (not in_hook) and next_label == "hook"
        phrase_end = phrase_bar == (phrase_len - 1)
        intro_guard = bar < intro_guard_bars
        intensity = (0.65 + 0.55 * float(density[bar])) * section["energy"]

        motif_idx = (motif_seed + (bar // 4)) % len(kick_motifs)
        if trap_mode:
            kick_steps = [0.0, 1.75, 2.5] if not in_hook else [0.0, 1.5, 2.5, 3.25]
        elif southern_mode:
            kick_steps = [0.0, 1.5, 2.5] if not in_hook else [0.0, 1.5, 2.5, 3.25]
        elif melodic_mode:
            kick_steps = [0.0, 2.0] if not in_hook else [0.0, 1.5, 2.0, 3.0]
        elif eastcoast_mode:
            kick_steps = [0.0, 2.0] if not in_hook else [0.0, 1.5, 2.0]
        elif boombap_mode:
            kick_steps = [0.0, 2.0] if not in_hook else [0.0, 1.5, 2.0]
        elif lofi_mode:
            kick_steps = [0.0, 2.0]
        elif westcoast_mode:
            kick_steps = [0.0, 2.25] if not in_hook else [0.0, 1.5, 2.25]
        else:
            kick_steps = [0.0, 2.5] if safe_mode else kick_motifs[motif_idx]
            if in_hook:
                kick_steps = [0.0, 2.5, 3.25] if safe_mode else kick_steps
        if in_intro_outro:
            kick_steps = kick_steps[: max(2, len(kick_steps) - 1)]
        elif in_hook and (is_turnaround or phrase_end):
            kick_steps = sorted(set(kick_steps + [3.5]))
        kick_times = [bar_start + s * sec_per_beat for s in kick_steps]

        if trap_mode:
            snare_times = [bar_start + 2.0 * sec_per_beat]
        elif southern_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        elif melodic_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        elif eastcoast_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        elif boombap_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        elif lofi_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        elif westcoast_mode:
            snare_times = [bar_start + 1.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        else:
            snare_times = [bar_start + (3.0 if safe_mode else 1.0) * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        if safe_mode and not trap_mode:
            snare_times = [bar_start + 3.0 * sec_per_beat]
        if (not safe_mode) and in_hook and rng.random() < 0.3:
            snare_times.append(bar_start + 3.5 * sec_per_beat)

        hat_idx = (motif_seed + (bar % 3)) % len(hat_motifs)
        hat_steps = hat_motifs[hat_idx]
        if trap_mode:
            hat_steps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            if in_hook and (pre_hook or phrase_end):
                hat_steps.extend([3.5, 3.625, 3.75, 3.875])  # trap-style roll
        elif southern_mode:
            hat_steps = [i * 0.25 for i in range(16)]
            if in_hook or pre_hook:
                hat_steps.extend([3.5, 3.625, 3.75, 3.875])
            if in_intro_outro:
                hat_steps = [0.0, 1.0, 2.0, 3.0]
        elif melodic_mode:
            hat_steps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            if in_hook or pre_hook:
                hat_steps.extend([1.75, 3.5, 3.75])
            if in_intro_outro:
                hat_steps = [0.0, 1.0, 2.0, 3.0]
        elif eastcoast_mode:
            # Gritty swung hats, restrained movement.
            swing = profile.get("swing", 0.08)
            hat_steps = [0.0, 0.5 + swing, 1.0, 1.5 + swing, 2.0, 2.5 + swing, 3.0, 3.5 + swing]
            if in_intro_outro:
                hat_steps = [0.0, 1.0, 2.0, 3.0]
        elif boombap_mode:
            # Sparse swung eighth hats for head-nod groove.
            swing = profile.get("swing", 0.07)
            hat_steps = [0.0, 0.5 + swing, 1.0, 1.5 + swing, 2.0, 2.5 + swing, 3.0, 3.5 + swing]
            if in_intro_outro:
                hat_steps = [0.0, 1.0, 2.0, 3.0]
        elif lofi_mode:
            # Lo-fi triplet-ish soft hats with swing.
            swing = profile.get("swing", 0.09)
            hat_steps = [0.0, (1.0/3.0) + swing * 0.3, (2.0/3.0), 1.0, 1.0 + (1.0/3.0), 1.0 + (2.0/3.0), 2.0, 2.0 + (2.0/3.0), 3.0]
            if in_intro_outro:
                hat_steps = [0.0, 1.0, 2.0, 3.0]
        elif westcoast_mode:
            swing = profile.get("swing", 0.06)
            hat_steps = [0.0, 0.5 + swing, 1.5 + swing, 2.0, 2.5 + swing, 3.5 + swing]
        elif safe_mode:
            hat_steps = [] if not in_hook else [0.5, 2.5]
            if phrase_bar in (3, 7):
                hat_steps = []
        if in_intro_outro:
            hat_steps = [] if safe_mode else hat_motifs[0]
        hat_times = np.array([bar_start + hs * sec_per_beat for hs in hat_steps], dtype=np.float32)

        bar_onsets = onsets[(onsets >= bar_start) & (onsets < bar_start + bar_len)]

        for t in kick_times:
            s = _kick(synth_sr, intensity) if not use_drum_samples else _choose_sample(
                (drum_paths or {}).get("kicks", []),
                rng,
                synth_sr,
                _kick(synth_sr, intensity),
            )
            # Slight deterministic accent pattern for more intentional groove.
            local_beat = ((t - bar_start) / sec_per_beat) % 4.0
            if local_beat < 0.2:
                s = (s * 1.08).astype(np.float32)
            elif 1.4 < local_beat < 1.9:
                s = (s * 0.94).astype(np.float32)
            _add_wave(drums, s, int(t * synth_sr))

        for t in snare_times:
            s = _snare(synth_sr, intensity, rng) if not use_drum_samples else _choose_sample(
                (drum_paths or {}).get("claps", []),
                rng,
                synth_sr,
                _snare(synth_sr, intensity, rng),
            )
            s = (s * (1.02 if in_hook else 0.96)).astype(np.float32)
            _add_wave(drums, s, int(t * synth_sr))

        # Add restrained ghost notes for sophistication on rap styles.
        if eastcoast_mode:
            ghost_steps = [1.75, 3.75] if not in_hook else [1.75, 2.75, 3.75]
            for gs in ghost_steps:
                g = (_snare(synth_sr, intensity * 0.42, rng) * 0.28).astype(np.float32)
                _add_wave(drums, g, int((bar_start + gs * sec_per_beat) * synth_sr))
        elif boombap_mode:
            ghost_steps = [1.75, 3.75] if not in_hook else [1.75, 2.75, 3.75]
            for gs in ghost_steps:
                g = (_snare(synth_sr, intensity * 0.45, rng) * 0.32).astype(np.float32)
                _add_wave(drums, g, int((bar_start + gs * sec_per_beat) * synth_sr))
        elif westcoast_mode:
            for gs in [1.75, 3.75]:
                g = (_snare(synth_sr, intensity * 0.38, rng) * 0.24).astype(np.float32)
                _add_wave(drums, g, int((bar_start + gs * sec_per_beat) * synth_sr))
        elif southern_mode and (pre_hook or phrase_end):
            for gs in [3.5, 3.75]:
                g = (_snare(synth_sr, intensity * 0.44, rng) * 0.28).astype(np.float32)
                _add_wave(drums, g, int((bar_start + gs * sec_per_beat) * synth_sr))
        elif trap_mode and (pre_hook or phrase_end):
            # Short transition flourish before hook/phrase turn.
            for gs in [3.5, 3.75]:
                g = (_snare(synth_sr, intensity * 0.42, rng) * 0.28).astype(np.float32)
                _add_wave(drums, g, int((bar_start + gs * sec_per_beat) * synth_sr))

        for t in hat_times:
            if trap_mode:
                hat_prob = 0.52 if in_hook else 0.34
            elif southern_mode:
                hat_prob = 0.58 if in_hook else 0.42
            elif melodic_mode:
                hat_prob = 0.5 if in_hook else 0.36
            elif eastcoast_mode:
                hat_prob = 0.46 if in_hook else 0.34
            elif boombap_mode:
                hat_prob = 0.42 if in_hook else 0.3
            elif lofi_mode:
                hat_prob = 0.34 if in_hook else 0.26
            elif westcoast_mode:
                hat_prob = 0.34 if in_hook else 0.24
            else:
                hat_prob = 0.22 if in_hook else 0.0
            if is_turnaround and not pre_hook:
                hat_prob *= 0.55
            if rng.random() < (hat_prob * profile["hat_density"]):
                s = _hihat(synth_sr, intensity * 0.82, rng) if not use_drum_samples else _choose_sample(
                    (drum_paths or {}).get("hats", []),
                    rng,
                    synth_sr,
                    _hihat(synth_sr, intensity * (0.86 + 0.1 * rng.random()), rng),
                )
                # Light velocity shape for a less robotic hi-hat lane.
                local_beat = ((float(t) - bar_start) / sec_per_beat) % 1.0
                hat_gain = 0.88 if local_beat > 0.45 else 1.0
                s = (s * hat_gain).astype(np.float32)
                _add_wave(drums, s, int(float(t) * synth_sr))

        # Extra top percussion layer
        if (not safe_mode) and density[bar] > 0.28 and in_hook:
            for t in np.arange(bar_start + 0.5 * sec_per_beat, bar_start + bar_len, sec_per_beat / 2.0, dtype=np.float32):
                if rng.random() < (profile["shaker_density"] * 0.9 * section["drums"]):
                    _add_wave(drums, _shaker(synth_sr, intensity, rng), int(float(t) * synth_sr))

        degree = progression_use[bar % len(progression_use)]
        chord = _minor_chord_freqs(root, degree, inversion=0)
        bass_hz = float(chord[0] * 0.5)
        if southern_mode:
            bass_pattern = [0.0, 1.0, 2.0, 2.75]
        elif melodic_mode:
            bass_pattern = [0.0, 1.5, 2.0, 3.0]
        elif eastcoast_mode:
            bass_pattern = [0.0, 2.0]
        elif boombap_mode:
            bass_pattern = [0.0, 2.0]
        elif lofi_mode:
            bass_pattern = [0.0, 2.0]
        elif westcoast_mode:
            bass_pattern = [0.0, 1.5, 2.5]
        else:
            bass_pattern = [0.0, 2.0] if safe_mode else bass_motifs[(motif_seed + (bar // 2)) % len(bass_motifs)]
        if safe_mode and in_hook:
            bass_pattern = [0.0, 1.5, 2.0, 3.0]
        if in_intro_outro:
            bass_pattern = [0.0, 2.0]
        bass_hit_times = [bar_start + b * sec_per_beat for b in bass_pattern]
        bass_events = ((midi_patterns or {}).get("basslines", {}) or {}).get("events", [])
        if bass_events:
            bass_hit_times = []
            for e in bass_events:
                ebeat = float(e["beat"])
                bar_beat_start = float(bar * 4)
                if bar_beat_start <= ebeat < bar_beat_start + 4.0:
                    local_beats = ebeat - bar_beat_start
                    bass_hit_times.append(bar_start + local_beats * sec_per_beat)
            if not bass_hit_times:
                bass_hit_times = [bar_start]
        for t in bass_hit_times:
            sustain = sec_per_beat * (0.68 if density[bar] > 0.5 else 0.52) * profile["bass_sustain"]
            s = _bass(synth_sr, bass_hz, sustain, intensity)
            _add_wave(bass_layer, s, int(t * synth_sr))

        # Chord bed + rhythmic key stabs with voice presets.
        inversion = 1 if section_name == "hook" and (bar % 2 == 1) else 0
        chord = _minor_chord_freqs(root, degree, inversion=inversion)
        render_pad = (not intro_guard) and (in_hook or (phrase_bar in (0, 4)) or (not in_intro_outro and density[bar] > 0.45))
        if render_pad:
            for ch_hz in chord:
                pad_note = _keys_voice(
                    synth_sr,
                    ch_hz,
                    bar_len,
                    strength=0.038 * intensity * section["music"],
                    voice=palette["pad"],
                    long_env=True,
                )
                _add_wave(music, pad_note, int(bar_start * synth_sr))
        if safe_mode and not in_hook and phrase_bar in (2, 6):
            # Leave breathing space every phrase to avoid constant wall of sound.
            music[int(bar_start * synth_sr): int((bar_start + bar_len) * synth_sr)] *= 0.82

        # Controlled melodic line for trap/boom bap without harsh stab spikes.
        if (trap_mode or southern_mode or melodic_mode or eastcoast_mode or boombap_mode or lofi_mode or westcoast_mode) and (not intro_guard) and (not in_intro_outro):
            motif_steps = [0.75, 2.25]
            if in_hook:
                motif_steps = [0.5, 1.75, 3.0]
            lead_pool = [chord[0] * 2.0, chord[1] * 2.0, chord[2] * 2.0]
            for mi, step in enumerate(motif_steps):
                note_hz = float(lead_pool[(bar + mi) % len(lead_pool)])
                mel = _keys_voice(
                    synth_sr,
                    note_hz,
                    sec_len=sec_per_beat * 0.24,
                    strength=(0.014 if lofi_mode else (0.017 if westcoast_mode else (0.016 if eastcoast_mode else (0.017 if southern_mode else (0.02 if melodic_mode else 0.018))))) * intensity * section["music"],
                    voice=("rhodes" if lofi_mode else ("analog_keys" if westcoast_mode else ("rhodes" if eastcoast_mode else ("analog_keys" if southern_mode else ("fm_bell" if melodic_mode else "soft_sine"))))),
                    long_env=False,
                )
                _add_wave(music, mel, int((bar_start + step * sec_per_beat) * synth_sr))

        stab_times = [bar_start, bar_start + sec_per_beat, bar_start + 2.0 * sec_per_beat, bar_start + 3.0 * sec_per_beat]
        chord_events = ((midi_patterns or {}).get("chords", {}) or {}).get("events", [])
        if (not intro_guard) and allow_key_stabs and (not safe_mode) and chord_events and not in_intro_outro:
            for e in chord_events:
                ebeat = float(e["beat"])
                bar_beat_start = float(bar * 4)
                if not (bar_beat_start <= ebeat < bar_beat_start + 4.0):
                    continue
                local_beats = ebeat - bar_beat_start
                st = bar_start + local_beats * sec_per_beat
                # Use MIDI timing but quantize pitch to in-key chord tones.
                note_hz = float(chord[int(e["note"]) % len(chord)])
                dur_beats = max(0.12, float(e.get("duration_beats", 0.35)))
                stab = _keys_voice(
                    synth_sr,
                    note_hz,
                    sec_len=min(bar_len, dur_beats * sec_per_beat),
                    strength=0.068 * intensity * section["music"],
                    voice=palette["keys"],
                    long_env=False,
                )
                _add_wave(music, stab, int(st * synth_sr))
        else:
            for st in stab_times:
                stab_prob = 0.42 if safe_mode else (0.62 if in_hook else (0.38 if not in_intro_outro else 0.16))
                if allow_key_stabs and (not intro_guard) and rng.random() < (stab_prob * profile["stab_density"] * section["music"]):
                    note_hz = float(chord[int(rng.integers(0, len(chord)))])
                    stab = _keys_voice(
                        synth_sr,
                        note_hz,
                        sec_len=sec_per_beat * (0.28 if in_hook else 0.35),
                        strength=0.032 * intensity * section["music"],
                        voice=palette["keys"],
                        long_env=False,
                    )
                    _add_wave(music, stab, int(st * synth_sr))

        # Sparse lead notes around vocal accents + arpeggio fragments
        melody_events = [] if (trap_mode or southern_mode or eastcoast_mode or boombap_mode or westcoast_mode) else ((midi_patterns or {}).get("melodies", {}) or {}).get("events", [])
        if (not intro_guard) and (not safe_mode) and melody_events and not in_intro_outro:
            lead_hits = 0
            for e in melody_events:
                ebeat = float(e["beat"])
                bar_beat_start = float(bar * 4)
                if not (bar_beat_start <= ebeat < bar_beat_start + 4.0):
                    continue
                if lead_hits >= (3 if in_hook else 2):
                    break
                local_beats = ebeat - bar_beat_start
                t = bar_start + local_beats * sec_per_beat
                # Keep lead in-key and narrow range.
                lead_pool = [chord[0], chord[1], chord[2], chord[0] * 2.0, chord[1] * 2.0]
                note_hz = float(lead_pool[int(e["note"]) % len(lead_pool)])
                dur_beats = max(0.1, float(e.get("duration_beats", 0.25)))
                lead = _keys_voice(
                    synth_sr,
                    note_hz,
                    sec_len=min(bar_len, dur_beats * sec_per_beat),
                    strength=0.058 * intensity,
                    voice=palette["lead"],
                    long_env=False,
                )
                _add_wave(music, lead, int(float(t) * synth_sr))
                lead_hits += 1
        elif (not intro_guard) and (not safe_mode) and bar_onsets.size > 0:
            lead_candidates = bar_onsets[: min(4, bar_onsets.size)]
            for t in lead_candidates:
                lead_prob = 0.4 if in_hook else (0.22 if not in_intro_outro else 0.0)
                if rng.random() < (lead_prob * profile["lead_density"] * section["music"]):
                    note_hz = float(chord[int(rng.integers(0, len(chord)))]) * rng.choice([1.0, 2.0])
                    lead = _keys_voice(
                        synth_sr,
                        note_hz,
                        sec_len=sec_per_beat * 0.28,
                        strength=0.055 * intensity,
                        voice=palette["lead"],
                        long_env=False,
                    )
                    _add_wave(music, lead, int(float(t) * synth_sr))

        if (not safe_mode) and in_hook and rng.random() < (0.34 + 0.44 * profile["arp_density"] * section["music"]):
            arp_start = bar_start + 0.5 * sec_per_beat
            arp_step = sec_per_beat / 2.0
            arp_notes = [chord[0], chord[1], chord[2], chord[1]]
            for i, note in enumerate(arp_notes):
                swing_push = profile["swing"] * sec_per_beat if i % 2 == 1 else 0.0
                at = arp_start + i * arp_step + swing_push
                arp = _keys_voice(
                    synth_sr,
                    float(note) * 2.0,
                    sec_len=sec_per_beat * 0.22,
                    strength=0.05 * intensity,
                    voice=palette["lead"],
                    long_env=False,
                )
                _add_wave(music, arp, int(at * synth_sr))
        # Fade music in during opening bars to avoid front-loaded dominant instrument.
        if intro_guard:
            seg_start = int(bar_start * synth_sr)
            seg_end = int((bar_start + bar_len) * synth_sr)
            fade = 0.18 if bar == 0 else 0.38
            music[seg_start:seg_end] *= fade
    drums = _one_pole_highpass(drums, synth_sr, 28.0)
    drums = _soft_clip(drums * 1.08, drive=1.12)
    drums = _bus_compress(drums, amount=0.22)

    bass_layer = _one_pole_lowpass(bass_layer, synth_sr, 1200.0)
    bass_layer = _soft_clip(bass_layer * 1.14, drive=1.2)

    music = _one_pole_highpass(music, synth_sr, 115.0)
    if loop_layer is not None and loop_layer.size == music.size:
        loop_shaped = _one_pole_highpass(loop_layer, synth_sr, 180.0)
        loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 6200.0)
        if boombap_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 2800.0)
            loop_mix = 0.1
        elif eastcoast_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 2600.0)
            loop_mix = 0.12
        elif southern_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 2800.0)
            loop_mix = 0.05
        elif melodic_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 4200.0)
            loop_mix = 0.1
        elif lofi_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 2200.0)
            loop_mix = 0.08
        elif westcoast_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 3000.0)
            loop_mix = 0.06
        elif trap_mode:
            loop_shaped = _one_pole_lowpass(loop_shaped, synth_sr, 2400.0)
            loop_mix = 0.03
        else:
            loop_mix = 0.0 if safe_mode else 0.045
        music = (music + loop_shaped * loop_mix).astype(np.float32)

    if lofi_mode:
        # Subtle vinyl-ish texture for lo-fi atmosphere.
        noise = rng.normal(0.0, 1.0, music.size).astype(np.float32) * 0.004
        noise = _one_pole_lowpass(noise, synth_sr, 5000.0)
        music = (music + noise).astype(np.float32)

    music = _simple_reverb(music, synth_sr, mix=0.14 if (instrument_pack or "auto").lower() != "dark" else 0.08)
    music = _soft_clip(music * 1.03, drive=1.06)
    # Keep melodic content audible: gently duck rhythm buses when music is present.
    music_env = np.convolve(np.abs(music), np.ones(2048, dtype=np.float32) / 2048.0, mode="same")
    rhythm_duck = 1.0 - np.clip(music_env * 0.22, 0.0, 0.22)
    drums = (drums * rhythm_duck).astype(np.float32)
    bass_layer = (bass_layer * (0.9 + 0.1 * rhythm_duck)).astype(np.float32)
    # Global anti-spike guard: no single instrument burst should dominate.
    rhythm_guard_bus = drums * profile["drum_level"] + bass_layer * profile["bass_level"]
    m_env = np.convolve(np.abs(music), np.ones(1024, dtype=np.float32) / 1024.0, mode="same")
    r_env = np.convolve(np.abs(rhythm_guard_bus), np.ones(1024, dtype=np.float32) / 1024.0, mode="same")
    target = 0.62 * r_env + 0.03
    guard_gain = np.minimum(1.0, target / (m_env + 1e-6))
    music = (music * guard_gain).astype(np.float32)
    # Hard per-bar cap for recurring spikes in any song section.
    bar_samples = max(1, int(round(bar_len * synth_sr)))
    for b in range(bars):
        s0 = b * bar_samples
        s1 = min(music.size, s0 + bar_samples)
        if s1 <= s0:
            continue
        seg_m = music[s0:s1]
        seg_r = rhythm_guard_bus[s0:s1]
        peak_m = float(np.max(np.abs(seg_m)))
        peak_r = float(np.max(np.abs(seg_r)))
        cap = max(0.08, 0.95 * peak_r)
        if peak_m > cap and peak_m > 0.0:
            music[s0:s1] = (seg_m * (cap / peak_m)).astype(np.float32)
    # Final short-window ceiling for music spikes anywhere in timeline.
    m_short = np.convolve(np.abs(music), np.ones(256, dtype=np.float32) / 256.0, mode="same")
    r_short = np.convolve(np.abs(rhythm_guard_bus), np.ones(256, dtype=np.float32) / 256.0, mode="same")
    hard_cap = np.maximum(0.06, 0.72 * r_short + 0.02)
    hard_gain = np.minimum(1.0, hard_cap / (m_short + 1e-6))
    music = (music * hard_gain).astype(np.float32)
    rhythm_bus = (
        drums * profile["drum_level"]
        + bass_layer * profile["bass_level"]
    )
    music_bus = music * profile["music_level"]
    if trap_mode:
        music_ratio = 2.35
    elif southern_mode:
        music_ratio = 2.2
    elif melodic_mode:
        music_ratio = 2.3
    elif eastcoast_mode:
        music_ratio = 1.95
    elif boombap_mode:
        music_ratio = 2.1
    elif lofi_mode:
        music_ratio = 1.9
    elif westcoast_mode:
        music_ratio = 2.0
    else:
        music_ratio = 1.4
    full = (rhythm_bus * 1.0 + music_bus * music_ratio).astype(np.float32)
    if safe_mode:
        full = _one_pole_lowpass(full, synth_sr, 3200.0)
    elif trap_mode:
        full = _one_pole_lowpass(full, synth_sr, 5200.0)
    elif southern_mode:
        full = _one_pole_lowpass(full, synth_sr, 4600.0)
    elif melodic_mode:
        full = _one_pole_lowpass(full, synth_sr, 5200.0)
    elif eastcoast_mode:
        full = _one_pole_lowpass(full, synth_sr, 3400.0)
    elif boombap_mode:
        full = _one_pole_lowpass(full, synth_sr, 3600.0)
    elif lofi_mode:
        full = _one_pole_lowpass(full, synth_sr, 3000.0)
    elif westcoast_mode:
        full = _one_pole_lowpass(full, synth_sr, 4200.0)
    full = _bus_compress(full, amount=0.24)
    full = _soft_clip(full, drive=1.08)
    # Per-bar debug map for arrangement verification.
    step_len = sec_per_beat / 4.0
    for bar in range(bars):
        bar_start = bar * bar_len
        bar_end = bar_start + bar_len
        onset_steps = []
        if flow_grid is not None and bar < len(flow_grid):
            onset_steps = flow_grid[bar].get("onset_steps", [])
        k_steps: List[int] = []
        s_steps: List[int] = []
        h_steps: List[int] = []
        # Reconstruct coarse hit map from rendered layers by simple peak sampling.
        # This is intentionally lightweight for debugging only.
        for st in range(16):
            t = bar_start + st * step_len
            i = int(t * synth_sr)
            if 0 <= i < drums.size:
                v = abs(float(drums[i]))
                if v > 0.18:
                    k_steps.append(st)
                elif v > 0.1:
                    s_steps.append(st)
                elif v > 0.05:
                    h_steps.append(st)
        sec_name = section_labels[bar] if section_labels is not None and bar < len(section_labels) else "unknown"
        phrase_bar = bar % phrase_len
        next_label = section_labels[bar + 1] if section_labels is not None and (bar + 1) < len(section_labels) else sec_name
        bar_debug.append(
            {
                "bar": int(bar),
                "section": sec_name,
                "bar_density": float(round(float(density[bar]), 4)),
                "onset_steps": onset_steps,
                "kick_steps": sorted(set(k_steps)),
                "snare_steps": sorted(set(s_steps)),
                "hat_steps": sorted(set(h_steps)),
                "active_layers": {
                    "drums": True,
                    "bass": bool(np.max(np.abs(bass_layer[int(bar_start * synth_sr): int(bar_end * synth_sr)])) > 0.01),
                    "music": bool(np.max(np.abs(music[int(bar_start * synth_sr): int(bar_end * synth_sr)])) > 0.01),
                },
                "flags": {
                    "phrase_end": bool(phrase_bar == (phrase_len - 1)),
                    "pre_hook": bool((sec_name != "hook") and (next_label == "hook")),
                },
            }
        )
    return {
        "full": full.astype(np.float32),
        "drums": drums.astype(np.float32),
        "bass": bass_layer.astype(np.float32),
        "music": music.astype(np.float32),
        "palette": palette,
        "style_family": style_family,
        "arrangement_debug": bar_debug,
    }


