from typing import Dict, List

import av
import numpy as np
def _safe_bpm(value: float, fallback: int = 92) -> int:
    try:
        bpm = int(round(float(value)))
    except Exception:
        return fallback
    return max(70, min(190, bpm))

def _decode_audio_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise ValueError("No audio stream found")

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=target_sr,
    )

    chunks: List[np.ndarray] = []
    for frame in container.decode(stream):
        for rframe in resampler.resample(frame):
            arr = rframe.to_ndarray()
            if arr.size:
                chunks.append(arr[0].astype(np.float32, copy=True))

    for rframe in resampler.resample(None):
        arr = rframe.to_ndarray()
        if arr.size:
            chunks.append(arr[0].astype(np.float32, copy=True))

    if not chunks:
        raise ValueError("Audio decoding produced no samples")

    audio = np.concatenate(chunks)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio

def _detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    frame_len = 1024
    hop = 256
    if audio.size < frame_len * 2:
        return np.array([], dtype=np.float32)

    energies = []
    for i in range(0, audio.size - frame_len, hop):
        frame = audio[i : i + frame_len]
        energies.append(float(np.mean(frame * frame)))
    energy = np.array(energies, dtype=np.float32)
    novelty = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    novelty = np.convolve(novelty, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    threshold = float(np.mean(novelty) + 1.2 * np.std(novelty))
    refractory = max(1, int(0.08 * sr / hop))

    picks: List[int] = []
    last = -999999
    for i in range(1, novelty.size - 1):
        if novelty[i] < threshold:
            continue
        if novelty[i] < novelty[i - 1] or novelty[i] < novelty[i + 1]:
            continue
        if i - last < refractory:
            continue
        picks.append(i)
        last = i

    if not picks:
        return np.array([], dtype=np.float32)
    return np.array(picks, dtype=np.float32) * (hop / float(sr))

def _estimate_bpm(onsets: np.ndarray, fallback_bpm: int) -> int:
    if onsets.size < 4:
        return fallback_bpm

    intervals = np.diff(onsets)
    intervals = intervals[(intervals > 0.2) & (intervals < 1.5)]
    if intervals.size == 0:
        return fallback_bpm

    bpms = 60.0 / intervals
    folded = []
    for bpm in bpms:
        while bpm < 70:
            bpm *= 2.0
        while bpm > 190:
            bpm /= 2.0
        folded.append(bpm)
    if not folded:
        return fallback_bpm
    return _safe_bpm(float(np.median(np.array(folded, dtype=np.float32))), fallback_bpm)

def _bar_density(onsets: np.ndarray, bars: int, bar_len: float) -> np.ndarray:
    density = np.zeros(bars, dtype=np.float32)
    for t in onsets:
        idx = int(t // bar_len)
        if 0 <= idx < bars:
            density[idx] += 1.0
    if float(np.max(density)) > 0:
        density = density / float(np.max(density))
    return density

def _build_bar_grid(onsets: np.ndarray, bars: int, bar_len: float, sec_per_beat: float) -> List[Dict]:
    grid: List[Dict] = []
    step_len = sec_per_beat / 4.0
    for bar in range(bars):
        start = bar * bar_len
        end = start + bar_len
        bar_onsets = onsets[(onsets >= start) & (onsets < end)]
        steps = np.zeros(16, dtype=np.int32)
        rel: List[float] = []
        for t in bar_onsets:
            local = float(t - start)
            rel.append(local)
            st = int(round(local / max(1e-6, step_len)))
            st = max(0, min(15, st))
            steps[st] = 1
        first = rel[0] if rel else None
        grid.append(
            {
                "bar": bar,
                "onset_count": int(len(rel)),
                "onset_steps": [int(i) for i, v in enumerate(steps.tolist()) if v > 0],
                "active_steps": int(np.sum(steps)),
                "downbeat_hit": bool(steps[0] or steps[1]),
                "pickup": bool(first is not None and first > (sec_per_beat * 0.2) and first < (sec_per_beat * 1.25)),
            }
        )
    return grid

def _classify_sections_from_grid(grid: List[Dict]) -> List[str]:
    if not grid:
        return []
    counts = np.array([float(g.get("onset_count", 0)) for g in grid], dtype=np.float32)
    active = np.array([float(g.get("active_steps", 0)) for g in grid], dtype=np.float32)
    energy = counts + 0.35 * active
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    smooth = np.convolve(energy, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
    nz = smooth[smooth > 0.01]
    q_hi = float(np.quantile(nz, 0.75)) if nz.size else 0.65
    q_mid = float(np.quantile(nz, 0.45)) if nz.size else 0.35

    labels: List[str] = []
    for i, g in enumerate(grid):
        if g["onset_count"] <= 0 or g["active_steps"] <= 1:
            labels.append("break")
            continue
        if smooth[i] >= q_hi:
            labels.append("hook")
        elif smooth[i] >= q_mid:
            labels.append("busy_verse")
        else:
            labels.append("chill_verse")

    # Smooth single-bar spikes/dips into neighboring section.
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            labels[i] = labels[i - 1]
    return labels

def _section_levels(name: str) -> Dict[str, float]:
    table = {
        "hook": {"energy": 1.12, "music": 1.0, "drums": 1.08},
        "busy_verse": {"energy": 0.98, "music": 0.86, "drums": 1.0},
        "chill_verse": {"energy": 0.86, "music": 0.74, "drums": 0.9},
        "break": {"energy": 0.7, "music": 0.55, "drums": 0.65},
    }
    return table.get(name, {"energy": 0.95, "music": 0.85, "drums": 0.95})


