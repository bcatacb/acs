import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import mido
except Exception:  # pragma: no cover
    mido = None

from services.library_index import load_catalog_for_generation

from .analysis import _decode_audio_mono
from .common import ASSET_DROP_DIR
from .render import _bus_compress, _one_pole_lowpass

_AUDIO_CACHE: Dict[str, np.ndarray] = {}
def _resample_linear(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or audio.size == 0:
        return audio.astype(np.float32, copy=True)
    ratio = float(to_sr) / float(from_sr)
    out_len = max(1, int(round(audio.size * ratio)))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)

def _audio_files_under(path: Path) -> List[Path]:
    if not path.exists():
        return []
    out: List[Path] = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac", "*.aif", "*.aiff"):
        out.extend(path.rglob(ext))
    return sorted(set(out))

def _midi_files_under(path: Path) -> List[Path]:
    if not path.exists():
        return []
    out: List[Path] = []
    out.extend(path.rglob("*.mid"))
    out.extend(path.rglob("*.midi"))
    return sorted(set(out))

def _decode_audio_cached(path: Path, sr: int) -> np.ndarray:
    key = f"{str(path)}::{sr}"
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    data = _decode_audio_mono(str(path), target_sr=sr)
    _AUDIO_CACHE[key] = data
    return data

def _collect_asset_catalog() -> Dict:
    # Prefer indexed assets from SQLite; fall back to filesystem scan if DB is empty.
    try:
        indexed = load_catalog_for_generation()
        if (
            indexed.get("drums", {}).get("kicks")
            or indexed.get("drums", {}).get("hats")
            or indexed.get("drums", {}).get("claps")
            or indexed.get("loops")
            or indexed.get("midi_files")
        ):
            return indexed
    except Exception:
        pass

    drum_base = ASSET_DROP_DIR / "Instruments" / "OneShots" / "Drums"
    loops_base = ASSET_DROP_DIR / "Loops"
    midi_base = ASSET_DROP_DIR / "MIDI"

    kicks = _audio_files_under(drum_base / "Kicks")
    hats = _audio_files_under(drum_base / "Hats")
    claps = _audio_files_under(drum_base / "Clapsnares")
    loops = _audio_files_under(loops_base)
    midi_files = _midi_files_under(midi_base)
    midi_by_role = {
        "chords": _midi_files_under(midi_base / "Chords"),
        "melodies": _midi_files_under(midi_base / "Melodies"),
        "basslines": _midi_files_under(midi_base / "Basslines"),
        "drums": _midi_files_under(midi_base / "Drums"),
    }

    loop_entries = []
    pat = re.compile(r"([A-G](?:#|b)?(?:Maj|min))\\s+(\\d{2,3})bpm", re.IGNORECASE)
    for p in loops:
        # Skip stem files for primary loop source.
        lower_parts = [part.lower() for part in p.parts]
        if "stems" in lower_parts:
            continue
        m = pat.search(p.stem)
        bpm = int(m.group(2)) if m else None
        key = m.group(1) if m else None
        loop_entries.append({"path": p, "bpm": bpm, "key": key})

    return {
        "drums": {"kicks": kicks, "hats": hats, "claps": claps},
        "loops": loop_entries,
        "midi_files": midi_files,
        "midi_by_role": midi_by_role,
    }

def _find_reference_instrumental() -> Optional[Path]:
    inbox = ASSET_DROP_DIR / "Inbox"
    if not inbox.exists():
        return None
    preferred = sorted(inbox.glob("*Rotattion*.wav"))
    if preferred:
        return preferred[0]
    any_wav = sorted(inbox.glob("*.wav"))
    if any_wav:
        return any_wav[0]
    return None

def _prepare_reference_layer(target_len: int, sr: int, seed: int) -> Dict:
    ref = _find_reference_instrumental()
    if ref is None:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}
    try:
        audio = _decode_audio_cached(ref, sr)
    except Exception:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}
    if audio.size == 0:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}
    if audio.size >= target_len:
        # Deterministic and stable quality: use the opening section.
        start = 0
        layer = audio[start : start + target_len].astype(np.float32)
    else:
        reps = int(math.ceil(target_len / float(audio.size)))
        layer = np.tile(audio, max(1, reps))[:target_len].astype(np.float32)
    # Keep reference backbone controlled.
    layer = _one_pole_lowpass(layer.astype(np.float32), sr, 5400.0)
    layer = _bus_compress(layer, amount=0.18)
    return {"layer": layer, "source": str(ref)}

def _choose_sample(paths: List[Path], rng: np.random.Generator, sr: int, fallback: np.ndarray) -> np.ndarray:
    if not paths:
        return fallback
    try:
        p = paths[int(rng.integers(0, len(paths)))]
        s = _decode_audio_cached(p, sr).astype(np.float32, copy=True)
        if s.size == 0:
            return fallback
        # Remove DC and normalize for more consistent one-shot balance.
        s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)))
        if peak > 0:
            s = s / peak * 0.92
        return s
    except Exception:
        return fallback

def _prepare_loop_layer(catalog: Dict, target_bpm: int, target_len: int, sr: int, rng: np.random.Generator) -> Dict:
    loops = catalog.get("loops", [])
    if not loops:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}

    # Prefer entries with BPM metadata close to target BPM.
    scored = []
    for e in loops:
        if e.get("bpm") is None:
            score = 999.0
        else:
            score = abs(float(e["bpm"]) - float(target_bpm))
        scored.append((score, e))
    scored.sort(key=lambda x: x[0])
    top = [e for _, e in scored[: min(8, len(scored))]]
    pick = top[int(rng.integers(0, len(top)))]
    src = pick["path"]

    try:
        loop_audio = _decode_audio_cached(src, sr)
    except Exception:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}

    src_bpm = pick.get("bpm")
    if src_bpm and src_bpm > 0:
        ratio = float(src_bpm) / float(target_bpm)
        stretched = _resample_linear(loop_audio, sr, int(sr * ratio))
        loop_audio = _resample_linear(stretched, int(sr * ratio), sr)

    if loop_audio.size == 0:
        return {"layer": np.zeros(target_len, dtype=np.float32), "source": None}

    reps = int(math.ceil(target_len / float(loop_audio.size)))
    tiled = np.tile(loop_audio, max(1, reps))[:target_len].astype(np.float32)
    return {"layer": tiled, "source": str(src)}

def _midi_to_events(path: Path) -> List[Dict]:
    if mido is None:
        return []
    try:
        mf = mido.MidiFile(str(path))
    except Exception:
        return []

    ticks_per_beat = float(max(1, mf.ticks_per_beat))
    events: List[Dict] = []

    note_on_beats: Dict[int, List[float]] = {}
    for track in mf.tracks:
        beat = 0.0
        for msg in track:
            beat += float(msg.time) / ticks_per_beat
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                note_on_beats.setdefault(int(msg.note), []).append(float(beat))
            elif msg.type in ("note_off", "note_on"):
                if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                    continue
                note = int(msg.note)
                starts = note_on_beats.get(note, [])
                if starts:
                    start = starts.pop(0)
                    dur = max(0.1, float(beat) - start)
                    events.append(
                        {
                            "beat": start,
                            "note": note,
                            "velocity": int(getattr(msg, "velocity", 90) or 90),
                            "duration_beats": dur,
                        }
                    )
    events.sort(key=lambda e: e["beat"])
    return events

def _prepare_midi_events(catalog: Dict, role: str, bars: int, beats_per_bar: float, rng: np.random.Generator) -> Dict:
    files = (catalog.get("midi_by_role", {}) or {}).get(role, [])
    if not files:
        return {"source": None, "events": []}

    src = files[int(rng.integers(0, len(files)))]
    base = _midi_to_events(src)
    if not base:
        return {"source": str(src), "events": []}

    max_beat = max(e["beat"] + e["duration_beats"] for e in base)
    pattern_beats = max(beats_per_bar, float(max_beat))
    total_beats = bars * beats_per_bar

    tiled: List[Dict] = []
    offset = 0.0
    while offset < total_beats:
        for e in base:
            b = float(e["beat"]) + offset
            if b >= total_beats:
                continue
            tiled.append(
                {
                    "beat": b,
                    "note": int(e["note"]),
                    "velocity": int(e["velocity"]),
                    "duration_beats": float(e["duration_beats"]),
                }
            )
        offset += pattern_beats

    tiled.sort(key=lambda e: e["beat"])
    return {"source": str(src), "events": tiled}


