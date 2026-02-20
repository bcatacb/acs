import hashlib
import json
import math
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .analysis import _bar_density, _build_bar_grid, _classify_sections_from_grid, _decode_audio_mono, _detect_onsets, _estimate_bpm, _safe_bpm
from .assets import _collect_asset_catalog, _find_reference_instrumental, _prepare_loop_layer, _prepare_midi_events, _resample_linear
from .common import GENERATED_DIR
from .render import _build_vocal_mix, _compute_repetition_score, _render_mix, _write_wav_stereo
from .style import _genre_root_hz, _is_boom_bap_style, _is_east_coast_style, _is_lofi_style, _is_melodic_style, _is_southern_style, _is_trap_style, _is_west_coast_style

logger = logging.getLogger(__name__)

# Optional: Import the SoundBankProvider if available
try:
    from soundbank_api import SoundBankProvider
    SOUNDBANK_AVAILABLE = True
except ImportError:
    SOUNDBANK_AVAILABLE = False
    SoundBankProvider = None

# Determine Sound Bank file paths
def _get_soundbank_paths():
    """Locate Sound Bank files in either ACS or app backend directories."""
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "app" / "backend",  # ../../../app/backend
        Path(__file__).parent.parent.parent,  # ../../.. (ACS/backend)
        Path.cwd(),  # Current working directory
    ]
    
    for base_path in candidates:
        master_wav = base_path / "master_library.wav"
        db_path = base_path / "vocal_vault.db"
        if master_wav.exists() and db_path.exists():
            logger.info(f"Found Sound Bank files at {base_path}")
            return str(master_wav), str(db_path)
    
    logger.debug(f"Sound Bank files not found in checked paths: {[str(p) for p in candidates]}")
    return None, None
async def generate_true_accompaniment(
    project_id: str,
    audio_file_path: str,
    analysis: Dict,
    genre: str,
    instrument_pack: str = "auto",
) -> Dict:
    target_sr = 16000
    synth_sr = 44100

    decoded = _decode_audio_mono(audio_file_path, target_sr=target_sr)
    vocal_duration = decoded.size / float(target_sr)

    fallback_bpm = _safe_bpm(analysis.get("bpm", 92), 92)
    onsets = _detect_onsets(decoded, target_sr)
    bpm = _estimate_bpm(onsets, fallback_bpm)
    if _is_southern_style(genre):
        # Dirty South commonly spans slower menace to uptempo bounce.
        if bpm < 70 or bpm > 140:
            bpm = 96
    elif _is_melodic_style(genre):
        # Melodic rap usually runs in a higher-energy modern pocket.
        if bpm < 120 or bpm > 160:
            bpm = 140
    elif _is_trap_style(genre):
        # Trap typically lives at 70-80 BPM or double-time 140-160.
        if 82 <= bpm <= 130:
            bpm = 75
        bpm = max(70, min(160, bpm))
    elif _is_east_coast_style(genre):
        # East Coast boom-bap pocket is usually around 85-100.
        if bpm < 85 or bpm > 100:
            bpm = 92
    elif _is_boom_bap_style(genre):
        # Boom bap pocket is usually around 80-95.
        if bpm < 80 or bpm > 95:
            bpm = 90
    elif _is_lofi_style(genre):
        # Lo-fi sits around 70-90.
        if bpm < 70 or bpm > 90:
            bpm = 82
    elif _is_west_coast_style(genre):
        # West Coast groove commonly sits around 85-100.
        if bpm < 85 or bpm > 100:
            bpm = 92

    sec_per_beat = 60.0 / float(bpm)
    bar_len = sec_per_beat * 4.0
    total_duration = max(vocal_duration + 1.0, bar_len * 8.0)
    bars = max(8, int(math.ceil(total_duration / bar_len)))
    total_samples = int(bars * bar_len * synth_sr)

    density = _bar_density(onsets, bars, bar_len)
    flow_grid = _build_bar_grid(onsets, bars, bar_len, sec_per_beat)
    section_labels = _classify_sections_from_grid(flow_grid)
    root = _genre_root_hz(genre)
    base_seed_src = f"{project_id}:{bpm}:{onsets.size}:{genre}".encode("utf-8")
    base_seed = int(hashlib.sha256(base_seed_src).hexdigest()[:16], 16)
    
    # Initialize Sound Bank Provider if available.
    # This replaces folder-based asset collection with intensity-matched queries.
    loop_layer_from_provider = None
    provider = None
    soundbank_used = False
    
    if SOUNDBANK_AVAILABLE:
        master_wav_path, db_path = _get_soundbank_paths()
        if master_wav_path and db_path:
            try:
                provider = SoundBankProvider(
                    master_wav=master_wav_path,
                    db_path=db_path
                )
                
                # Calculate the average vocal energy (0.0-1.0)
                avg_density = float(np.mean(density))
                
                # Query the provider for an intensity-matched loop
                loop_id = provider.get_by_normalized_intensity(avg_density)
                
                if loop_id is not None:
                    loop_layer_from_provider = provider.get_audio(loop_id)
                    soundbank_used = True
                    logger.info(f"[{project_id}] Sound Bank: Loaded loop {loop_id} with intensity {avg_density:.2f}")
                else:
                    logger.warning(f"[{project_id}] Sound Bank: No assets matched intensity {avg_density:.2f}")
            except Exception as e:
                logger.warning(f"[{project_id}] Sound Bank error: {e}. Falling back to folder scanning.")
                provider = None
        else:
            logger.info(f"[{project_id}] Sound Bank files not found. Using folder-based catalog.")
    else:
        logger.debug(f"[{project_id}] Sound Bank Provider not available (module not found).")
    
    # Fall back to folder-based catalog if provider is not available
    catalog = _collect_asset_catalog()
    rng_assets = np.random.default_rng(base_seed + 1337)
    
    # Use provider-sourced loop if available, otherwise prepare from catalog
    if loop_layer_from_provider is not None:
        loop_pick = {
            "layer": loop_layer_from_provider.astype(np.float32) if loop_layer_from_provider.size > 0 else None,
            "source": "sound_bank_provider",
        }
        logger.info(f"[{project_id}] Using Sound Bank loop layer")
    else:
        loop_pick = _prepare_loop_layer(
            catalog=catalog,
            target_bpm=bpm,
            target_len=total_samples,
            sr=synth_sr,
            rng=rng_assets,
        )
    
    reference_source = _find_reference_instrumental()
    midi_patterns = {
        "chords": _prepare_midi_events(catalog, "chords", bars, 4.0, rng_assets),
        "melodies": _prepare_midi_events(catalog, "melodies", bars, 4.0, rng_assets),
        "basslines": _prepare_midi_events(catalog, "basslines", bars, 4.0, rng_assets),
    }
    if _is_southern_style(genre):
        progressions = [[0, 5, 4, 3], [0, 5, 3, 4], [0, 4, 3, 5]]
    elif _is_melodic_style(genre):
        progressions = [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 4, 5]]
    elif _is_trap_style(genre):
        progressions = [[0, 5, 3, 4], [0, 5, 0, 4], [0, 3, 5, 4]]
    elif _is_east_coast_style(genre):
        progressions = [[0, 3, 5, 4], [0, 5, 3, 4], [0, 4, 5, 3]]
    elif _is_boom_bap_style(genre):
        progressions = [[0, 3, 5, 4], [0, 5, 3, 4], [0, 4, 5, 3]]
    elif _is_lofi_style(genre):
        progressions = [[0, 3, 5, 4], [0, 4, 5, 3], [0, 3, 4, 5]]
    elif _is_west_coast_style(genre):
        progressions = [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 5, 4]]
    else:
        progressions = [[0, 5, 3, 4], [0, 3, 5, 4]]
    progression = progressions[int(rng_assets.integers(0, len(progressions)))]

    # Rule check: reject/rerender overly repetitive arrangements.
    repetition_score = _compute_repetition_score(density)
    max_attempts = 3
    accepted_score = 0.72
    selected_mix = None
    selected_attempt = 0
    best_score = 1.0

    for attempt in range(max_attempts):
        # Each attempt increases arrangement variation pressure.
        jitter = np.clip(np.random.default_rng(base_seed + attempt).normal(0, 0.12, bars), -0.22, 0.22).astype(np.float32)
        trial_density = np.clip(density + jitter, 0.0, 1.0)
        trial_rep = _compute_repetition_score(trial_density)
        trial_layers = _render_mix(
            bars=bars,
            bar_len=bar_len,
            sec_per_beat=sec_per_beat,
            synth_sr=synth_sr,
            onsets=onsets,
            density=trial_density,
            root=root,
            seed=base_seed + (attempt * 7919),
            genre=genre,
            instrument_pack=instrument_pack,
            drum_paths=catalog.get("drums", {}),
            loop_layer=loop_pick.get("layer"),
            flow_grid=flow_grid,
            section_labels=section_labels,
            progression=progression,
            midi_patterns=midi_patterns,
        )

        if trial_rep < best_score:
            best_score = trial_rep
            selected_mix = trial_layers
            selected_attempt = attempt + 1

        if trial_rep <= accepted_score:
            repetition_score = trial_rep
            selected_mix = trial_layers
            selected_attempt = attempt + 1
            break

    if selected_mix is None:
        repetition_score = best_score
        selected_mix = _render_mix(
            bars=bars,
            bar_len=bar_len,
            sec_per_beat=sec_per_beat,
            synth_sr=synth_sr,
            onsets=onsets,
            density=density,
            root=root,
            seed=base_seed,
            genre=genre,
            instrument_pack=instrument_pack,
            drum_paths=catalog.get("drums", {}),
            loop_layer=loop_pick.get("layer"),
            flow_grid=flow_grid,
            section_labels=section_labels,
            progression=progression,
            midi_patterns=midi_patterns,
        )
        selected_attempt = 1

    out_name = f"{project_id}_accompaniment.wav"
    out_path = GENERATED_DIR / out_name
    _write_wav_stereo(out_path, selected_mix["full"], synth_sr)

    # Build mixed render: acapella + generated accompaniment
    vocal_resampled = _resample_linear(decoded, target_sr, synth_sr)
    pre_roll_samples = int(round(bar_len * synth_sr)) if (onsets.size > 0 and float(onsets[0]) > (sec_per_beat * 0.35)) else 0
    if pre_roll_samples > 0:
        vocal_for_mix = np.concatenate([np.zeros(pre_roll_samples, dtype=np.float32), vocal_resampled], axis=0)
    else:
        vocal_for_mix = vocal_resampled
    mixed_render = _build_vocal_mix(selected_mix["full"], vocal_for_mix)
    mix_out_name = f"{project_id}_mix.wav"
    mix_out_path = GENERATED_DIR / mix_out_name
    _write_wav_stereo(mix_out_path, mixed_render, synth_sr)
    debug_path = GENERATED_DIR / f"{project_id}_arrangement_debug.json"
    debug_payload = {
        "project_id": project_id,
        "genre": genre,
        "bpm": bpm,
        "progression": progression,
        "pre_roll_samples": int(pre_roll_samples),
        "bars": selected_mix.get("arrangement_debug", []),
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2)

    return {
        "success": True,
        "audio_path": str(out_path),
        "mix_path": str(mix_out_path),
        "arrangement_debug_path": str(debug_path),
        "duration": round(float(total_duration), 2),
        "bpm": bpm,
        "onset_count": int(onsets.size),
        "repetition_score": round(float(repetition_score), 3),
        "render_attempts": int(selected_attempt),
        "instrument_palette": selected_mix.get("palette", {}),
        "style_family": selected_mix.get("style_family"),
        "instrument_pack": (instrument_pack or "auto").lower(),
        "instrument_system_version": "v2",
        "loop_source": loop_pick.get("source"),
        "soundbank_enabled": SOUNDBANK_AVAILABLE,
        "soundbank_used": soundbank_used,
        "soundbank_provider": "Available" if provider is not None else "Not available/Not configured",
        "vocal_intensity_matched": float(np.mean(density)) if soundbank_used else None,
        "reference_source": str(reference_source) if reference_source else None,
        "midi_sources": {
            "chords": (midi_patterns.get("chords") or {}).get("source"),
            "melodies": (midi_patterns.get("melodies") or {}).get("source"),
            "basslines": (midi_patterns.get("basslines") or {}).get("source"),
        },
        "asset_counts": {
            "kick_samples": len(catalog.get("drums", {}).get("kicks", [])),
            "hat_samples": len(catalog.get("drums", {}).get("hats", [])),
            "clap_samples": len(catalog.get("drums", {}).get("claps", [])),
            "loops": len(catalog.get("loops", [])),
            "midi_files": len(catalog.get("midi_files", [])),
        },
        "title": f"Flow-locked {genre.replace('_', ' ').title()} Accompaniment",
    }


