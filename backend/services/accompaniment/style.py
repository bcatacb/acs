from typing import Dict, List
def _genre_root_hz(genre: str) -> float:
    roots = {
        "trap": 43.65,       # F1
        "drill": 41.20,      # E1
        "boom_bap": 49.00,   # G1
        "lo_fi": 55.00,      # A1
        "west_coast": 46.25, # F#1
        "east_coast": 51.91, # G#1
        "southern": 38.89,   # D#1
        "melodic": 58.27,    # A#1
    }
    return roots.get(genre, 43.65)

def _genre_palette(genre: str) -> Dict[str, str]:
    palettes = {
        "trap": {"pad": "dark_pad", "keys": "analog_keys", "lead": "fm_bell"},
        "drill": {"pad": "dark_pad", "keys": "saw_pluck", "lead": "fm_bell"},
        "boom_bap": {"pad": "warm_pad", "keys": "rhodes", "lead": "analog_keys"},
        "lo_fi": {"pad": "warm_pad", "keys": "rhodes", "lead": "soft_sine"},
        "west_coast": {"pad": "warm_pad", "keys": "analog_keys", "lead": "saw_pluck"},
        "east_coast": {"pad": "dark_pad", "keys": "rhodes", "lead": "analog_keys"},
        "southern": {"pad": "dark_pad", "keys": "analog_keys", "lead": "saw_pluck"},
        "melodic": {"pad": "warm_pad", "keys": "rhodes", "lead": "fm_bell"},
    }
    return palettes.get(genre, {"pad": "dark_pad", "keys": "analog_keys", "lead": "fm_bell"})

def _resolve_palette(genre: str, instrument_pack: str) -> Dict[str, str]:
    pack = (instrument_pack or "auto").lower()
    if pack == "dark":
        return {"pad": "dark_pad", "keys": "saw_pluck", "lead": "fm_bell"}
    if pack == "warm":
        return {"pad": "warm_pad", "keys": "rhodes", "lead": "soft_sine"}
    if pack == "orchestral":
        return {"pad": "string_ensemble", "keys": "piano_keys", "lead": "flute_lead"}
    return _genre_palette(genre)

def _pack_profile(instrument_pack: str) -> Dict[str, float]:
    pack = (instrument_pack or "auto").lower()
    if pack == "dark":
        return {
            "drum_level": 1.18,
            "bass_level": 1.28,
            "music_level": 0.78,
            "hat_density": 0.72,
            "shaker_density": 0.35,
            "stab_density": 0.55,
            "arp_density": 0.45,
            "lead_density": 0.48,
            "bass_sustain": 1.15,
            "swing": 0.03,
        }
    if pack == "warm":
        return {
            "drum_level": 0.88,
            "bass_level": 0.94,
            "music_level": 1.22,
            "hat_density": 0.95,
            "shaker_density": 0.62,
            "stab_density": 0.95,
            "arp_density": 0.85,
            "lead_density": 0.72,
            "bass_sustain": 0.9,
            "swing": 0.09,
        }
    if pack == "orchestral":
        return {
            "drum_level": 0.75,
            "bass_level": 0.82,
            "music_level": 1.34,
            "hat_density": 0.58,
            "shaker_density": 0.28,
            "stab_density": 1.05,
            "arp_density": 0.98,
            "lead_density": 0.88,
            "bass_sustain": 1.05,
            "swing": 0.05,
        }
    return {
        "drum_level": 1.0,
        "bass_level": 1.0,
        "music_level": 1.0,
        "hat_density": 1.0,
        "shaker_density": 0.5,
        "stab_density": 0.85,
        "arp_density": 0.72,
        "lead_density": 0.7,
        "bass_sustain": 1.0,
        "swing": 0.05,
    }

def _is_trap_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"trap", "drill"}

def _is_southern_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"southern", "dirty_south", "dirty south", "atlanta", "memphis", "crunk", "bounce"}

def _is_boom_bap_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"boom_bap", "boom bap"}

def _is_east_coast_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"east_coast", "east coast", "ny_boom_bap", "new_york"}

def _is_lofi_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"lo_fi", "lofi", "chillhop"}

def _is_west_coast_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"west_coast", "west coast", "g_funk", "gfunk"}

def _is_melodic_style(genre: str) -> bool:
    g = (genre or "").lower()
    return g in {"melodic", "melodic_rap", "melodic rap", "emo_rap", "emo rap", "sing_rap", "pop_rap"}

def _style_family(genre: str) -> str:
    if _is_southern_style(genre):
        return "southern"
    if _is_melodic_style(genre):
        return "melodic"
    if _is_trap_style(genre):
        return "trap"
    if _is_east_coast_style(genre):
        return "east_coast"
    if _is_boom_bap_style(genre):
        return "boom_bap"
    if _is_lofi_style(genre):
        return "lo_fi"
    if _is_west_coast_style(genre):
        return "west_coast"
    return "general"


