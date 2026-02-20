import numpy as np

from services.accompaniment.analysis import (
    _build_bar_grid,
    _classify_sections_from_grid,
    _estimate_bpm,
    _safe_bpm,
)
from services.accompaniment.render import _compute_repetition_score, _render_mix


def test_safe_bpm_clamps_and_falls_back():
    assert _safe_bpm("bad", 92) == 92
    assert _safe_bpm(10, 92) == 70
    assert _safe_bpm(250, 92) == 190
    assert _safe_bpm(96.4, 92) == 96


def test_estimate_bpm_folds_octaves():
    onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)  # 120 BPM
    assert _estimate_bpm(onsets, 92) == 120

    sparse = np.array([0.0, 2.0], dtype=np.float32)
    assert _estimate_bpm(sparse, 92) == 92


def test_bar_grid_and_section_labels_are_stable():
    onsets = np.array([0.1, 0.4, 1.2, 4.1, 4.4, 8.2], dtype=np.float32)
    grid = _build_bar_grid(onsets=onsets, bars=4, bar_len=4.0, sec_per_beat=0.5)
    labels = _classify_sections_from_grid(grid)

    assert len(grid) == 4
    assert len(labels) == 4
    assert all(label in {"hook", "busy_verse", "chill_verse", "break"} for label in labels)


def test_repetition_score_range():
    dense = np.ones(16, dtype=np.float32) * 0.5
    varied = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    s1 = _compute_repetition_score(dense)
    s2 = _compute_repetition_score(varied)
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0
    assert s1 >= s2


def test_render_mix_is_deterministic_for_same_seed():
    bars = 8
    sec_per_beat = 0.5
    bar_len = sec_per_beat * 4.0
    synth_sr = 22050
    onsets = np.array([], dtype=np.float32)
    density = np.ones(bars, dtype=np.float32) * 0.3

    a = _render_mix(
        bars=bars,
        bar_len=bar_len,
        sec_per_beat=sec_per_beat,
        synth_sr=synth_sr,
        onsets=onsets,
        density=density,
        root=43.65,
        seed=1234,
        genre="trap",
        instrument_pack="auto",
    )
    b = _render_mix(
        bars=bars,
        bar_len=bar_len,
        sec_per_beat=sec_per_beat,
        synth_sr=synth_sr,
        onsets=onsets,
        density=density,
        root=43.65,
        seed=1234,
        genre="trap",
        instrument_pack="auto",
    )

    assert a["full"].shape == b["full"].shape
    assert np.array_equal(a["full"], b["full"])
