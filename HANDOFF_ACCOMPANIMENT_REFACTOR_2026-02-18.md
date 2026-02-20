# ACS Handoff: Accompaniment Refactor (2026-02-18)

## Objective Completed
Refactor the large accompaniment generator into modular components while preserving behavior, and add a regression harness for safe continued refactoring.

## What Was Changed
1. Monolith split into new package:
- `backend/services/accompaniment/common.py`
- `backend/services/accompaniment/analysis.py`
- `backend/services/accompaniment/style.py`
- `backend/services/accompaniment/assets.py`
- `backend/services/accompaniment/render.py`
- `backend/services/accompaniment/generator.py`
- `backend/services/accompaniment/__init__.py`

2. Backward-compatible wrapper preserved:
- `backend/services/accompaniment_generator.py`
  - Now re-exports: `generate_true_accompaniment` from the new package.

3. Regression tests added:
- `backend/tests/test_accompaniment_refactor.py`
  - Tests `_safe_bpm`, `_estimate_bpm`, bar grid + section classification, repetition score bounds, and deterministic `_render_mix` output for fixed seed.

## Current Status
- Refactor compiles successfully:
  - `python -m py_compile` passed for all new modules and new test file.
- Smoke run of `_render_mix` works from `backend/`.
- `pytest` not runnable yet in this environment because `pytest` is not installed in active Python interpreter.

## Blocking Issue
- Missing test runner dependency in runtime env:
  - Error seen: `No module named pytest`

## Immediate Next Steps (for next engineer)
1. Set up backend venv and dependencies:
```powershell
cd C:\Users\OGTommyP\Desktop\acs\app\backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_accompaniment_refactor.py
```

2. If tests pass, continue modularization of render layer:
- Split `backend/services/accompaniment/render.py` into:
  - `backend/services/accompaniment/dsp.py`
  - `backend/services/accompaniment/arrangement.py`
- Keep function signatures unchanged while moving code.

3. Add one integration test around top-level API function:
- Target: `generate_true_accompaniment(...)` returns expected keys and writes wav/json outputs.
- Use deterministic seed inputs and short synthetic audio fixture.

## Git/Workspace Notes
- Relevant changed paths:
  - `backend/services/accompaniment_generator.py` (modified)
  - `backend/services/accompaniment/*` (new)
  - `backend/tests/test_accompaniment_refactor.py` (new)
- Unrelated pre-existing local changes detected (left untouched):
  - `backend/.env`
  - `frontend/.env`

## Risk Notes
- `render.py` is still large; biggest remaining maintainability risk.
- Behavior should be preserved, but full confidence requires pytest run in configured venv.
- Asset/MIDI loading paths depend on local `asset_drop` and SQLite index availability.

## Done/Not Done
- Done: structural split + compatibility wrapper + baseline regression tests.
- Not done: execute pytest in proper environment, deeper render sub-split, API-level integration test.
