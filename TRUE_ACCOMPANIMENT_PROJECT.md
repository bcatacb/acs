# True Accompaniment Project (FlowState)

## Goal
Build a rap-focused app where a user records acapella vocals and receives an instrumental that follows the vocal performance, instead of a generic prompt-only beat.

## Current Scope Implemented
- Local-first accompaniment generation pipeline (no Suno required for the new path).
- Vocal timing extraction from uploaded audio (`.webm`/audio stream decode).
- Rule-driven drum and bass synthesis aligned to detected vocal rhythm.
- Anti-repetition rule checking with automatic rerender attempts.
- Local `.wav` beat render and secure serving through authenticated API.
- Existing UI flow preserved (Record -> Analyze -> Generate -> Play/Download).

## What Was Added
- New service: `backend/services/accompaniment_generator.py`
  - Decodes audio with PyAV
  - Detects onsets from vocal energy changes
  - Estimates BPM from onset intervals
  - Builds bar-level density profile from vocal events
  - Synthesizes drum+bass accompaniment with variation tied to vocal density
  - Scores arrangement repetition (`repetition_score`) and retries generation when score is too high
  - Exports stereo WAV in `backend/generated/`
- API integration updates in `backend/server.py`
  - `/api/projects/{id}/generate` now uses local true-accompaniment generation
  - `/api/projects/{id}/beat-status` returns local beat URLs when complete
  - `/api/projects/{id}/beat-audio` serves generated WAV file securely
- Frontend update in `frontend/src/pages/StudioPage.js`
  - Immediate status check after generation so completed local beats appear without waiting for polling

## Architecture (MVP)
1. User records acapella in browser (`MediaRecorder`, webm).
2. Backend stores upload.
3. Analyze step still gathers high-level metadata (flow/mood/BPM hint).
4. Generate step:
   - Extract vocal onsets and tempo from raw audio.
   - Create rhythm map and section intensity.
   - Render accompaniment audio (kick/snare/hats+bass).
5. Frontend receives playable/downloadable beat URL.

## Why This Is "True Accompaniment MVP"
- Beat structure is derived from the vocal timing signal (onsets + density), not only text prompts.
- Output adapts bar intensity and rhythmic placements based on the recorded performance.

## Known Limitations
- Harmonic content is currently lightweight (bass + percussive bed, no full melodic orchestration).
- Section detection is still simple and based on onset density.
- No stem export yet (single stereo render).
- Quality will improve with richer arrangement and stronger vocal feature extraction.

## Next Milestones
1. Add section-aware arrangement (verse/hook/bridge with stronger contrast).
2. Add melodic layers (chords, motifs, counter-rhythms).
3. Add anti-repetition rule checks and automatic re-render if repetitive score is high.
4. Add stem export (`drums.wav`, `bass.wav`, `melody.wav`, `full_mix.wav`).
5. Add user controls for complexity, swing, and instrument palette.

## Run Notes
- Backend now requires `av` (already added to `backend/requirements.txt`).
- Generated outputs are stored at:
  - `backend/generated/<project_id>_accompaniment.wav`
