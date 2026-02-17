# Starter Sampler Pack v1

Drop your `.wav` files into:
- `samples/keys_main/`
- `samples/pad_main/`
- `samples/lead_main/`

Then update `manifest.json` file paths and MIDI ranges to match your recordings.

## Quick rules
- WAV only (`16-bit` or `24-bit`)
- Preferred sample rate: `44100 Hz`
- Clean note starts, no clipping
- Keep tails natural (don't hard-cut too early)

## Minimum viable setup
- `keys_main`: at least one sustained note sample
- `pad_main`: at least one sustained note sample
- `lead_main`: at least one short or sustained note sample

## Better sounding setup
- Record multiple root notes (ex: C2, F2, A2, C3...)
- Add velocity layers (soft/hard)
- Add round-robin takes for repeated notes
