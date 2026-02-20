# Proprietary Sound Bank Builder

A standalone Python utility that transforms raw audio samples into a single, addressable Master WAV-Container and a Relational Index (SQLite database).

## Features

- **Spectral Notch Filter**: Permanently carves out the 1kHz-3kHz "vocal pocket" to ensure beats sit behind vocals
- **Normalization**: Level all assets to consistent Peak/RMS standards
- **Bit-Crushing**: Lo-fi effect for unique sound character
- **Time-Stretching**: Phase vocoder implementation for tempo-independent stretching
- **Intensity-Based Retrieval**: Query audio by RMS energy level
- **Sample-Accurate Slicing**: High-speed seeking without loading entire files into RAM

## Installation

```bash
cd /app/backend
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Test Samples

```bash
# Generate 10 white-noise bursts for testing
cd /app/backend
python -m soundbank generate -o ./test_samples -n 10

# Or generate category-organized samples
python -m soundbank generate -o ./test_samples --categories -n 5
```

### 2. Ingest Samples into Sound Bank

```bash
# Process samples into master_bank.wav + bank.db
python -m soundbank ingest ./test_samples -o ./output -c loops

# Process specific category
python -m soundbank ingest ./test_samples/808 -o ./output -c 808
```

### 3. Query the Sound Bank

```bash
# View sound bank info
python -m soundbank info ./output

# Query by intensity
python -m soundbank query ./output --intensity 0.15 --category loops

# List all assets
python -m soundbank query ./output
```

### 4. Verify Sample Accuracy

```bash
python -m soundbank verify ./output
```

## CLI Reference

### `soundbank ingest`
Process audio samples into Master WAV container.

```bash
python -m soundbank ingest <input_dir> [options]

Options:
  -o, --output DIR       Output directory (default: ./output)
  -c, --category CAT     Category: 808, snare, loops, atmospheres
  --normalize MODE       rms, peak, or none (default: rms)
  --target-rms FLOAT     Target RMS level (default: 0.1)
  --notch-db FLOAT       Notch attenuation in dB (default: -12.0)
  --no-notch             Skip spectral notch filter
  --no-recursive         Don't search subdirectories
```

### `soundbank generate`
Generate test samples for validation.

```bash
python -m soundbank generate [options]

Options:
  -o, --output DIR       Output directory (default: ./test_samples)
  -n, --count INT        Number of samples (default: 10)
  --duration FLOAT       Base duration in seconds (default: 0.5)
  --categories           Organize samples by category
  --no-markers           Skip boundary markers
```

### `soundbank transform`
Apply transformations to audio files.

```bash
python -m soundbank transform <input> <output> -t <transform>

Transforms:
  notch      Apply vocal pocket notch filter
  bitcrush   Apply bit-crushing effect
  stretch    Apply time-stretching
```

## Python API Usage

### Provider (Retrieval API)

```python
from soundbank import SoundBankProvider

# Initialize provider
provider = SoundBankProvider('output/master_bank.wav', 'output/bank.db')

# Get audio by intensity (matches vocal energy)
audio = provider.get_by_intensity(target_rms=0.15, category='808')

# Get specific slice by ID
audio = provider.request_slice(asset_id=5)

# Get all matches with metadata
results = provider.get_by_intensity(
    target_rms=0.15,
    tolerance=0.1,
    return_all_matches=True
)
for asset, audio in results:
    print(f"Asset {asset.id}: {asset.original_filename}")
```

### Transformations

```python
from soundbank import Transformations
import soundfile as sf

# Load audio
audio, sr = sf.read('sample.wav')

# Apply spectral notch (vocal pocket carving)
processed, spectral_hole = Transformations.apply_spectral_notch(
    audio, sr, attenuation_db=-12.0
)

# Apply bit-crushing
crushed = Transformations.bit_crush(audio, bit_depth=8)

# Apply time-stretching
stretched = Transformations.time_stretch(audio, stretch_factor=1.5)

# Calculate metrics
rms = Transformations.calculate_rms_energy(audio)
mid_density = Transformations.calculate_mid_range_density(audio, sr)
```

## Adding Custom Transformations

You can extend the Transformations class with custom effects:

```python
from soundbank.transformations import add_transformation_module
import numpy as np

# Define your transformation
def wavefold(audio, threshold=0.5):
    """Wavefolder distortion effect."""
    folded = audio.copy()
    while np.any(np.abs(folded) > threshold):
        folded = np.where(
            folded > threshold,
            2 * threshold - folded,
            folded
        )
        folded = np.where(
            folded < -threshold,
            -2 * threshold - folded,
            folded
        )
    return folded

# Register the transformation
add_transformation_module(
    'wavefold',
    wavefold,
    'Wavefolder distortion for harmonic saturation'
)

# Now use it
from soundbank import Transformations
processed = Transformations.wavefold(audio, threshold=0.3)
```

### Example: Ring Modulation

```python
def ring_mod(audio, freq=440, sample_rate=44100):
    """Ring modulation effect."""
    t = np.arange(len(audio)) / sample_rate
    carrier = np.sin(2 * np.pi * freq * t)
    return audio * carrier

add_transformation_module('ring_mod', ring_mod, 'Ring modulation effect')
```

### Example: Saturation

```python
def soft_clip(audio, drive=2.0):
    """Soft clipping saturation."""
    return np.tanh(audio * drive) / np.tanh(drive)

add_transformation_module('soft_clip', soft_clip, 'Soft clipping saturation')
```

## Database Schema

The `bank.db` SQLite database contains:

### `assets` table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| original_filename | TEXT | Source file name |
| start_sample | INTEGER | Start offset in master WAV |
| end_sample | INTEGER | End offset in master WAV |
| category | TEXT | Asset category |
| intensity_score | REAL | RMS energy for matching |
| spectral_hole | REAL | Depth of 1k-3k notch |
| rms_energy | REAL | Calculated RMS |
| mid_range_density | REAL | Mid-frequency content |
| sample_rate | INTEGER | Sample rate (44100) |
| duration_samples | INTEGER | Length in samples |

### `bank_metadata` table
| Column | Type | Description |
|--------|------|-------------|
| key | TEXT | Metadata key |
| value | TEXT | Metadata value |

## File Structure

```
output/
├── master_bank.wav    # Concatenated audio container
└── bank.db            # SQLite index database
```

## Portability

The system only needs these two files to function:
- `master_bank.wav` - The audio container
- `bank.db` - The index database

Copy these files to any system with Python and the required libraries to use the sound bank.

## Categories

| Category | Description |
|----------|-------------|
| 808 | Bass drums, sub-bass |
| snare | Snare drums, claps |
| loops | Rhythmic loops, patterns |
| atmospheres | Pads, textures, ambience |

## Requirements

- Python 3.9+
- NumPy
- SciPy
- SoundFile
- librosa (optional, for advanced analysis)
