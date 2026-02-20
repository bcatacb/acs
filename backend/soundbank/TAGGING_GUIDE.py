"""
Sound Bank Tagging System - Usage Guide
=======================================
A complete example of how to use the intelligent tagging system
for building and querying a scalable sound bank.

This system automatically analyzes audio during ingestion and assigns
relevant tags for smart, cross-genre asset discovery.
"""

# ============================================================================
# EXAMPLE 1: Ingest samples with automatic tagging
# ============================================================================

from soundbank.ingest import IngestionEngine

# Create an ingestion engine
engine = IngestionEngine(
    output_dir="./output",
    target_rms=0.1,
    target_peak=0.95,
    notch_attenuation_db=-12.0
)

# Ingest drum samples - they will be automatically classified and tagged
processed, failed = engine.process_directory(
    input_dir="./asset_drop/Instruments/Drums/Kicks",
    category="808",
    apply_notch=True,
    normalize_mode="rms",
    recursive=False
)

print(f"Processed {processed} files, {failed} failed")
# Output during ingestion will show:
# Processing [1/3]: kick_01.wav
#   Resampled: 48000Hz -> 44100Hz
#   Applied notch filter: -12.0dB @ 1-3kHz
#   Normalized RMS: 0.2341 -> 0.1000
#   Tags: 808-kick, kick, bass, low-energy, percussive, tight
#
# Processing [2/3]: kick_02.wav
#   Tags: acoustic-kick, kick, bass, fast-attack, warm, organic
# etc...


# ============================================================================
# EXAMPLE 2: Query by intensity + tags
# ============================================================================

from soundbank.provider import SoundBankProvider

provider = SoundBankProvider(
    master_wav_path="./output/master_bank.wav",
    db_path="./output/bank.db"
)

# Get a "warm" kick that matches vocal intensity of 0.8
asset, audio = provider.get_by_tag(
    tag="warm",
    intensity_target=0.8,
    category="808",
    limit=10
)

print(f"Selected: {asset.original_filename}")
print(f"Tags: {provider.get_asset_tags(asset.id)}")
# Output:
# Selected: kick_soft_warm.wav
# Tags: ['acoustic-kick', 'kick', 'warm', 'organic', 'smooth', 'fast-attack', 'low-energy']


# ============================================================================
# EXAMPLE 3: Search for assets matching multiple characteristics
# ============================================================================

# Find hi-hats that are "punchy" and "crispy" suitable for trap
results = provider.search_by_characteristics(
    characteristics=["punchy", "crispy", "trap"],
    intensity_target=0.6,
    category="snare",
    limit=5
)

for asset, audio in results:
    tags = provider.get_asset_tags(asset.id)
    print(f"{asset.original_filename}: {', '.join(tags)}")


# ============================================================================
# EXAMPLE 4: Get all assets with a specific tag
# ============================================================================

from soundbank.database import SoundBankDB

db = SoundBankDB("./output/bank.db")
with db:
    db.connect()
    
    # Get all lofi-friendly samples
    lofi_samples = db.get_assets_by_tag("lofi-friendly", limit=100)
    
    for asset in lofi_samples:
        tags = db.get_asset_tags(asset.id)
        print(f"{asset.original_filename}: Intensity={asset.intensity_score:.2f}")
        print(f"  Tags: {', '.join(tags)}")


# ============================================================================
# EXAMPLE 5: Find assets matching ALL specified tags (strict matching)
# ============================================================================

# Find samples that are BOTH "punchy" AND "melodic-friendly"
strict_results = db.get_assets_by_tags(
    tags=["punchy", "melodic-friendly"],
    match_all=True,  # Must have BOTH tags
    limit=10
)

print(f"Found {len(strict_results)} assets that are punchy AND melodic-friendly")


# ============================================================================
# EXAMPLE 6: Available tag categories
# ============================================================================

TAG_CATEGORIES = {
    'instrument': [
        'kick', 'snare', 'hi-hat', 'clap', 'tom', 'percussion', 'bass',
        'synth', 'pad', 'pluck', 'horn', 'strings', 'vocal', 'piano', 'guitar'
    ],
    'drum_type': [
        '808-kick', 'acoustic-kick', 'synth-kick', 'closed-hat', 'open-hat',
        'kick-roll', 'snare-roll'
    ],
    'genre': [
        'hip-hop', 'trap', 'boom-bap', 'lofi', 'grime', 'house', 'techno',
        'dubstep', 'future-bass', 'edm', 'dnb', 'uk-garage', 'rnb',
        'neo-soul', 'pop', 'synthpop'
    ],
    'characteristic': [
        'punchy', 'warm', 'bright', 'dark', 'metallic', 'organic', 'digital',
        'crispy', 'muddy', 'thin', 'resonant', 'tight', 'loose', 'filtered',
        'aggressive', 'smooth', 'colored'
    ],
    'frequency': [
        'sub-bass', 'bass', 'low-mid', 'mid', 'high-mid', 'treble',
        'wide-spectrum', 'narrow-band'
    ],
    'envelope': [
        'fast-attack', 'slow-attack', 'short-decay', 'long-decay',
        'percussive', 'sustained', 'plucked', 'pad-like'
    ],
    'use_case': [
        'melodic-friendly', 'drums-only', 'vocal-carrier', 'loop-able',
        'glitchy', 'cinematic', 'minimalist', 'texture', 'ambient',
        'upfront'
    ],
    'intensity': [
        'low-energy', 'medium-energy', 'high-energy', 'explosive'
    ]
}

# Example: Find all "trap" + "melodic" samples
trap_melodic = provider.get_by_tags(
    tags=["trap", "melodic-friendly"],
    match_all=False,  # ANY of these tags
    limit=20
)


# ============================================================================
# EXAMPLE 7: Create a curated kit using tags
# ============================================================================

# Build a "lo-fi hip-hop production kit"
lofi_kit = {}

lofi_kit['kicks'] = [
    asset for asset in db.get_assets_by_tags(
        ['lofi', 'kick'],
        match_all=False,
        limit=5
    )
]

lofi_kit['snares'] = [
    asset for asset in db.get_assets_by_tags(
        ['lofi', 'snare'],
        match_all=False,
        limit=5
    )
]

lofi_kit['hi_hats'] = [
    asset for asset in db.get_assets_by_tags(
        ['lofi', 'hi-hat'],
        match_all=False,
        limit=5
    )
]

lofi_kit['pads'] = [
    asset for asset in db.get_assets_by_tags(
        ['ambient', 'texture', 'warm'],
        match_all=False,
        limit=5
    )
]

print(f"Lo-fi Kit Contents:")
print(f"  Kicks: {len(lofi_kit['kicks'])}")
print(f"  Snares: {len(lofi_kit['snares'])}")
print(f"  Hi-Hats: {len(lofi_kit['hi_hats'])}")
print(f"  Pads: {len(lofi_kit['pads'])}")


# ============================================================================
# EXAMPLE 8: Advanced queries using the classifier directly
# ============================================================================

from soundbank.classifier import AudioClassifier
import soundfile as sf

# Classify an audio file to see what tags would be assigned
classifier = AudioClassifier(sr=44100)
audio, sr = sf.read("drum_sample.wav")

tags = classifier.classify(audio, sample_rate=sr)
print(f"Auto-detected tags: {tags}")

# Then you can manually add more tags to the database
db.add_tags_batch(asset_id=42, tags=tags, confidence=0.95)
db.add_tag(asset_id=42, tag_name="custom-tag", confidence=1.0)


# ============================================================================
# EXAMPLE 9: Export a tagged subset as individual files
# ============================================================================

# Export all "punchy" + "hi-hat" samples
export_assets = db.get_assets_by_tags(
    tags=["punchy", "hi-hat"],
    match_all=False,
    limit=10
)

for asset in export_assets:
    audio = provider.get_audio_by_id(asset.id)
    output_file = f"exports/{asset.original_filename}"
    provider.export_asset(asset.id, output_file)
    print(f"Exported: {output_file}")


# ============================================================================
# EXAMPLE 10: Database statistics and introspection
# ============================================================================

stats = provider.get_statistics()
print(f"Sound Bank Statistics:")
print(f"  Total Assets: {stats['total_assets']}")
print(f"  By Category: {stats['by_category']}")
print(f"  Intensity Range: {stats['intensity_range']}")
print(f"  Master Duration: {stats['master_duration_seconds']:.1f} seconds")

# List all available tags
with db:
    db.connect()
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT category, COUNT(*) as count 
        FROM tag_definitions 
        GROUP BY category
    """)
    print(f"\nAvailable Tags by Category:")
    for row in cursor.fetchall():
        print(f"  {row['category']}: {row['count']} tags")


# ============================================================================
# WORKFLOW: Complete Ingestion + Querying Pipeline
# ============================================================================

"""
STEP 1: Organize folder structure
asset_drop/
├── Instruments/
│   ├── Drums/
│   │   ├── Kicks/
│   │   │   ├── 808_01.wav
│   │   │   ├── 808_02.wav
│   │   │   └── acoustic_01.wav
│   │   ├── Snares/
│   │   │   ├── snare_tight_01.wav
│   │   │   └── snare_bright_01.wav
│   │   └── Hi_Hats/
│   │       ├── closed_hat_01.wav
│   │       └── open_hat_01.wav
│   └── Bass/
│       ├── synth_bass_01.wav
│       └── sub_bass_01.wav
├── Genre_Kits/
│   ├── Hip_Hop/
│   │   ├── Boom_Bap/
│   │   └── Trap/
│   └── EDM/
│       └── House/
└── MIDI/
    └── Drum_Patterns/

STEP 2: Ingest categories one at a time
python -m soundbank.ingest ./asset_drop/Instruments/Drums/Kicks --output ./output --category 808
python -m soundbank.ingest ./asset_drop/Instruments/Drums/Snares --output ./output --category snare
python -m soundbank.ingest ./asset_drop/Instruments/Drums/Hi_Hats --output ./output --category snare

STEP 3: Query intelligently
- Find trap-suitable hi-hats
- Find melodic-friendly pads
- Build genre-specific kits
- Create intensity-matched collections

All tags are automatically assigned during ingest based on audio analysis.
You can add manual tags or trust the classifier.
"""

print(__doc__)
