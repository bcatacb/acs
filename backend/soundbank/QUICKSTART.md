#!/usr/bin/env python3
"""
SOUND BANK TAGGING SYSTEM - QUICK START GUIDE
==============================================

Your system is ready. Here's how to use it.
"""

# =============================================================================
# STEP 1: ORGANIZE YOUR SAMPLES
# =============================================================================

"""
Create a folder structure. Pick ONE of these approaches:

APPROACH A: Instrument-Based (Recommended for growth)
──────────────────────────────────────────────────
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
│   ├── Bass/
│   │   ├── 808_bass.wav
│   │   └── synth_bass.wav
│   └── ...more instruments
├── Genre_Kits/
│   ├── Hip_Hop/
│   └── EDM/
└── Loops/


APPROACH B: Genre-First (Recommended for production)
──────────────────────────────────────────────────
asset_drop/
├── Hip_Hop/
│   ├── Boom_Bap/
│   ├── Trap/
│   └── Lofi/
├── EDM/
│   ├── House/
│   ├── Techno/
│   └── Dubstep/
└── R&B/


APPROACH C: Intensity-Based (Recommended for vocal matching)
──────────────────────────────────────────────────────────
asset_drop/
├── Drums/
│   ├── Low_Energy/
│   ├── Medium_Energy/
│   └── High_Energy/
├── Bass/
│   └── [same structure]
└── Pads/
    └── [same structure]


IMPORTANT: It doesn't matter which structure you choose!
The tagging system bridges all approaches.
"""

# =============================================================================
# STEP 2: INGEST YOUR SAMPLES
# =============================================================================

"""
From PowerShell in your project root:

BASIC INGESTION
───────────────

# Single category
python -m soundbank.ingest ./asset_drop/Instruments/Drums/Kicks \\
  --output ./output \\
  --category 808

# With custom RMS level (default 0.1, range 0.05-0.3)
python -m soundbank.ingest ./asset_drop/Instruments/Drums/Snares \\
  --output ./output \\
  --category snare \\
  --target-rms 0.15

# Disable spectral notch filter if not needed
python -m soundbank.ingest ./asset_drop/Instruments/Bass \\
  --output ./output \\
  --category loops \\
  --no-notch

# Non-recursive (only process files in direct folder)
python -m soundbank.ingest ./asset_drop/Loops \\
  --output ./output \\
  --category loops \\
  --no-recursive


WHAT HAPPENS
────────────
For each file, you'll see:
  Processing [1/10]: kick_808_01.wav
    Resampled: 48000Hz -> 44100Hz
    Applied notch filter: -12.0dB @ 1-3kHz
    Normalized RMS: 0.2315 -> 0.1000
    Tags: 808-kick, kick, bass, percussion, tight, fast-attack, ...

After ingestion:
  ✓ master_bank.wav created (concatenated audio)
  ✓ bank.db created (index with tags)
  ✓ All tags automatically assigned
"""

# =============================================================================
# STEP 3: QUERY YOUR SAMPLES
# =============================================================================

"""
In your Python code (ACS generator.py or anywhere):

BASIC QUERIES
─────────────

from soundbank.provider import SoundBankProvider

provider = SoundBankProvider(
    master_wav_path='./output/master_bank.wav',
    db_path='./output/bank.db'
)


# Get ANY sample with a specific tag
asset, audio = provider.get_by_tag('warm')

# Get a sample with tag + intensity match
asset, audio = provider.get_by_tag(
    'punchy',
    intensity_target=0.7,  # 0.0-1.0 scale
    limit=5  # Try 5 candidates
)

# Get samples matching MULTIPLE tags
results = provider.get_by_tags(
    tags=['lofi', 'melodic-friendly'],
    intensity_target=0.5,
    limit=10
)

# Returns list of (asset, audio) tuples
for asset, audio in results:
    print(f"Using: {asset.original_filename}")
    # audio is ready to use


CHARACTERISTIC SEARCH
─────────────────────

# Find "punchy" + "crispy" samples
punchy_crispy = provider.search_by_characteristics(
    characteristics=['punchy', 'crispy'],
    intensity_target=0.8,
    limit=20
)

for asset, audio in punchy_crispy:
    tags = provider.get_asset_tags(asset.id)
    print(f"{asset.original_filename}")
    print(f"  Tags: {tags}")
    print(f"  Intensity: {asset.intensity_score:.2f}")


ADVANCED: MANUAL DATABASE ACCESS
──────────────────────────────────

from soundbank.database import SoundBankDB

db = SoundBankDB('./output/bank.db')

# Get all samples with tag
with db:
    samples = db.get_assets_by_tag('trap', limit=100)
    for asset in samples:
        tags = db.get_asset_tags(asset.id)
        print(f"{asset.original_filename}: {tags}")

# Get samples matching ALL tags (strict)
with db:
    strict_results = db.get_assets_by_tags(
        tags=['punchy', 'melodic-friendly'],
        match_all=True,  # MUST have both
        limit=20
    )

# Add manual tags if needed
with db:
    db.add_tag(asset_id=5, tag_name='custom-tag', confidence=0.9)
"""

# =============================================================================
# STEP 4: ACS INTEGRATION
# =============================================================================

"""
In your ACS accompaniment generator:

# At the top of your generator.py
from soundbank.provider import SoundBankProvider

# Initialize once
provider = SoundBankProvider(
    master_wav_path='./output/master_bank.wav',
    db_path='./output/bank.db'
)

# In your generate function, after vocal analysis:

def generate_accompaniment(vocal_audio, vocal_intensity):
    '''
    vocal_intensity: 0.0 (quiet) to 1.0 (loud)
    '''
    
    # Get intensity-matched loops from Sound Bank
    try:
        loop_asset, loop_audio = provider.get_by_tag(
            'loops',
            intensity_target=vocal_intensity,
            limit=5
        )
        
        # Use loop for accompaniment
        # ...
        
        return {
            'soundbank_used': True,
            'loop_source': loop_asset.original_filename,
            'loop_tags': provider.get_asset_tags(loop_asset.id),
            'loop_intensity': loop_asset.intensity_score,
        }
        
    except Exception as e:
        # Fallback to folder-based loops
        print(f"Sound Bank error: {e}. Using folder catalog.")
        return folder_based_accompaniment()
"""

# =============================================================================
# STEP 5: AVAILABLE TAGS REFERENCE
# =============================================================================

"""
Auto-detected tags fall into categories:

INSTRUMENTS (15 total)
──────────────────────
kick, snare, hi-hat, clap, tom, percussion, bass, synth, pad, pluck,
horn, strings, vocal, piano, guitar

DRUM SPECIFICS (7 total)
────────────────────────
808-kick, acoustic-kick, synth-kick, closed-hat, open-hat, kick-roll,
snare-roll

GENRES (15+ total)
──────────────────
hip-hop, trap, boom-bap, lofi, grime, house, techno, dubstep,
future-bass, edm, dnb, uk-garage, rnb, neo-soul, pop, synthpop

SONIC CHARACTERISTICS (20+ total)
──────────────────────────────────
punchy, warm, bright, dark, metallic, organic, digital, crispy, muddy,
thin, resonant, tight, loose, filtered, aggressive, smooth, colored,
glitchy

FREQUENCY FOCUS (8 total)
─────────────────────────
sub-bass (20-60Hz), bass (60-250Hz), low-mid (250-500Hz),
mid (500Hz-2kHz), high-mid (2-5kHz), treble (5-20kHz),
wide-spectrum, narrow-band

ENVELOPE TYPES (8 total)
────────────────────────
fast-attack, slow-attack, short-decay, long-decay, percussive,
sustained, plucked, pad-like

USE CASES (10+ total)
─────────────────────
melodic-friendly, drums-only, vocal-carrier, loop-able, glitchy,
cinematic, minimalist, texture, ambient, upfront

ENERGY LEVELS (4 total)
───────────────────────
low-energy, medium-energy, high-energy, explosive


EXAMPLE QUERIES
───────────────

# Get warm hi-hats suitable for lofi
lofi_hats = provider.get_by_tags(['lofi', 'warm', 'hi-hat'], limit=10)

# Get all punchy + tight samples
punchy = provider.search_by_characteristics(['punchy', 'tight'], limit=20)

# Get melodic-friendly pads
pads = provider.get_by_tags(['melodic-friendly', 'pad'], limit=10)

# Get everything NOT drums
non_drums = [
    a for a in db.get_all_assets()
    if 'drums-only' not in db.get_asset_tags(a.id)
]
"""

# =============================================================================
# STEP 6: EXAMPLE WORKFLOWS
# =============================================================================

"""
WORKFLOW 1: Build a Trap Production Kit
────────────────────────────────────────

from soundbank.database import SoundBankDB

db = SoundBankDB('./output/bank.db')

trap_kit = {}

with db:
    trap_kit['kicks'] = db.get_assets_by_tags(
        ['trap', 'kick'],
        match_all=False,
        limit=10
    )
    
    trap_kit['snares'] = db.get_assets_by_tags(
        ['trap', 'snare'],
        match_all=False,
        limit=10
    )
    
    trap_kit['hats'] = db.get_assets_by_tags(
        ['trap', 'hi-hat'],
        match_all=False,
        limit=10
    )
    
    trap_kit['bass'] = db.get_assets_by_tags(
        ['trap', 'bass'],
        match_all=False,
        limit=5
    )
    
    trap_kit['pads'] = db.get_assets_by_tags(
        ['ambient', 'warm'],
        match_all=False,
        limit=5
    )

# trap_kit['kicks'] = [asset1, asset2, asset3, ...]
# trap_kit['snares'] = [asset1, asset2, ...]
# etc.


WORKFLOW 2: ACS Vocal-Matched Accompaniment
─────────────────────────────────────────────

def generate_from_vocal(vocal_audio):
    # Analyze vocal
    energy = analyze_vocal_density(vocal_audio)  # Returns 0.0-1.0
    
    # Find matching companion loop
    try:
        loop_asset, loop_audio = provider.get_by_tag(
            'loops',
            intensity_target=energy,
            limit=5
        )
    except:
        loop_asset, loop_audio = provider.get_by_tag('loops')
    
    # Find complementary pad (lower energy)
    try:
        pad_asset, pad_audio = provider.get_by_tag(
            'pad',
            intensity_target=energy * 0.6,  # Pad is quieter
            limit=5
        )
    except:
        pad_asset, pad_audio = provider.get_by_tag('ambient')
    
    # Mix them
    return mix_audio(vocal_audio, loop_audio, pad_audio)


WORKFLOW 3: Intensity-Driven Selection
──────────────────────────────────────

# Low energy section (0.2-0.3)
sparse_assets = provider.get_by_tags(
    ['low-energy'],
    intensity_target=0.25,
    limit=10
)

# Medium energy section (0.4-0.6)
balanced_assets = provider.get_by_tags(
    ['medium-energy'],
    intensity_target=0.5,
    limit=10
)

# High energy drop (0.8-1.0)
explosive_assets = provider.get_by_tags(
    ['explosive', 'high-energy'],
    intensity_target=0.9,
    limit=10
)
"""

# =============================================================================
# STEP 7: COMMON QUESTIONS
# =============================================================================

"""
Q: What if my samples aren't tagged correctly?
A: Manual tags override auto-detected ones:
   db.add_tag(asset_id=5, tag_name='correct-tag', confidence=1.0)

Q: Can I query by intensity without any tags?
A: Yes! Intensity is stored separately from tags:
   assets = db.get_by_intensity(target_rms=0.15, tolerance=0.05)

Q: How do I export individual samples?
A: provider.export_asset(asset_id=5, output_path='./export.wav')

Q: Can I see all tags for an asset?
A: tags = provider.get_asset_tags(asset_id=5)

Q: What's the difference between my samples and those in the ACS?
A: ACS was manually built. This system is automated to scale.
   They can be used together or separately.

Q: Should I ingest everything at once?
A: No. Ingest by category:
   • Category 808 (all 808 kicks)
   • Category snare (all snares/claps)
   • Category loops (all loops)
   This keeps organization clean.

Q: Can I re-ingest and update?
A: Clear the database first and start fresh:
   rm ./output/bank.db ./output/master_bank.wav
   Then re-ingest.
"""

# =============================================================================
# STEP 8: FILE LOCATIONS
# =============================================================================

"""
Sound Bank files live in:

/app/backend/soundbank/
├── database.py              ← Tag system core
├── classifier.py            ← Auto-detection engine (NEW)
├── ingest.py                ← Ingestion with tagging (UPDATED)
├── provider.py              ← Queries with tags (UPDATED)
├── ARCHITECTURE.md          ← Full system design
├── TAGGING_GUIDE.py         ← Code examples
└── IMPLEMENTATION_SUMMARY.md ← This feature overview

/ACS/backend/soundbank/ 
└── [All files synced from /app/backend]

Output location (after ingestion):
/app/backend/output/
├── master_bank.wav          ← Concatenated audio
├── bank.db                  ← SQLite index + tags
└── default/
    └── [other outputs]
"""

print(__doc__)

# =============================================================================
# YOU'RE READY! 
# =============================================================================

"""
1. ✅ Organize samples
2. ✅ Run ingestion (tags auto-detect)
3. ✅ Query using tags + intensity
4. ✅ Build kits, workflows, ACS integrations

Your system is production-ready and scales to thousands of assets.

Next: See ARCHITECTURE.md for deep dive, or TAGGING_GUIDE.py for examples.
"""
