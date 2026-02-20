"""
SOUND BANK INTELLIGENT TAGGING SYSTEM
======================================
A scalable, production-ready architecture for organizing and discovering
audio assets based on automated analysis and cross-genre applicability.

Built for growth - designed to handle thousands of assets with intelligent
cross-referencing and genre-agnostic discovery.
"""

# =============================================================================
# ARCHITECTURE OVERVIEW
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SOUND BANK TAGGING SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────┐
                            │  Raw     │
                            │  Audio   │
                            │  Files   │
                            └────┬─────┘
                                 │
                    ┌────────────▼────────────┐
                    │  IngestionEngine        │
                    │  (ingest.py)            │
                    │                         │
                    │ • Resample to 44.1kHz  │
                    │ • Apply spectral notch │
                    │ • Normalize RMS energy │
                    │ • Calculate metrics    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  AudioClassifier       │
                    │  (classifier.py)       │
                    │                        │
                    │ • Detect instruments   │
                    │ • Analyze frequency    │
                    │ • Classify envelope    │
                    │ • Assess characteristics
                    │ • Measure energy level │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  SoundBankDB            │
                    │  (database.py)          │
                    │                         │
                    │ tables:                 │
                    │ • assets                │
                    │ • tag_definitions       │
                    │ • asset_tags (M2M)      │
                    │ • bank_metadata         │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────▼────┐         ┌──────▼───────┐      ┌──────▼──────┐
    │master_   │         │  Provider    │      │  Database   │
    │bank.wav  │         │  (provider.py)      │  Index      │
    │          │         │              │      │   bank.db   │
    │• 24-bit  │         │ • Intensity  │      │             │
    │• PCM     │         │   queries    │      │ Tags:       │
    │• Concat. │         │ • Tag-based  │      │ • Instrument│
    │  audio   │         │   retrieval  │      │ • Genre     │
    └──────────┘         │ • Spectral   │      │ • Sonic     │
                         │   analysis   │      │ • Envelope  │
                         └──────────────┘      │ • Use case  │
                                               └─────────────┘

FLOW:
1. User uploads audio files organized in folders
2. IngestionEngine processes each file:
   - Resample to uniform 44.1kHz
   - Apply spectral notch filter (vocal pocket preservation)
   - Normalize RMS levels for consistency
   - Calculate spectral/envelope metrics
3. AudioClassifier analyzes the processed audio:
   - Detects instrument type (kick, snare, hi-hat, etc.)
   - Measures frequency distribution (bass, mid, treble focus)
   - Analyzes envelope (fast/slow attack, decay type)
   - Detects sonic characteristics (punchy, warm, bright, etc.)
   - Measures energy/density level
4. Tags are stored in database (asset_tags M2M table)
5. SoundBankProvider enables intelligent queries:
   - Get by intensity + category + tags
   - Get by multiple tags (AND / OR logic)
   - Get by sonic characteristics
   - Full-text metadata search
"""

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

"""
TABLE: assets
─────────────
Stores core audio metadata and master WAV references.

Columns:
  id                    INTEGER PRIMARY KEY
  original_filename     TEXT          (source filename)
  start_sample          INTEGER       (offset in master_bank.wav)
  end_sample            INTEGER       (offset + duration)
  category              TEXT          (808, snare, loops, atmospheres)
  intensity_score       REAL          (RMS energy: 0.0-1.0 normalized)
  spectral_hole         REAL          (notch filter depth)
  rms_energy            REAL          (raw RMS value)
  mid_range_density     REAL          (energy in 500Hz-3kHz band)
  sample_rate           INTEGER       (always 44100)
  duration_samples      INTEGER       (end - start)
  created_at            TIMESTAMP     (ingestion time)

Indexes:
  idx_intensity         - Fast intensity-based queries
  idx_category          - Fast category filtering
  idx_category_intensity - Combined queries
  

TABLE: tag_definitions
──────────────────────
Curated set of all possible tags (controlled vocabulary).

Columns:
  id                    INTEGER PRIMARY KEY
  tag_name              TEXT UNIQUE   (e.g., "punchy", "trap", "kick")
  category              TEXT          (instrument, genre, characteristic, etc.)
  description           TEXT          (what this tag means)
  created_at            TIMESTAMP

Tag Categories:
  instrument          - Instrument type (kick, snare, hi-hat, bass, synth, etc.)
  drum_type           - Drum specifics (808-kick, acoustic-kick, synth-kick, etc.)
  genre               - Music genre (hip-hop, trap, house, dubstep, edm, etc.)
  characteristic      - Sonic quality (punchy, warm, bright, dark, metallic, etc.)
  frequency           - Frequency focus (sub-bass, bass, mid, treble, etc.)
  envelope            - Attack/decay type (fast-attack, slow-attack, sustained, etc.)
  use_case            - Application (melodic-friendly, vocal-carrier, ambient, etc.)
  intensity           - Energy level (low-energy, medium-energy, high-energy, etc.)


TABLE: asset_tags (Many-to-Many Junction)
──────────────────────────────────────────
Maps assets to tags (allows 1 asset → N tags relationship).

Columns:
  asset_id              INTEGER NOT NULL (FK → assets.id)
  tag_id                INTEGER NOT NULL (FK → tag_definitions.id)
  confidence            REAL            (1.0 = auto-detected by classifier)
                                        (< 1.0 = manual tag with lower confidence)
  PRIMARY KEY           (asset_id, tag_id)

Indexes:
  idx_asset_tags        - Fast tag-based lookups
  

TABLE: bank_metadata
────────────────────
Key-value store for bank-wide settings.

Columns:
  key                   TEXT PRIMARY KEY
  value                 TEXT

Example entries:
  'created_at'          - ISO timestamp of bank creation
  'sample_rate'         - Target sample rate (44100)
  'notch_attenuation_db' - Spectral notch depth (-12.0)
  'total_duration'      - Seconds of audio
"""

# =============================================================================
# TAG SYSTEM DESIGN
# =============================================================================

"""
WHY TAGS INSTEAD OF CATEGORIES?

PROBLEM with simple categories:
  • A drum kit can be used in hip-hop, lofi, edm, boom-bap, etc.
  • A single snare might be suitable for trap, house, AND synthpop
  • Hard to scale - new categories needed constantly
  • Cross-genre discovery requires manual mapping

SOLUTION: Many-to-many tagging
  • One sample = multiple tags from different categories
  • A sample is simultaneously a "snare", "percussive", "tight", "trap-friendly", etc.
  • Genre-agnostic - assets tagged by intrinsic properties, not assumed use
  • Scales infinitely - add new tags without schema changes
  • Cross-genre discovery is implicit in tag overlap

EXAMPLE:
  File: tight_snare_01.wav
  Tags: 
    - snare (instrument)
    - percussion (instrument category)
    - punchy (characteristic)
    - tight (characteristic)
    - crispy (characteristic)
    - trap (genre)
    - house (genre)
    - uk-garage (genre)
    - fast-attack (envelope)
    - high-mid (frequency focus)
    - drums-only (use case)

Queries enabled:
  ✓ "Give me snares" → 1 tag lookup
  ✓ "Give me punchy snares" → 2 tag lookup
  ✓ "Give me snares for trap" → 2 tag lookup (snare + trap)
  ✓ "Give me samples not suitable for vocals" → exclude vocal-carrier tag
  ✓ "Give me punchy + tight samples" → 2 characteristic intersection
  ✓ "Find snares used in both trap AND house" → recursive genre overlap
"""

# =============================================================================
# CLASSIFIER CAPABILITIES
# =============================================================================

"""
The AudioClassifier analyzes audio and automatically assigns tags.
It uses signal processing (librosa, scipy) to measure:

1. PERCUSSION DETECTION
   ├─ Onset detection → hi-hats vs kick vs snare
   ├─ Attack/decay envelope analysis
   ├─ Snap detection for snare/clap
   ├─ Low-freq emphasis for kick detection
   └─ Fundamental pitch estimation for 808 vs acoustic kicks

2. FREQUENCY CONTENT
   ├─ Sub-bass (20-60Hz) energy
   ├─ Bass (60-250Hz) energy
   ├─ Low-mid (250-500Hz) energy
   ├─ Mid (500Hz-2kHz) energy
   ├─ High-mid (2-5kHz) energy
   ├─ Treble (5-20kHz) energy
   ├─ Spectrum spread (wide vs narrow)
   └─ Brightness ratio (treble/bass)

3. ENVELOPE CHARACTERISTICS
   ├─ Attack time (< 10ms = percussive, > 200ms = sustained)
   ├─ Decay rate (fast vs long)
   ├─ Plucked detection (quick attack + long decay)
   └─ Pad-like characteristics

4. SPECTRAL CHARACTER
   ├─ Resonance detection (spectral peaks)
   ├─ Zero-crossing rate (noisiness → crispy vs smooth)
   ├─ Spectral centroid (frequency center)
   ├─ MFCC variance (tonal complexity)
   └─ Organic vs digital assessment

5. ENERGY & COMPLEXITY
   ├─ Spectral flux (rate of change)
   ├─ Onset density (Events per second)
   ├─ RMS energy level
   └─ Low/medium/high energy classification

6. USE CASE INFERENCE
   ├─ Melodic-friendly (low complexity + suitable frequency range)
   ├─ Ambient (sparse onsets + low energy)
   ├─ Vocal-carrier (clear frequency bands not overlapping vocals)
   └─ Drums-only (percussive with wide frequency spread)

CONFIDENCE LEVEL:
  • Auto-detected tags = confidence 1.0 (classifier was sure)
  • Manual corrections = confidence < 1.0 (human override)
  • Database queries can filter by confidence if desired
"""

# =============================================================================
# PRACTICAL WORKFLOWS
# =============================================================================

"""
WORKFLOW 1: Build from Scratch
───────────────────────────────

1. Organize samples in folders:
   asset_drop/
   ├── Instruments/
   │   ├── Drums/Kicks/
   │   ├── Drums/Snares/
   │   ├── Drums/Hi_Hats/
   │   ├── Bass/
   │   └── Strings/
   ├── Genre_Kits/
   │   ├── Hip_Hop/
   │   ├── Electronic/
   │   └── Pop/
   └── MIDI/

2. Ingest each category:
   python -m soundbank.ingest ./asset_drop/Instruments/Drums/Kicks \\
     --output ./output --category 808

   During ingestion:
   ✓ Samples resampled, normalized, filtered
   ✓ Classifier analyzes each sample
   ✓ Tags auto-assigned and stored in database
   ✓ Master WAV concatenated
   ✓ Progress shown with detected tags

3. Query intelligently:
   from soundbank.provider import SoundBankProvider
   
   provider = SoundBankProvider('output/master_bank.wav', 'output/bank.db')
   
   # Get a trap-friendly snare
   asset, audio = provider.get_by_tag('trap', limit=10)
   
   # Get lofi-friendly pads matching intensity 0.5
   results = provider.get_by_tags(
       ['lofi', 'pad'],
       intensity_target=0.5,
       limit=10
   )

WORKFLOW 2: Add Manual Tags
────────────────────────────

1. Ingest samples normally (auto-tags are assigned)

2. Review results and add/correct manual tags:
   from soundbank.database import SoundBankDB
   
   db = SoundBankDB('output/bank.db')
   with db:
       # Add manual tag with lower confidence
       db.add_tag(asset_id=5, tag_name='custom-genre', confidence=0.7)
       
       # Add multiple tags
       db.add_tags_batch(
           asset_id=5,
           tags=['lofi-friendly', 'warm', 'soulful'],
           confidence=0.9
       )

WORKFLOW 3: Create Genre-Specific Kits
───────────────────────────────────────

1. Let samples be tagged automatically
2. Create kit by collecting tagged assets:

   trap_kit = {
       'kicks': db.get_assets_by_tags(['trap', 'kick'], match_all=False, limit=10),
       'snares': db.get_assets_by_tags(['trap', 'snare'], match_all=False, limit=10),
       'hats': db.get_assets_by_tags(['trap', 'hi-hat'], match_all=False, limit=10),
       'bass': db.get_assets_by_tags(['trap', 'bass'], match_all=False, limit=5),
       'pads': db.get_assets_by_tags(['trap', 'pad'], match_all=False, limit=5),
   }

WORKFLOW 4: Intensity-Matched Collections
───────────────────────────────────────────

1. Analyze vocal energy (0.0 = quiet, 1.0 = max energy)

2. Query Sound Bank for intensity-matched loops:

   vocals_intensity = 0.75
   
   results = provider.get_by_tags(
       ['loops', 'melodic-friendly'],
       intensity_target=vocals_intensity,
       limit=5
   )
   
   # Get complementary pad for dynamic range
   pad_intensity = vocals_intensity * 0.5  # Lower energy pad
   pad_asset, pad_audio = provider.get_by_tag(
       'ambient',
       intensity_target=pad_intensity
   )

WORKFLOW 5: Cross-Genre Discovery
──────────────────────────────────

1. A "punchy" + "hi-hat" sample can be used in:
   - Trap (has 'trap' tag)
   - House (has 'house' tag)
   - UK Garage (has 'uk-garage' tag)
   - Synthpop (has 'synthpop' tag)

2. Query finds all overlapping uses:

   # Find EVERYTHING that's punchy + hi-hat
   versatile_hats = db.get_assets_by_tags(
       ['punchy', 'hi-hat'],
       match_all=False,  # Has at least one
       limit=50
   )
   
   # The tags inside tell you which genres it works for
   for asset in versatile_hats:
       genres = [t for t in db.get_asset_tags(asset.id) 
                 if t in TAG_CATEGORIES['genre']]
       print(f"{asset.original_filename}: Works in {genres}")
"""

# =============================================================================
# API EXAMPLES
# =============================================================================

"""
BASIC QUERIES
─────────────

# Single tag
assets = db.get_assets_by_tag('punch', limit=20)

# Multiple tags (any match)
results = db.get_assets_by_tags(['trap', 'edm'], match_all=False, limit=20)

# Multiple tags (all required)
results = db.get_assets_by_tags(['punchy', 'melodic-friendly'], match_all=True, limit=20)

# Via provider with intensity
asset, audio = provider.get_by_tag('warm', intensity_target=0.6, limit=10)

# Characteristic search
results = provider.search_by_characteristics(
    characteristics=['punchy', 'crispy', 'bright'],
    intensity_target=0.7,
    limit=10
)


ADVANCED QUERIES
────────────────

# Find samples suitable for BOTH trap AND melodic content
trap_melodic = db.get_assets_by_tags(
    ['trap', 'melodic-friendly'],
    match_all=False,  # Union (at least one)
    limit=50
)

# Find samples that are neither drums-only NOR glitchy
results = [
    a for a in db.get_all_assets()
    if 'drums-only' not in db.get_asset_tags(a.id)
    and 'glitchy' not in db.get_asset_tags(a.id)
]

# Intensity-based filtering with characteristics
high_intensity_warm_pads = provider.get_by_tags(
    tags=['pad', 'warm'],
    intensity_target=0.8,
    limit=20
)

# Get all tags for an asset
tags = db.get_asset_tags(asset_id=5)
print(tags)
# Output: ['snare', 'percussion', 'punchy', 'trap', 'fast-attack', 'tight', ...]

# Filter by tag category
with db:
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT DISTINCT td.tag_name
        FROM tag_definitions td
        WHERE td.category = ?
    """, ('genre',))
    genres = [row['tag_name'] for row in cursor.fetchall()]
    print(genres)
    # Output: ['hip-hop', 'trap', 'boom-bap', 'lofi', ...]
"""

# =============================================================================
# SCALING CONSIDERATIONS
# =============================================================================

"""
DATABASE GROWTH
───────────────
• Each asset = 1 row in assets table
• Auto-detected tags = multiple rows in asset_tags (M2M)
• Typical asset = 8-15 tags assigned
• 1000 assets × 10 tags/asset = 10,000 asset_tag rows (lightweight)
• SQLite can handle 100,000+ assets before optimization needed

PERFORMANCE
───────────
• Index on intensity_score          → O(log n) intensity queries
• Index on category                 → O(log n) category filtering
• Index on asset_tags               → O(log n) tag lookups
• Typical 1000-asset bank query     → < 10ms

OPTIMIZATION (when needed)
──────────────────────────
• Migrate to PostgreSQL for multi-user
• Add caching layer (Redis) for popular queries
• Create materialized views for complex queries
• Archive old assets to separate storage
• Implement tag frequency analysis for unused tags

FILE GROWTH
───────────
• Master WAV grows linearly with samples
• Typical sample = 2-10 seconds × 44.1kHz × 3 bytes = 300KB-3MB
• 1000 5-second samples = ~5GB master WAV (very manageable)
• Database file (bank.db) = ~5-10MB for 1000 assets (negligible)

NETWORK TRANSFER
────────────────
• Lazy-loading: Only load audio when needed
• Database queries return metadata only (fast, small)
• Assets loaded on-demand from master WAV
• Typical query + audio load = < 100ms
"""

# =============================================================================
# FUTURE ENHANCEMENTS
# =============================================================================

"""
POTENTIAL ADDITIONS
────────────────────

1. ML-based Genre Classification
   - Train classifier on genre samples
   - Auto-tag genre with confidence levels
   - Fine-tune and correct over time

2. Perceptual Hashing
   - Detect duplicate/very-similar samples
   - Find "close cousins" for variation
   - Automatic deduplication suggestions

3. Spectral Similarity
   - Find acoustically similar samples
   - "Show me more like this" functionality
   - Cross-pollinate ideas between genres

4. Dynamic Intensity Binning
   - Create "intensity personas" (Low/Mid/High)
   - Smart intensity ranges per category
   - Vocal-matched accompaniment selection

5. Multi-Sample Sequences
   - Group related samples (kick + sub, snare + tail)
   - Manage together as "sample sets"
   - Maintain cohesion in output

6. Usage Analytics
   - Track which samples are used most
   - Identify underutilized gems
   - Recommend based on usage patterns

7. Collaborative Tagging
   - User suggestions for tags
   - Community-driven curation
   - Build genre-specific tag sets

8. Time-based Indexing
   - Variations over time (drum progression)
   - Stem separation (extract body vs tail)
   - Chop points for beat-matching
"""

print(__doc__)
