#!/usr/bin/env python3
"""
SOUND BANK SYSTEM - READY TO USE
==================================

The complete Sound Bank system is now set up and ready for ingestion.
No more command-line errors. Just click and ingest!

✓ GUI Tool Built & Tested
✓ All Components Verified  
✓ Documentation Complete
✓ Ready for First Ingestion
"""

# =============================================================================
# WHAT YOU HAVE NOW
# =============================================================================

"""
✓ INTELLIGENT SOUND BANK SYSTEM
  • Master WAV file (all samples in one file)
  • SQLite index (fast queries by tag, intensity, category)
  • 120+ curated tags auto-assigned during ingestion
  • Vocal-intensity-matched loop selection for ACS

✓ GUI INGESTION TOOL (Click-Based)
  • No command-line syntax needed
  • Visual directory selection with "Browse..." buttons
  • Real-time progress display as files process
  • Automatic tag detection (shows "808-kick, percussion, tight, fast-attack...")
  • Settings: category, recursive search, notch filter, normalization
  • Success/error alerts in popup windows

✓ AUTO-DETECTION ENGINE
  • Detects drums (kick, snare, hi-hat, clap, tom)
  • Classifies frequency content (bass, mid, treble)
  • Identifies envelope patterns (fast-attack, sustained, plucked)
  • Assigns sonic characteristics (punchy, warm, bright, smooth, organic)
  • Detects song sections (intro, verse, chorus, drop, outro)
  • Labels energy levels (low, medium, high, explosive)

✓ QUERY API
  • Get samples by tag: "punchy", "trap", "808-kick"
  • Multi-tag search: find samples tagged ["trap", "kick"]
  • Vocal-intensity matching: select loops matching vocal energy
  • Category filtering: drums-only, loops-only, etc.
"""

# =============================================================================
# HOW TO USE (SUPER QUICK)
# =============================================================================

"""
STEP 1: Open Command Prompt
─────────────────────────────

Windows PowerShell:
  cd c:\Users\OGTommyP\Desktop\Vocal'\ DB\app\backend


STEP 2: Launch the GUI
──────────────────────

  python soundbank/soundbank_ingest.py

A window appears with:
  • "Audio files to ingest" - Click "Browse..." to select folder
  • "Output directory" - Where master_bank.wav gets saved
  • Category selection - Choose 808, snare, loops, or atmospheres
  • Options - Defaults usually fine (can adjust if needed)
  • Progress display - Real-time log of processing


STEP 3: Select Your Audio Folder
──────────────────────────────────

Click "Browse..." button

Navigate to any folder with audio files:
  • c:\Users\OGTommyP\Desktop\Vocal DB\asset_drop\Instruments\Drums\Kicks
  • c:\Users\OGTommyP\Desktop\Vocal DB\asset_drop\Loops\Drums
  • c:\Users\OGTommyP\Desktop\Vocal DB\asset_drop\Instruments\Bass
  • Or ANY custom folder on your computer

Click "Open"


STEP 4: Click "Start Ingestion"
────────────────────────────────

Processing begins!

You see logs like:
  Processing [1/10]: kick_808_01.wav
    Tags: 808-kick, kick, bass, percussion, tight, fast-attack

  Processing [2/10]: kick_acoustic.wav
    Tags: acoustic-kick, kick, bass, warm, organic

After done:
  ✓ Success message
  ✓ Files saved to output/master_bank.wav
  ✓ Index saved to output/bank.db


STEP 5: Done!
──────────────

Your sound bank is ready to use in ACS!

ACS can now:
  • Query loops by tag (e.g., "find all trap drops")
  • Match loop intensity to vocal energy
  • Mix accompaniments from 120+ tag categories
  • Build genre-specific arrangements
"""

# =============================================================================
# EXAMPLE: BUILDING YOUR FIRST SOUND BANK
# =============================================================================

"""
SCENARIO: You have samples organized in folders

  asset_drop/
    Instruments/
      Drums/
        Kicks/        ← 50 kick drum files
        Snares/       ← 30 snare drum files
      Bass/           ← 20 bass files
    Loops/            ← 100 drum loops

PLAN: Build one master sound bank with all of them

STEP 1: Ingest Kicks
  ─────────────────
  Run GUI → Select "asset_drop/Instruments/Drums/Kicks"
  Category: 808
  Output: ./output
  Start → Wait for completion ✓

  Result: master_bank.wav (50 kicks, all tagged with 808-kick, kick, bass, etc.)

STEP 2: Ingest Snares
  ─────────────────────
  Run GUI → Select "asset_drop/Instruments/Drums/Snares"
  Category: snare
  Output: ./output   (SAME folder - appends to master_bank.wav)
  Start → Wait ✓

  Result: master_bank.wav (now 80 samples: 50 kicks + 30 snares)

STEP 3: Ingest Bass
  ──────────────────
  Run GUI → Select "asset_drop/Instruments/Bass"
  Category: 808 or snare (whichever makes sense)
  Output: ./output   (same)
  Start → Wait ✓

  Result: master_bank.wav (now 100 samples: kicks + snares + bass)

STEP 4: Ingest Loops
  ──────────────────
  Run GUI → Select "asset_drop/Loops"
  Category: loops
  Output: ./output   (same)
  Start → Wait ✓

  Result: master_bank.wav (now 200 samples: everything!)

DONE! One master_bank.wav with:
  ✓ All 200 samples indexed
  ✓ All tagged automatically (kick, snare, bass, loop, trap, punchy, etc.)
  ✓ Ready for ACS to query and use
"""

# =============================================================================
# WHAT IF THERE'S AN ERROR?
# =============================================================================

"""
ERROR: "No audio files found"
─────────────────────────────
→ The folder doesn't have .wav, .mp3, .flac files
→ Check settings: Is "Search subdirectories" checked?
→ Verify the folder path is correct in the selector


ERROR: "Directory not found"
──────────────────────────────
→ The folder was moved or deleted
→ Click "Browse..." again and re-select


ERROR: "Failed to process [filename]"
──────────────────────────────────────
→ One file might be corrupted
→ Delete that file and re-run ingestion
→ Or check if file is open/locked in another program


Samples sound too loud/quiet
──────────────────────────────
→ Adjust "Target RMS" setting (0.05 to 0.3)
→ Re-run ingestion with new setting


Can't find the GUI window
─────────────────────────
→ It might be hidden behind other windows
→ Check task bar for "Sound Bank Ingestion Tool" button
→ Click it to bring window to front
"""

# =============================================================================
# WHAT HAPPENS NEXT IN ACS
# =============================================================================

"""
Once your Sound Bank is built (master_bank.wav + bank.db), ACS can:

1. QUERY BY TAG
   Provider.get_by_tag("trap", limit=10)
   → Returns 10 trap-tagged loops

2. MULTI-TAG SEARCH
   Provider.get_by_tags(["trap", "drop"], match_all=False, limit=5)
   → Returns loops that match either "trap" OR "drop"

3. INTENSITY MATCHING
   Provider.get_by_normalized_intensity(0.65)
   → Returns loops with energy matching your vocal (0.65 = 65% intensity)

4. MIX ACCOMPANIMENT
   Get kick tagged [808-kick]
   Get snare tagged [snare]
   Get loop tagged [trap, drop]
   Get pad tagged [ambient, sustained]
   → Mix into one 4-bar accompaniment

All from ONE master_bank.wav file!
No file management hassle.
Just query and retrieve.
"""

# =============================================================================
# GOOD PRACTICES
# =============================================================================

"""
1. START SMALL
   ─────────────
   Don't ingest 1000 files on day 1.
   Try 10-20 samples first to see how tags are detected.
   If tags look good, ingest larger batches.

2. ORGANIZE BY PURPOSE
   ────────────────────
   Don't mix genres in one ingest.
   • One run: All trap kicks
   • Next run: All house kicks
   • Next run: All lofi kicks
   This makes finding samples easier (search by tag).

3. CHECK SAMPLE QUALITY
   ───────────────────
   Delete corrupted/silence files before ingesting.
   (The auto-detector will still try, but clean data = better tags)

4. KEEP OUTPUTS ORGANIZED
   ──────────────────────
   Don't overwrite your first sound bank immediately.
   Try:
     Output 1: ./output_trap (10 trap files)
     Output 2: ./output_house (10 house files)
     Output 3: ./output_master (mix both)
   This lets you test before committing.

5. USE TARGET RMS CONSISTENTLY
   ──────────────────────────
   If you ingest at 0.1 RMS, keep it at 0.1 for all future ingests.
   Consistency = volume consistency across all samples.
"""

# =============================================================================
# FILES INVOLVED
# =============================================================================

"""
IN YOUR PROJECT (/app/backend/soundbank/):

INGESTION TOOLS:
  • soundbank_ingest.py      → Launcher (click this)
  • ingest_gui.py            → GUI window (called by launcher)
  • ingest.py                → Processing engine (called by GUI)

DATA PROCESSING:
  • classifier.py            → Auto-tagging engine
  • database.py              → Tag system & indexing
  • provider.py              → Query API

DOCUMENTATION:
  • GUI_GUIDE.md             → Detailed GUI instructions
  • ARCHITECTURE.md          → System design details
  • TAGGING_GUIDE.py         → All 120+ tags explained
  • verify_setup.py          → Checks everything is installed

OUTPUT (IN ./output/):
  • master_bank.wav          → All samples concatenated
  • bank.db                  → SQLite index (metadata + tags)
"""

# =============================================================================
# TROUBLESHOOTING CHECKLIST
# =============================================================================

"""
Before running ingestion, check:

☐ Python is installed (python --version in terminal)
☐ Required packages installed (librosa, numpy, scipy)
☐ Audio files are in a folder (.wav, .mp3, .flac)
☐ Output folder exists or is writable
☐ Enough disk space for master_bank.wav
  (each sample ~ 1-10 MB, so 100 samples = 100-1000 MB)

If GUI won't start:
☐ Try: python -c "import tkinter; print(tkinter.TkVersion)"
  Should print a version number (8.0+)

If audio files won't process:
☐ Check file format (supported: .wav, .mp3, .flac, .aiff, .ogg)
☐ Check file duration (should be >0.1 seconds)
☐ Check file isn't corrupted (try opening in Audacity)
☐ Check file isn't locked by another program

If tags look wrong:
☐ Check Target RMS setting (too high/low can affect detection)
☐ Check "Apply spectral notch filter" toggle
  (off = no vocal pocket carving, may affect detection)
☐ Run on small batch to debug before full ingest
"""

print(__doc__)
