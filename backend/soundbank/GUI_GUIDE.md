#!/usr/bin/env python3
"""
SOUND BANK GUI INGESTION - QUICK START
========================================

No command-line arguments needed. Just click buttons!
"""

# =============================================================================
# HOW TO USE
# =============================================================================

"""
STEP 1: Open the GUI
─────────────────────

From PowerShell in your project root:
    cd c:\Users\OGTommyP\Desktop\Vocal DB\app\backend
    python soundbank/soundbank_ingest.py

A window will open.


STEP 2: Select Input Directory
───────────────────────────────

Click "Browse..." next to "Audio files to ingest"

Navigate to your sample folder, e.g.:
    c:\Users\OGTommyP\Desktop\Vocal DB\asset_drop\Instruments\Drums\Kicks

Click "Open"


STEP 3: Select Output Directory (Optional)
────────────────────────────────────────────

Default is "./output" which is fine.

If you want to change it, click "Browse..." and select different folder.

This is where master_bank.wav and bank.db will be saved.


STEP 4: Choose Sample Category
───────────────────────────────

Radio buttons for:
    • 808       (bass drums)
    • snare     (snares, claps, percussion)
    • loops     (full drum loops, musical patterns)
    • atmospheres (pads, ambient, textures)

Pick the one matching your samples.


STEP 5: (OPTIONAL) Adjust Advanced Settings
─────────────────────────────────────────────

Defaults usually work fine, but available options:

    ☑ Search subdirectories
       Default: ON (searches in subfolders)
       OFF: Only processes direct folder

    ☑ Apply spectral notch filter
       Default: ON (preserves vocal space)
       OFF: Skip notch filtering

    Normalization:
       • RMS (default) - Volume consistency
       • Peak - Loudness capping
       • None - No normalization

    Target RMS: 0.1 (default)
       Range: 0.05 (quieter) to 0.3 (louder)
       Adjust if samples are too loud/quiet


STEP 6: Click "Start Ingestion"
───────────────────────────────

Processing begins!

You'll see logs showing:
    Processing [1/10]: kick_808_01.wav
      Resampled: 48000Hz -> 44100Hz
      Applied notch filter: -12.0dB @ 1-3kHz
      Normalized RMS: 0.2315 -> 0.1000
      Tags: 808-kick, kick, bass, percussion, tight, fast-attack, ...

    Processing [2/10]: kick_acoustic_01.wav
      Tags: acoustic-kick, kick, bass, fast-attack, warm, organic, ...

After all files processed:
    INGESTION COMPLETE
    ─────────────────
    Processed: 10 files
    Failed: 0 files
    Output: ./output/master_bank.wav
    Index: ./output/bank.db

    ✓ All files processed successfully!


STEP 7: Check Success
──────────────────────

Two files created:
    • master_bank.wav  - All samples concatenated
    • bank.db         - Index with metadata + tags


That's it!
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
"ERROR: No audio files found"
─────────────────────────────
Problem: The folder doesn't contain audio files
Solution: 
  1. Make sure the folder path is correct
  2. Check that files have .wav, .mp3, .flac extension
  3. If using subdirectories, make sure "Search subdirectories" is checked


"ERROR: Input directory not found"
───────────────────────────────────
Problem: The path you selected doesn't exist anymore
Solution:
  1. Click "Browse..." again and re-select the folder
  2. Make sure the folder hasn't been moved or deleted


"ERROR: Failed to process file: ..."
────────────────────────────────────
Problem: One file couldn't be processed
Solution:
  1. The file might be corrupted
  2. Try removing that file and re-running ingestion
  3. Check the file isn't locked/open in another program


"My samples sound too loud/quiet"
──────────────────────────────────
Problem: Target RMS setting is wrong
Solution:
  1. Re-run ingestion with different Target RMS:
     • For quieter samples: 0.15 or 0.2
     • For louder samples: 0.05 or 0.08
  2. Re-ingest by:
     - Delete old output files (master_bank.wav, bank.db)
     - Adjust Target RMS slider
     - Click "Start Ingestion" again


"How do I OVERWRITE an existing sound bank?"
──────────────────────────────────────────
1. Delete these files:
   - output/master_bank.wav
   - output/bank.db

2. Run ingestion again with new samples

3. Tags will be recalculated for new samples
"""

# =============================================================================
# EXAMPLE WORKFLOWS
# =============================================================================

"""
WORKFLOW 1: Build Sound Bank from Scratch
───────────────────────────────────────────

Step 1: Run GUI for 808 kicks
    Input:  as set_drop/Instruments/Drums/Kicks
    Output: ./output
    Category: 808
    → Creates master_bank.wav with 808-kick tagged samples

Step 2: Run GUI for snares
    Input:  asset_drop/Instruments/Drums/Snares
    Output: ./output    (SAME folder)
    Category: snare
    → Appends to master_bank.wav, adds snare-tagged samples

Step 3: Run GUI for loops
    Input:  asset_drop/Loops
    Output: ./output    (SAME folder)
    Category: loops
    → Appends to master_bank.wav, adds loop-tagged samples

Result: One master_bank.wav with all samples indexed by:
    - category (808, snare, loops)
    - tags (kick, snare, punchy, warm, trap, etc.)
    - intensity (0.0-1.0 energy scale)
    - song section (intro, verse, chorus, drop, etc.)


WORKFLOW 2: Add More Samples Later
────────────────────────────────────

Once you have a sound bank:

Step 1: Delete old files (to rebuild)
    rm output/master_bank.wav
    rm output/bank.db

Step 2: Copy your NEW samples to a folder

Step 3: Run GUI with new folder
    Input: your_new_samples_folder
    Output: ./output
    Category: choose appropriate one

Result: Fresh sound bank with better samples, same structure


WORKFLOW 3: Test Before Ingesting
───────────────────────────────────

Want to try one sample first?

Step 1: Create test folder with just 1-2 samples

Step 2: Run GUI on test folder
    Input: test_folder_with_2_samples
    Output: ./output_test
    Category: loops

Step 3: Check if settings are good
    Look at the logs - are the tags correct?
    Do the auto-detected tags make sense?

Step 4: If settings good, ingest full folder
    If not, adjust Target RMS and try again
"""

# =============================================================================
# COMMAND LINE ALTERNATIVE (Still Available)
# =============================================================================

"""
If you prefer command line (advanced users):

python -m soundbank.ingest ./asset_drop/Drums/Kicks \\
    --output ./output \\
    --category 808 \\
    --target-rms 0.1 \\
    --normalize rms

But the GUI is simpler and shows progress visually!
"""

print(__doc__)
