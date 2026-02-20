#!/usr/bin/env python3
"""
SOUND BANK INGESTION - COMPLETE SETUP VERIFICATION
====================================================

This script checks that all Sound Bank components are installed and ready to use.
Run this before attempting ingestion to ensure everything is in place.
"""

import os
import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists and report status."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if not exists:
        print(f"     Missing: {path}")
    return exists

def check_directory_exists(path, description):
    """Check if a directory exists and report status."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if not exists:
        print(f"     Missing: {path}")
    return exists

def main():
    print("\n" + "="*60)
    print("SOUND BANK SETUP VERIFICATION")
    print("="*60 + "\n")
    
    # Get the soundbank directory
    soundbank_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(soundbank_dir)
    
    print(f"Soundbank Directory: {soundbank_dir}\n")
    
    # Check core files
    print("CORE INGESTION FILES:")
    print("─" * 60)
    all_good = True
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "ingest.py"),
        "ingest.py - Ingestion engine"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "ingest_gui.py"),
        "ingest_gui.py - GUI tool (click-based)"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "soundbank_ingest.py"),
        "soundbank_ingest.py - GUI launcher"
    )
    
    print("\nDATABASE & AUDIO PROCESSING:")
    print("─" * 60)
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "database.py"),
        "database.py - Tag system & indexing"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "classifier.py"),
        "classifier.py - Auto-detection engine"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "provider.py"),
        "provider.py - Query API"
    )
    
    print("\nDOCUMENTATION:")
    print("─" * 60)
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "GUI_GUIDE.md"),
        "GUI_GUIDE.md - How to use the GUI tool"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "ARCHITECTURE.md"),
        "ARCHITECTURE.md - System design"
    )
    
    all_good &= check_file_exists(
        os.path.join(soundbank_dir, "TAGGING_GUIDE.py"),
        "TAGGING_GUIDE.py - All 120+ tags explained"
    )
    
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL FILES PRESENT - READY TO INGEST!")
        print("="*60)
        print("\nQUICK START:")
        print("  python soundbank_ingest.py")
        print("\nWill open a GUI window. Click to select folders and start ingesting!")
        return 0
    else:
        print("✗ MISSING FILES - SETUP INCOMPLETE")
        print("="*60)
        print("\nPlease ensure all files are in:")
        print(f"  {soundbank_dir}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
