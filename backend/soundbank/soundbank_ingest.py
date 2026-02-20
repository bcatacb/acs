#!/usr/bin/env python3
"""
Sound Bank GUI Launcher
=======================
Simply run this script to open the GUI tool for ingesting samples.

No command-line arguments needed!

Usage:
    python soundbank_ingest.py

Or in Windows:
    Double-click this file
    Or run: python soundbank_ingest.py
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ingest_gui import SoundBankIngestionGUI
    import tkinter as tk
    
    print("Launching Sound Bank Ingestion GUI...")
    
    root = tk.Tk()
    app = SoundBankIngestionGUI(root)
    root.mainloop()
    
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("\nMake sure you have installed the required packages:")
    print("  pip install librosa soundfile numpy scipy")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
