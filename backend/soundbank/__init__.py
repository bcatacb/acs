"""
Proprietary Sound Bank Builder
==============================
A standalone Python utility that transforms raw audio samples into a single,
addressable Master WAV-Container and a Relational Index.

Modules:
- ingest.py: CLI tool for processing and concatenating audio samples
- provider.py: Retrieval API for intensity-based audio access
- transformations.py: Audio transformation modules (notch filter, bit-crush, time-stretch)
- database.py: SQLite database operations for the sound bank index
- test_generator.py: Test sample generator for validation
"""

__version__ = "1.0.0"
__author__ = "Sound Bank Builder"

from .database import SoundBankDB
from .provider import SoundBankProvider
from .transformations import Transformations
