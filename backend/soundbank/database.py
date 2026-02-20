"""
Database Schema and Operations for Sound Bank Index (bank.db)
SQLite database storing the map of the Master WAV file.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SoundAsset:
    """Represents a single audio asset in the sound bank."""
    id: int
    original_filename: str
    start_sample: int
    end_sample: int
    category: str
    intensity_score: float
    spectral_hole: float
    rms_energy: float
    mid_range_density: float
    sample_rate: int
    duration_samples: int
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_samples / self.sample_rate


class SoundBankDB:
    """SQLite database manager for the sound bank index."""
    
    CATEGORIES = ["808", "snare", "loops", "atmospheres"]
    
    def __init__(self, db_path: str = "bank.db"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        
    def connect(self) -> None:
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def create_schema(self) -> None:
        """Create the database schema for sound bank assets."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        
        # Main assets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                start_sample INTEGER NOT NULL,
                end_sample INTEGER NOT NULL,
                category TEXT NOT NULL,
                intensity_score REAL NOT NULL,
                spectral_hole REAL NOT NULL,
                rms_energy REAL NOT NULL,
                mid_range_density REAL NOT NULL,
                sample_rate INTEGER NOT NULL DEFAULT 44100,
                duration_samples INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tag definitions table (curated list of all possible tags)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Asset-to-tag mapping (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_tags (
                asset_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                PRIMARY KEY (asset_id, tag_id),
                FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tag_definitions(id) ON DELETE CASCADE
            )
        """)
        
        # Index for fast intensity-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_intensity 
            ON assets(intensity_score)
        """)
        
        # Index for category-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category 
            ON assets(category)
        """)
        
        # Index for combined intensity + category queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category_intensity 
            ON assets(category, intensity_score)
        """)
        
        # Index for tag queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_tags
            ON asset_tags(tag_id, asset_id)
        """)
        
        # Metadata table for bank info
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bank_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        self._initialize_tag_definitions()
        self.conn.commit()
        
    def reset_database(self) -> None:
        """Clear all data and recreate schema."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS asset_tags")
        cursor.execute("DROP TABLE IF EXISTS tag_definitions")
        cursor.execute("DROP TABLE IF EXISTS assets")
        cursor.execute("DROP TABLE IF EXISTS bank_metadata")
        self.conn.commit()
        self.create_schema()
    
    def _initialize_tag_definitions(self) -> None:
        """Initialize curated tag definitions."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        cursor = self.conn.cursor()
        
        # Check if already initialized
        cursor.execute("SELECT COUNT(*) as count FROM tag_definitions")
        if cursor.fetchone()['count'] > 0:
            return
        
        tag_definitions = [
            # Instrument Types
            ("kick", "instrument", "Bass drum / kick drum"),
            ("snare", "instrument", "Snare drum"),
            ("hi-hat", "instrument", "Hi-hat cymbal"),
            ("clap", "instrument", "Clap sound"),
            ("tom", "instrument", "Tom drum"),
            ("percussion", "instrument", "Percussion element"),
            ("bass", "instrument", "Bass instrument"),
            ("synth", "instrument", "Synthesized sound"),
            ("pad", "instrument", "Sustained pad"),
            ("pluck", "instrument", "Plucked string sound"),
            ("horn", "instrument", "Horn/brass instrument"),
            ("strings", "instrument", "String instruments"),
            ("vocal", "instrument", "Vocal sound"),
            ("piano", "instrument", "Piano"),
            ("guitar", "instrument", "Guitar"),
            
            # Drum Specifics
            ("808-kick", "drum-type", "808-style bass drum"),
            ("acoustic-kick", "drum-type", "Acoustic/organic kick"),
            ("synth-kick", "drum-type", "Synthesized kick"),
            ("closed-hat", "drum-type", "Closed hi-hat"),
            ("open-hat", "drum-type", "Open hi-hat"),
            ("kick-roll", "drum-type", "Rapid kick drum"),
            ("snare-roll", "drum-type", "Rapid snare"),
            
            # Genres
            ("hip-hop", "genre", "Hip-hop production"),
            ("trap", "genre", "Trap music"),
            ("boom-bap", "genre", "Boom bap / classic hip-hop"),
            ("lofi", "genre", "Lo-fi beats"),
            ("grime", "genre", "Grime music"),
            ("house", "genre", "House music"),
            ("techno", "genre", "Techno music"),
            ("dubstep", "genre", "Dubstep"),
            ("future-bass", "genre", "Future bass"),
            ("edm", "genre", "Electronic dance music"),
            ("dnb", "genre", "Drum and bass"),
            ("uk-garage", "genre", "UK garage"),
            ("rnb", "genre", "R&B / Soul"),
            ("neo-soul", "genre", "Neo-soul"),
            ("pop", "genre", "Pop music"),
            ("synthpop", "genre", "Synth-pop"),
            
            # Sonic Characteristics
            ("punchy", "characteristic", "Sharp attack, impactful"),
            ("warm", "characteristic", "Warm, smooth tone"),
            ("bright", "characteristic", "Bright, high-end heavy"),
            ("dark", "characteristic", "Dark, low-end heavy"),
            ("metallic", "characteristic", "Metallic quality"),
            ("organic", "characteristic", "Natural, organic sound"),
            ("digital", "characteristic", "Digital, artificial sound"),
            ("crispy", "characteristic", "Clear, crisp definition"),
            ("muddy", "characteristic", "Thick, murky tone"),
            ("thin", "characteristic", "Thin, narrow frequency range"),
            ("resonant", "characteristic", "Resonant, ringing quality"),
            ("tight", "characteristic", "Tight, controlled sound"),
            ("loose", "characteristic", "Loose, expressive sound"),
            ("filtered", "characteristic", "EQ filtered/processed"),
            ("aggressive", "characteristic", "Aggressive, intense character"),
            ("smooth", "characteristic", "Smooth, mellow tone"),
            ("colored", "characteristic", "Heavily processed/colored"),
            
            # Frequency Focus
            ("sub-bass", "frequency", "Sub-bass range (20-60Hz)"),
            ("bass", "frequency", "Bass range (60-250Hz)"),
            ("low-mid", "frequency", "Low-mid range (250-500Hz)"),
            ("mid", "frequency", "Mid range (500Hz-2kHz)"),
            ("high-mid", "frequency", "High-mid range (2-5kHz)"),
            ("treble", "frequency", "Treble/presence (5-20kHz)"),
            ("wide-spectrum", "frequency", "Covers wide frequency range"),
            ("narrow-band", "frequency", "Narrow frequency range"),
            
            # Envelope/Dynamics
            ("fast-attack", "envelope", "Quick attack time"),
            ("slow-attack", "envelope", "Gradual attack"),
            ("short-decay", "envelope", "Quick decay/release"),
            ("long-decay", "envelope", "Extended decay/sustain"),
            ("percussive", "envelope", "Percussive transient"),
            ("sustained", "envelope", "Sustained tone"),
            ("plucked", "envelope", "Plucked/struck character"),
            ("pad-like", "envelope", "Pad-like envelope"),
            
            # Use Cases
            ("melodic-friendly", "use-case", "Works well with melody"),
            ("drums-only", "use-case", "Best for drums/percussion section"),
            ("vocal-carrier", "use-case", "Good for carrying vocals"),
            ("loop-able", "use-case", "Clean loop points"),
            ("glitchy", "use-case", "Glitchy/experimental"),
            ("cinematic", "use-case", "Cinematic/orchestral use"),
            ("minimalist", "use-case", "Minimal, sparser texture"),
            ("texture", "use-case", "Textural element"),
            ("ambient", "use-case", "Ambient/background use"),
            ("upfront", "use-case", "Mix-forward, prominent"),
            
            # Energy/Intensity
            ("low-energy", "intensity", "Sparse, minimal energy"),
            ("medium-energy", "intensity", "Moderate density"),
            ("high-energy", "intensity", "Dense, high energy"),
            ("explosive", "intensity", "Sudden, explosive burst"),
            
            # Song Sections
            ("intro", "song-section", "Intro section (establishes vibe, builds anticipation)"),
            ("verse", "song-section", "Verse section (main melodic content, storytelling)"),
            ("pre-chorus", "song-section", "Pre-chorus section (builds tension toward chorus)"),
            ("chorus", "song-section", "Chorus section (main hook, melody, energy peak)"),
            ("bridge", "song-section", "Bridge section (contrasting element, break from pattern)"),
            ("breakdown", "song-section", "Breakdown section (reduced instrumentation, tension drop)"),
            ("build-up", "song-section", "Build-up section (increasing energy toward drop/peak)"),
            ("drop", "song-section", "Drop section (sudden energy release, climax)"),
            ("outro", "song-section", "Outro section (wind-down, conclusion)"),
            ("pre-drop", "song-section", "Pre-drop section (tension before drop)"),
            ("fill", "song-section", "Fill/transition (connects sections smoothly)"),
            ("interlude", "song-section", "Interlude (instrumental break between sections)"),
        ]
        
        for tag_name, category, description in tag_definitions:
            cursor.execute("""
                INSERT OR IGNORE INTO tag_definitions (tag_name, category, description)
                VALUES (?, ?, ?)
            """, (tag_name, category, description))
        
        self.conn.commit()
    
    def add_tag(self, asset_id: int, tag_name: str, confidence: float = 1.0) -> None:
        """Add a tag to an asset."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        cursor = self.conn.cursor()
        
        # Get tag ID
        cursor.execute("SELECT id FROM tag_definitions WHERE tag_name = ?", (tag_name,))
        tag_row = cursor.fetchone()
        
        if not tag_row:
            raise ValueError(f"Tag '{tag_name}' not found in definitions")
        
        tag_id = tag_row['id']
        
        cursor.execute("""
            INSERT OR REPLACE INTO asset_tags (asset_id, tag_id, confidence)
            VALUES (?, ?, ?)
        """, (asset_id, tag_id, confidence))
        
        self.conn.commit()
    
    def add_tags_batch(self, asset_id: int, tags: List[str], confidence: float = 1.0) -> None:
        """Add multiple tags to an asset."""
        for tag in tags:
            self.add_tag(asset_id, tag, confidence)
    
    def get_asset_tags(self, asset_id: int) -> List[str]:
        """Get all tags for an asset."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT td.tag_name 
            FROM asset_tags at
            JOIN tag_definitions td ON at.tag_id = td.id
            WHERE at.asset_id = ?
            ORDER BY at.confidence DESC
        """, (asset_id,))
        
        return [row['tag_name'] for row in cursor.fetchall()]
    
    def get_assets_by_tag(self, tag_name: str, limit: int = 100) -> List[SoundAsset]:
        """Get all assets with a specific tag."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT a.* 
            FROM assets a
            JOIN asset_tags at ON a.id = at.asset_id
            JOIN tag_definitions td ON at.tag_id = td.id
            WHERE td.tag_name = ?
            ORDER BY at.confidence DESC
            LIMIT ?
        """, (tag_name, limit))
        
        return [SoundAsset(**{k: v for k, v in dict(row).items() if k != 'created_at'}) for row in cursor.fetchall()]
    
    def get_assets_by_tags(self, tags: List[str], match_all: bool = False, limit: int = 100) -> List[SoundAsset]:
        """
        Get assets matching tags.
        
        Args:
            tags: List of tag names
            match_all: If True, asset must have ALL tags; if False, ANY tag
            limit: Maximum results
        """
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        if not tags:
            return []
        
        cursor = self.conn.cursor()
        
        if match_all:
            # Asset must have ALL specified tags
            placeholders = ','.join('?' * len(tags))
            cursor.execute(f"""
                SELECT a.* 
                FROM assets a
                WHERE a.id IN (
                    SELECT at.asset_id
                    FROM asset_tags at
                    JOIN tag_definitions td ON at.tag_id = td.id
                    WHERE td.tag_name IN ({placeholders})
                    GROUP BY at.asset_id
                    HAVING COUNT(DISTINCT td.tag_name) = ?
                )
                LIMIT ?
            """, tags + [len(tags), limit])
        else:
            # Asset must have ANY of the specified tags
            placeholders = ','.join('?' * len(tags))
            cursor.execute(f"""
                SELECT DISTINCT a.* 
                FROM assets a
                JOIN asset_tags at ON a.id = at.asset_id
                JOIN tag_definitions td ON at.tag_id = td.id
                WHERE td.tag_name IN ({placeholders})
                LIMIT ?
            """, tags + [limit])
        
        return [SoundAsset(**{k: v for k, v in dict(row).items() if k != 'created_at'}) for row in cursor.fetchall()]
        
    def insert_asset(
        self,
        original_filename: str,
        start_sample: int,
        end_sample: int,
        category: str,
        intensity_score: float,
        spectral_hole: float,
        rms_energy: float,
        mid_range_density: float,
        sample_rate: int = 44100
    ) -> int:
        """Insert a new asset into the database. Returns the asset ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        duration_samples = end_sample - start_sample
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO assets (
                original_filename, start_sample, end_sample, category,
                intensity_score, spectral_hole, rms_energy, mid_range_density,
                sample_rate, duration_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            original_filename, start_sample, end_sample, category,
            intensity_score, spectral_hole, rms_energy, mid_range_density,
            sample_rate, duration_samples
        ))
        
        self.conn.commit()
        return cursor.lastrowid
        
    def get_asset_by_id(self, asset_id: int) -> Optional[SoundAsset]:
        """Retrieve a single asset by its ID."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM assets WHERE id = ?", (asset_id,))
        row = cursor.fetchone()
        
        if row:
            return SoundAsset(**{k: v for k, v in dict(row).items() if k != 'created_at'})
        return None
        
    def get_by_intensity(
        self,
        target_rms: float,
        tolerance: float = 0.1,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[SoundAsset]:
        """
        Query assets matching a target RMS energy level.
        
        Args:
            target_rms: Target RMS energy value to match
            tolerance: Acceptable deviation from target (default 0.1)
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of SoundAsset objects sorted by proximity to target RMS
        """
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT *, ABS(intensity_score - ?) as distance
                FROM assets
                WHERE category = ?
                AND intensity_score BETWEEN ? AND ?
                ORDER BY distance ASC
                LIMIT ?
            """, (target_rms, category, target_rms - tolerance, target_rms + tolerance, limit))
        else:
            cursor.execute("""
                SELECT *, ABS(intensity_score - ?) as distance
                FROM assets
                WHERE intensity_score BETWEEN ? AND ?
                ORDER BY distance ASC
                LIMIT ?
            """, (target_rms, target_rms - tolerance, target_rms + tolerance, limit))
            
        rows = cursor.fetchall()
        # Exclude non-dataclass fields
        excluded_fields = {'distance', 'created_at'}
        return [SoundAsset(**{k: v for k, v in dict(row).items() if k not in excluded_fields}) for row in rows]
        
    def get_by_category(self, category: str, limit: int = 100) -> List[SoundAsset]:
        """Get all assets in a category."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM assets WHERE category = ? LIMIT ?",
            (category, limit)
        )
        
        # Exclude created_at from results
        return [SoundAsset(**{k: v for k, v in dict(row).items() if k != 'created_at'}) for row in cursor.fetchall()]
        
    def get_all_assets(self) -> List[SoundAsset]:
        """Get all assets in the database."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM assets ORDER BY id")
        
        # Exclude created_at from results
        return [SoundAsset(**{k: v for k, v in dict(row).items() if k != 'created_at'}) for row in cursor.fetchall()]
        
    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata key-value pair."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO bank_metadata (key, value) VALUES (?, ?)
        """, (key, value))
        self.conn.commit()
        
    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value by key."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM bank_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row['value'] if row else None
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.conn:
            raise RuntimeError("Database not connected")
            
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total assets
        cursor.execute("SELECT COUNT(*) as count FROM assets")
        stats['total_assets'] = cursor.fetchone()['count']
        
        # Assets per category
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM assets 
            GROUP BY category
        """)
        stats['by_category'] = {row['category']: row['count'] for row in cursor.fetchall()}
        
        # Intensity range
        cursor.execute("""
            SELECT MIN(intensity_score) as min_intensity, 
                   MAX(intensity_score) as max_intensity,
                   AVG(intensity_score) as avg_intensity
            FROM assets
        """)
        row = cursor.fetchone()
        stats['intensity_range'] = {
            'min': row['min_intensity'],
            'max': row['max_intensity'],
            'avg': row['avg_intensity']
        }
        
        # Total duration
        cursor.execute("SELECT SUM(duration_samples) as total FROM assets")
        total_samples = cursor.fetchone()['total'] or 0
        stats['total_duration_seconds'] = total_samples / 44100
        
        return stats
