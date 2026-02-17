import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "backend" / "assets" / "acs.db"
SUPPORTED_AUDIO = {".wav", ".mp3", ".aiff", ".flac", ".m4a", ".ogg"}
SUPPORTED_MIDI = {".mid", ".midi"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(db_path: Optional[str] = None) -> Path:
    if db_path:
        return Path(db_path)
    env = os.getenv("ACS_ASSET_DB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                role TEXT,
                bpm INTEGER,
                musical_key TEXT,
                style_tag TEXT,
                energy TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_assets_type_role ON assets(type, role);
            CREATE INDEX IF NOT EXISTS idx_assets_style ON assets(style_tag);
            CREATE INDEX IF NOT EXISTS idx_assets_active ON assets(active);

            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                json_config TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                project_id TEXT,
                profile_id TEXT,
                asset_ids_json TEXT NOT NULL DEFAULT '[]',
                params_json TEXT NOT NULL DEFAULT '{}',
                output_path TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def _infer_type_role(path: Path) -> Tuple[str, str]:
    lower = [p.lower() for p in path.parts]
    ext = path.suffix.lower()
    if ext in SUPPORTED_MIDI:
        if "chords" in lower:
            return "midi", "chords"
        if "melodies" in lower:
            return "midi", "melodies"
        if "basslines" in lower:
            return "midi", "basslines"
        if "drums" in lower:
            return "midi", "drums"
        return "midi", "misc"
    if ext in SUPPORTED_AUDIO:
        if "loops" in lower:
            return "loop", "loop"
        if "kicks" in lower:
            return "drum", "kick"
        if "hats" in lower:
            return "drum", "hat"
        if "clapsnares" in lower or "claps" in lower or "snares" in lower:
            return "drum", "clap"
        if "drums" in lower:
            return "drum", "perc"
        return "audio", "misc"
    return "unknown", "unknown"


def _infer_style(path: Path) -> Optional[str]:
    name = path.as_posix().lower()
    candidates = [
        "boom_bap",
        "east_coast",
        "west_coast",
        "southern",
        "melodic",
        "trap",
        "drill",
        "lo_fi",
        "lofi",
    ]
    for c in candidates:
        if c in name or c.replace("_", " ") in name or c.replace("_", "") in name:
            if c == "lofi":
                return "lo_fi"
            return c
    if "boombap" in name:
        return "boom_bap"
    return None


def _infer_bpm_key(path: Path) -> Tuple[Optional[int], Optional[str]]:
    stem = path.stem
    bpm = None
    key = None
    bpm_match = re.search(r"(\d{2,3})\s*bpm", stem, re.IGNORECASE)
    if bpm_match:
        try:
            bpm = int(bpm_match.group(1))
        except Exception:
            bpm = None

    key_match = re.search(r"\b([A-G](?:#|b)?(?:maj|min|m)?)\b", stem, re.IGNORECASE)
    if key_match:
        key = key_match.group(1)
    return bpm, key


def ingest_assets(
    root_dir: str,
    style_tag: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    init_db(db_path)
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Asset root does not exist: {root}")

    exts = SUPPORTED_AUDIO.union(SUPPORTED_MIDI)
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files))

    inserted = 0
    updated = 0
    skipped = 0
    now = _now_iso()
    tags = extra_tags or []

    with _connect(db_path) as conn:
        for p in files:
            p_abs = str(p.resolve())
            atype, role = _infer_type_role(p)
            if atype == "unknown":
                skipped += 1
                continue
            bpm, mkey = _infer_bpm_key(p)
            st = style_tag or _infer_style(p)
            row = conn.execute("SELECT id FROM assets WHERE path = ?", (p_abs,)).fetchone()
            payload = (
                atype,
                role,
                bpm,
                mkey,
                st,
                json.dumps(tags),
                now,
                p_abs,
            )
            if row:
                conn.execute(
                    """
                    UPDATE assets
                    SET type = ?, role = ?, bpm = ?, musical_key = ?, style_tag = ?, tags_json = ?, updated_at = ?, active = 1
                    WHERE path = ?
                    """,
                    payload,
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO assets (id, path, type, role, bpm, musical_key, style_tag, tags_json, created_at, updated_at)
                    VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p_abs,
                        atype,
                        role,
                        bpm,
                        mkey,
                        st,
                        json.dumps(tags),
                        now,
                        now,
                    ),
                )
                inserted += 1
        conn.commit()

    return {
        "scanned": len(files),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
    }


def load_catalog_for_generation(db_path: Optional[str] = None) -> Dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path, type, role, bpm, musical_key FROM assets WHERE active = 1"
        ).fetchall()

    kicks: List[Path] = []
    hats: List[Path] = []
    claps: List[Path] = []
    loops: List[Dict] = []
    midi_files: List[Path] = []
    midi_by_role: Dict[str, List[Path]] = {
        "chords": [],
        "melodies": [],
        "basslines": [],
        "drums": [],
    }

    for r in rows:
        p = Path(r["path"])
        if not p.exists():
            continue
        atype = (r["type"] or "").lower()
        role = (r["role"] or "").lower()
        if atype == "drum":
            if role == "kick":
                kicks.append(p)
            elif role == "hat":
                hats.append(p)
            elif role == "clap":
                claps.append(p)
        elif atype == "loop":
            loops.append({"path": p, "bpm": r["bpm"], "key": r["musical_key"]})
        elif atype == "midi":
            midi_files.append(p)
            if role in midi_by_role:
                midi_by_role[role].append(p)

    return {
        "drums": {"kicks": sorted(kicks), "hats": sorted(hats), "claps": sorted(claps)},
        "loops": sorted(loops, key=lambda e: str(e["path"])),
        "midi_files": sorted(midi_files),
        "midi_by_role": {k: sorted(v) for k, v in midi_by_role.items()},
    }
