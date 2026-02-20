from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parent.parent.parent / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSET_DROP_DIR = PROJECT_ROOT / "asset_drop"
