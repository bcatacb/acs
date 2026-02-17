import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.library_index import ingest_assets, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest asset files into ACS SQLite index.")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "asset_drop"),
        help="Root directory to scan (default: project asset_drop)",
    )
    parser.add_argument("--style", default=None, help="Optional forced style tag for all ingested files.")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional extra tag (can be repeated). Example: --tag dark_fast",
    )
    args = parser.parse_args()

    init_db()
    stats = ingest_assets(
        root_dir=args.root,
        style_tag=args.style,
        extra_tags=args.tag,
    )
    print(
        f"ingest complete | scanned={stats['scanned']} inserted={stats['inserted']} "
        f"updated={stats['updated']} skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
