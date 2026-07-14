from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.metadata_builders import discover_cwru, discover_hust, discover_paderborn, merge_records
from shortcutfd.schema import validate_records, write_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CWRU/HUST metadata CSV.")
    parser.add_argument("--cwru-root", type=Path, default=None, help="Path to raw CWRU files.")
    parser.add_argument("--hust-root", type=Path, default=None, help="Path to raw HUST files.")
    parser.add_argument("--paderborn-root", type=Path, default=None, help="Path to extracted Paderborn files.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/metadata/records.csv"),
        help="Output metadata CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = []
    if args.cwru_root:
        groups.append(discover_cwru(args.cwru_root))
    if args.hust_root:
        groups.append(discover_hust(args.hust_root))
    if args.paderborn_root:
        groups.append(discover_paderborn(args.paderborn_root))
    if not groups:
        raise SystemExit("Provide at least one of --cwru-root, --hust-root, or --paderborn-root.")

    records = merge_records(groups)
    validate_records(records)
    write_records(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
