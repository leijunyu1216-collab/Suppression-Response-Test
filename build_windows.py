from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.schema import read_records
from shortcutfd.windows import build_windows, write_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build window index CSV from record metadata.")
    parser.add_argument("--records", type=Path, default=Path("data/metadata/records.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/metadata/windows.csv"))
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows-per-record", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.records)
    windows = build_windows(
        records,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_record=args.max_windows_per_record,
    )
    write_windows(windows, args.out)
    print(f"wrote {len(windows)} windows to {args.out}")


if __name__ == "__main__":
    main()

