"""Build measured-RPM metadata for the union of confirmatory test windows."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.paderborn import load_paderborn_speed, measured_window_rpm  # noqa: E402
from shortcutfd.windows import read_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", type=Path,
        default=ROOT / "data/metadata/windows_paderborn_full3class.csv",
    )
    parser.add_argument(
        "--splits", type=Path,
        default=ROOT / "data/splits/paderborn_inner_confirmatory_v1",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data/metadata/window_rpm_paderborn_inner_confirmatory_v1.csv",
    )
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()
    requested = set()
    split_paths = sorted(args.splits.glob("inner_ki*.csv"))
    if args.max_folds:
        split_paths = split_paths[: args.max_folds]
    for split_path in split_paths:
        with split_path.open(newline="", encoding="utf-8") as handle:
            requested.update(
                row["window_id"] for row in csv.DictReader(handle) if row["split"] == "test"
            )
    selected = [window for window in read_windows(args.windows) if window.window_id in requested]
    if len(selected) != len(requested):
        raise ValueError(f"found {len(selected)} of {len(requested)} requested test windows")
    by_path = defaultdict(list)
    for window in selected:
        by_path[window.path].append(window)
    rows = []
    for index, (path, windows) in enumerate(sorted(by_path.items()), 1):
        rpm, vibration_samples = load_paderborn_speed(path)
        rows.extend(measured_window_rpm(window, rpm, vibration_samples) for window in windows)
        if index % 50 == 0 or index == len(by_path):
            print(f"processed {index}/{len(by_path)} records", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} measured-RPM rows to {args.out}")


if __name__ == "__main__":
    main()
