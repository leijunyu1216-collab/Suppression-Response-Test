from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.splits import (
    make_p0_random_window,
    make_p1_record_wise,
    make_p2_condition_disjoint,
    make_p3_bearing_disjoint,
    make_p4_bearing_condition_disjoint,
    summarize_splits,
    write_splits,
)
from shortcutfd.windows import read_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-aware split CSV files.")
    parser.add_argument("--windows", type=Path, default=Path("data/metadata/windows.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--include-bearing-protocols", action="store_true")
    parser.add_argument("--bearing-folds", type=int, default=3)
    parser.add_argument("--min-bearings-per-class", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = read_windows(args.windows)
    rows = []
    rows.extend(
        make_p0_random_window(
            windows,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
    )
    rows.extend(
        make_p1_record_wise(
            windows,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
    )
    rows.extend(make_p2_condition_disjoint(windows, val_fraction=args.val_fraction, seed=args.seed))
    if args.include_bearing_protocols:
        rows.extend(
            make_p3_bearing_disjoint(
                windows,
                n_folds=args.bearing_folds,
                seed=args.seed,
                min_bearings_per_class=args.min_bearings_per_class,
            )
        )
        rows.extend(
            make_p4_bearing_condition_disjoint(
                windows,
                n_bearing_folds=args.bearing_folds,
                seed=args.seed,
                min_bearings_per_class=args.min_bearings_per_class,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "p0_p1_p2_p3_p4" if args.include_bearing_protocols else "p0_p1_p2"
    split_path = args.out_dir / f"splits_{suffix}.csv"
    summary_path = args.out_dir / f"splits_{suffix}_summary.csv"
    write_splits(rows, split_path)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        summary = summarize_splits(rows)
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol", "fold_id", "split", "n_windows", "n_records",
                "n_labels", "n_conditions", "n_bearings",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(rows)} split assignments to {split_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
