"""Stream P3 physical-bearing-disjoint assignments without materializing P0--P4."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.splits import SPLIT_COLUMNS, _bearing_fold_assignments, _row  # noqa: E402
from shortcutfd.windows import read_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-bearings-per-class", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    windows = read_windows(args.windows)
    assignments = _bearing_fold_assignments(
        windows, args.folds, args.seed, args.min_bearings_per_class
    )
    if args.dry_run:
        print(f"windows={len(windows)} folds={len(assignments)} planned_rows={len(windows) * len(assignments)}")
        for index, (test_bearings, val_bearings) in enumerate(assignments):
            print(
                f"leave_bearing_{index:02d} test={'|'.join(sorted(test_bearings))} "
                f"val={'|'.join(sorted(val_bearings))}"
            )
        return
    audit = defaultdict(lambda: {"windows": 0, "records": set(), "labels": set(), "conditions": set(), "bearings": set()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fold_index, (test_bearings, val_bearings) in enumerate(assignments):
        fold_id = f"leave_bearing_{fold_index:02d}"
        fold_path = args.out_dir / f"{fold_id}.csv"
        with fold_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
            writer.writeheader()
            for window in windows:
                split = (
                    "test" if window.bearing_id in test_bearings
                    else "val" if window.bearing_id in val_bearings
                    else "train"
                )
                row = _row("P3_bearing_disjoint", fold_id, split, window)
                writer.writerow({column: getattr(row, column) for column in SPLIT_COLUMNS})
                cell = audit[(fold_id, split)]
                cell["windows"] += 1
                cell["records"].add(window.record_id)
                cell["labels"].add(window.fault_label)
                cell["conditions"].add(window.condition_id)
                cell["bearings"].add(window.bearing_id)
                written += 1
        print(f"wrote {len(windows)} rows to {fold_path}", flush=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        fields = ["protocol", "fold_id", "split", "n_windows", "n_records", "n_labels", "n_conditions", "n_bearings"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (fold_id, split), cell in sorted(audit.items()):
            writer.writerow({
                "protocol": "P3_bearing_disjoint", "fold_id": fold_id, "split": split,
                "n_windows": cell["windows"], "n_records": len(cell["records"]),
                "n_labels": len(cell["labels"]), "n_conditions": len(cell["conditions"]),
                "n_bearings": len(cell["bearings"]),
            })
    print(f"wrote {written} P3 assignments across {len(assignments)} fold files")
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
