"""Audit whether local Paderborn data support individual-disjoint protocols."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.schema import Record, read_records  # noqa: E402


OFFICIAL_COUNTS = {
    "normal": 6,
    "outer_race": 12,
    "inner_race": 11,
    "compound_inner_outer": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paderborn_bearing_audit"))
    parser.add_argument("--min-bearings-per-class", type=int, default=4)
    parser.add_argument("--required-conditions", type=int, default=4)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the requested design is unsupported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [record for record in read_records(args.records) if record.dataset.lower() == "paderborn"]
    if not records:
        raise SystemExit(f"No Paderborn records found in {args.records}")

    inventory = _bearing_inventory(records, args.required_conditions)
    class_summary = _class_summary(inventory, args.min_bearings_per_class)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "bearing_inventory.csv", inventory)
    _write_csv(args.out_dir / "class_summary.csv", class_summary)

    all_supported = all(row["supports_requested_split"] == "yes" for row in class_summary)
    metadata_mismatches = sum(record.fault_label != _fault_location(record.bearing_id) for record in records)
    n_bearings = len(inventory)
    print(
        f"records={len(records)} physical_bearings={n_bearings} "
        f"metadata_label_mismatches={metadata_mismatches}"
    )
    for row in class_summary:
        expected = OFFICIAL_COUNTS.get(str(row["fault_label"]), "unknown")
        print(
            f"{row['fault_label']}: local={row['n_bearings']} official={expected} "
            f"complete={row['n_complete_bearings']} supports_split={row['supports_requested_split']}"
        )
    print(f"wrote audit CSVs to {args.out_dir}")
    if args.strict and (not all_supported or metadata_mismatches):
        raise SystemExit(
            "Local data do not support a trustworthy individual-disjoint design. "
            "Download more bearings and/or rebuild metadata/windows with corrected fault-location labels."
        )


def _bearing_inventory(records: list[Record], required_conditions: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        grouped[(_fault_location(record.bearing_id), record.bearing_id)].append(record)
    rows: list[dict[str, object]] = []
    for (fault_label, bearing_id), values in sorted(grouped.items()):
        condition_counts = Counter(record.condition_id for record in values)
        rows.append(
            {
                "fault_label": fault_label,
                "bearing_id": bearing_id,
                "metadata_labels": "|".join(sorted({record.fault_label for record in values})),
                "n_records": len(values),
                "n_conditions": len(condition_counts),
                "conditions": "|".join(sorted(condition_counts)),
                "min_records_per_condition": min(condition_counts.values()),
                "max_records_per_condition": max(condition_counts.values()),
                "complete_conditions": "yes" if len(condition_counts) >= required_conditions else "no",
            }
        )
    return rows


def _fault_location(bearing_id: str) -> str:
    upper = bearing_id.upper()
    if upper.startswith("KA"):
        return "outer_race"
    if upper.startswith("KI"):
        return "inner_race"
    if upper.startswith("KB"):
        return "compound_inner_outer"
    if upper.startswith("K") and upper[1:].isdigit():
        return "normal"
    return "unknown"


def _class_summary(inventory: list[dict[str, object]], min_bearings: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in inventory:
        grouped[str(row["fault_label"])].append(row)
    rows: list[dict[str, object]] = []
    for fault_label, values in sorted(grouped.items()):
        complete = [row for row in values if row["complete_conditions"] == "yes"]
        rows.append(
            {
                "fault_label": fault_label,
                "n_bearings": len(values),
                "n_complete_bearings": len(complete),
                "bearing_ids": "|".join(str(row["bearing_id"]) for row in values),
                "min_required": min_bearings,
                "supports_requested_split": "yes" if len(complete) >= min_bearings else "no",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
