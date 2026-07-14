"""Build frozen, physical-bearing-disjoint folds for the inner-race extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.splits import SPLIT_COLUMNS, _row  # noqa: E402
from shortcutfd.windows import Window, read_windows  # noqa: E402


NORMAL = ["K001", "K002", "K003", "K004", "K005", "K006"]
OUTER = ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", type=Path,
        default=ROOT / "data/metadata/windows_paderborn_full3class.csv",
    )
    parser.add_argument(
        "--ontology", type=Path,
        default=ROOT / "data/metadata/paderborn_inner_damage_ontology.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "data/splits/paderborn_inner_confirmatory_v1",
    )
    parser.add_argument("--train-windows-per-bearing", type=int, default=625)
    parser.add_argument("--eval-windows-per-bearing", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ontology = _read_ontology(args.ontology)
    inner = sorted(ontology)
    windows = read_windows(args.windows)
    by_bearing: dict[str, list[Window]] = defaultdict(list)
    for window in windows:
        if window.bearing_id in set(NORMAL + OUTER + inner):
            by_bearing[window.bearing_id].append(window)
    _validate_inventory(by_bearing, ontology)

    folds = [_fold(index, bearing, ontology) for index, bearing in enumerate(inner)]
    if args.dry_run:
        for fold in folds:
            print(_fold_text(fold))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        split_path = args.out_dir / f"{fold_id}.csv"
        rows_written = 0
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
            writer.writeheader()
            for split in ("train", "val", "test"):
                cap = (
                    args.train_windows_per_bearing
                    if split == "train" else args.eval_windows_per_bearing
                )
                for label, bearing in fold[split]:
                    selected = _record_balanced_cap(
                        by_bearing[bearing], cap,
                        f"{args.seed}:{fold_id}:{split}:{bearing}",
                    )
                    for window in selected:
                        row = _row("P3_inner_confirmatory_v1", fold_id, split, window)
                        writer.writerow({column: getattr(row, column) for column in SPLIT_COLUMNS})
                    rows_written += len(selected)
                    ontology_row = ontology.get(bearing, {})
                    manifest_rows.append({
                        "fold_id": fold_id,
                        "split": split,
                        "fault_label": label,
                        "bearing_id": bearing,
                        "analysis_stratum": ontology_row.get("analysis_stratum", "reference"),
                        "primary_confirmatory": ontology_row.get("primary_confirmatory", ""),
                        "n_available_windows": len(by_bearing[bearing]),
                        "n_selected_windows": len(selected),
                    })
            summary_rows.append({
                "fold_id": fold_id,
                "test_inner_bearing": fold["test_inner_bearing"],
                "test_inner_stratum": fold["test_inner_stratum"],
                "primary_confirmatory": fold["primary_confirmatory"],
                "n_rows": rows_written,
                "train_bearings": _bearing_text(fold["train"]),
                "val_bearings": _bearing_text(fold["val"]),
                "test_bearings": _bearing_text(fold["test"]),
            })
        print(f"wrote {rows_written} rows to {split_path}", flush=True)

    _write_csv(args.out_dir / "fold_manifest.csv", manifest_rows)
    _write_csv(args.out_dir / "fold_summary.csv", summary_rows)
    _validate_folds(manifest_rows, ontology)
    print(f"validated {len(folds)} physical-bearing folds")


def _read_ontology(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["bearing_id"]: row for row in rows}


def _fold(index: int, test_inner: str, ontology) -> dict[str, object]:
    artificial = sorted(
        bearing for bearing, row in ontology.items()
        if row["analysis_stratum"] == "artificial_single"
    )
    fatigue = sorted(
        bearing for bearing, row in ontology.items()
        if row["analysis_stratum"].startswith("fatigue_")
    )
    mixed = sorted(
        bearing for bearing, row in ontology.items()
        if row["analysis_stratum"] == "mixed_ir_or"
    )
    stratum = ontology[test_inner]["analysis_stratum"]
    if stratum == "artificial_single":
        val_group = artificial
    elif stratum.startswith("fatigue_"):
        val_group = fatigue
    else:
        val_group = mixed
    val_inner = val_group[(val_group.index(test_inner) + 1) % len(val_group)]

    available_artificial = [b for b in artificial if b not in {test_inner, val_inner}]
    available_fatigue = [b for b in fatigue if b not in {test_inner, val_inner}]
    train_inner = _cyclic_take(available_artificial, 2, index) + _cyclic_take(
        available_fatigue, 2, index
    )

    test_normal = NORMAL[index % len(NORMAL)]
    val_normal = NORMAL[(index + 1) % len(NORMAL)]
    train_normal = [b for b in NORMAL if b not in {test_normal, val_normal}]
    test_outer = OUTER[index % len(OUTER)]
    val_outer = OUTER[(index + 2) % len(OUTER)]
    train_outer = [b for b in OUTER if b not in {test_outer, val_outer}]

    fold_id = f"inner_{test_inner.lower()}"
    return {
        "fold_id": fold_id,
        "test_inner_bearing": test_inner,
        "test_inner_stratum": stratum,
        "primary_confirmatory": ontology[test_inner]["primary_confirmatory"],
        "train": (
            [("normal", b) for b in train_normal]
            + [("outer_race", b) for b in train_outer]
            + [("inner_race", b) for b in train_inner]
        ),
        "val": [
            ("normal", val_normal), ("outer_race", val_outer), ("inner_race", val_inner),
        ],
        "test": [
            ("normal", test_normal), ("outer_race", test_outer), ("inner_race", test_inner),
        ],
    }


def _cyclic_take(values: list[str], count: int, offset: int) -> list[str]:
    if len(values) < count:
        raise ValueError(f"requested {count} source bearings from only {len(values)}")
    return [values[(offset + index) % len(values)] for index in range(count)]


def _record_balanced_cap(windows: list[Window], cap: int, salt: str) -> list[Window]:
    if len(windows) < cap:
        raise ValueError(f"{windows[0].bearing_id}: {len(windows)} windows < cap {cap}")
    groups: dict[str, list[Window]] = defaultdict(list)
    for window in windows:
        groups[window.record_id].append(window)
    ranked = {
        record: sorted(values, key=lambda w: _hash(f"{salt}:{w.window_id}"))
        for record, values in groups.items()
    }
    selected: list[Window] = []
    depth = 0
    records = sorted(ranked, key=lambda value: _hash(f"{salt}:{value}"))
    while len(selected) < cap:
        added = False
        for record in records:
            if depth < len(ranked[record]):
                selected.append(ranked[record][depth])
                added = True
                if len(selected) == cap:
                    break
        if not added:
            break
        depth += 1
    return selected


def _validate_inventory(by_bearing, ontology) -> None:
    expected = set(NORMAL + OUTER + list(ontology))
    missing = sorted(expected - set(by_bearing))
    if missing:
        raise ValueError(f"missing bearings in windows metadata: {missing}")
    for bearing in sorted(expected):
        labels = {window.fault_label for window in by_bearing[bearing]}
        conditions = {window.condition_id for window in by_bearing[bearing]}
        if len(labels) != 1:
            raise ValueError(f"{bearing}: multiple labels {labels}")
        if len(conditions) != 4:
            raise ValueError(f"{bearing}: expected four conditions, found {len(conditions)}")


def _validate_folds(rows, ontology) -> None:
    by_fold: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_fold[str(row["fold_id"])].append(row)
    tested_inner = []
    for fold_id, fold_rows in by_fold.items():
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in fold_rows:
            assignments[str(row["bearing_id"])].add(str(row["split"]))
        overlap = {bearing: parts for bearing, parts in assignments.items() if len(parts) != 1}
        if overlap:
            raise ValueError(f"{fold_id}: bearing leakage {overlap}")
        for split, expected in (("train", 12), ("val", 3), ("test", 3)):
            actual = sum(row["split"] == split for row in fold_rows)
            if actual != expected:
                raise ValueError(f"{fold_id}: {split} has {actual} bearings, expected {expected}")
        test_inner = [
            str(row["bearing_id"]) for row in fold_rows
            if row["split"] == "test" and row["fault_label"] == "inner_race"
        ]
        if len(test_inner) != 1:
            raise ValueError(f"{fold_id}: expected one test inner bearing")
        tested_inner.extend(test_inner)
    if sorted(tested_inner) != sorted(ontology):
        raise ValueError("each inner-race bearing must be tested exactly once")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bearing_text(rows) -> str:
    return ";".join(f"{label}={bearing}" for label, bearing in rows)


def _fold_text(fold) -> str:
    return (
        f"{fold['fold_id']} primary={fold['primary_confirmatory']} "
        f"stratum={fold['test_inner_stratum']} "
        f"train=[{_bearing_text(fold['train'])}] "
        f"val=[{_bearing_text(fold['val'])}] test=[{_bearing_text(fold['test'])}]"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
