"""Relate BPFO/BPFI profile coverage to balanced-K4 class recall."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LABEL_MECHANISM = {"inner_race": "bpfi", "outer_race": "bpfo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out-rows", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = pd.read_csv(args.profiles)
    rotation = _add_recalls(pd.read_csv(args.rotation))
    vectors = _profile_vectors(profiles)
    test_bearings = _test_bearings(args.windows, args.splits)
    run_means = rotation.groupby(["fold_id", "rotation_index"], as_index=False).agg(
        selected_bearings=("selected_bearings", "first"),
        inner_race=("inner_race", "mean"),
        outer_race=("outer_race", "mean"),
    )
    rows = []
    for _, run in run_means.iterrows():
        selected = _parse_selected(run.selected_bearings)
        for label, primary in LABEL_MECHANISM.items():
            for mechanism in (primary, "bpfo" if primary == "bpfi" else "bpfi"):
                source = sorted(selected[label])
                target = test_bearings[str(run.fold_id)][label]
                source_values = np.stack([vectors[(bearing, mechanism)] for bearing in source])
                target_value = vectors[(target, mechanism)]
                scale = _mechanism_scale(vectors, label, mechanism)
                normalized_source = source_values / scale
                normalized_target = target_value / scale
                centroid_distance = float(np.linalg.norm(normalized_target - normalized_source.mean(axis=0)))
                minimum_distance = float(np.min(np.linalg.norm(normalized_source - normalized_target, axis=1)))
                rows.append({
                    "fold_id": run.fold_id,
                    "rotation_index": int(run.rotation_index),
                    "fault_label": label,
                    "mechanism": mechanism,
                    "is_primary_mechanism": mechanism == primary,
                    "test_bearing": target,
                    "source_bearings": "|".join(source),
                    "centroid_distance": centroid_distance,
                    "minimum_distance": minimum_distance,
                    "class_recall": float(run[label]),
                })
    frame = pd.DataFrame(rows)
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_rows, index=False)
    summary = {
        "rows": int(len(frame)),
        "independent_units": 6,
        "analyses": {},
    }
    for (label, mechanism), group in frame.groupby(["fault_label", "mechanism"]):
        key = f"{label}:{mechanism}"
        summary["analyses"][key] = _association(group)
        summary["analyses"][key]["primary_mechanism"] = bool(group.is_primary_mechanism.iloc[0])
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _profile_vectors(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    harmonic_columns = sorted(
        [column for column in frame if column.startswith("h") and column.endswith("_snr_db")],
        key=lambda value: int(value.split("_", 1)[0][1:]),
    )
    conditions = sorted(frame.condition_id.unique())
    output = {}
    for (bearing, mechanism), group in frame.groupby(["bearing_id", "mechanism"]):
        indexed = group.set_index("condition_id")
        if set(indexed.index) != set(conditions):
            raise ValueError(f"incomplete profile for {bearing}:{mechanism}")
        output[(bearing, mechanism)] = indexed.loc[conditions, harmonic_columns].to_numpy().reshape(-1)
    return output


def _mechanism_scale(vectors: dict, label: str, mechanism: str) -> np.ndarray:
    prefix = "KI" if label == "inner_race" else "KA"
    array = np.stack([value for (bearing, name), value in vectors.items() if name == mechanism and bearing.startswith(prefix)])
    scale = array.std(axis=0, ddof=1)
    return np.where(scale > 1e-6, scale, 1.0)


def _test_bearings(windows_path: Path, splits_path: Path) -> dict[str, dict[str, str]]:
    bearing = {}
    label = {}
    with windows_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bearing[row["window_id"]] = row["bearing_id"]
            label[row["window_id"]] = row["fault_label"]
    output = {}
    files = sorted(splits_path.glob("*.csv")) if splits_path.is_dir() else [splits_path]
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["protocol"] != "P3_bearing_disjoint" or row["split"] != "test":
                    continue
                fold = row["fold_id"]
                output.setdefault(fold, {})[label[row["window_id"]]] = bearing[row["window_id"]]
    return output


def _association(group: pd.DataFrame) -> dict:
    centered_x = group.centroid_distance - group.groupby("fold_id").centroid_distance.transform("mean")
    centered_y = group.class_recall - group.groupby("fold_id").class_recall.transform("mean")
    slope = float(np.dot(centered_x, centered_y) / max(np.dot(centered_x, centered_x), 1e-12))
    pearson = float(np.corrcoef(centered_x, centered_y)[0, 1])
    spearman = float(spearmanr(centered_x, centered_y).statistic)
    rng = np.random.default_rng(20260713)
    folds = sorted(group.fold_id.unique())
    boot = []
    for _ in range(10000):
        sampled = rng.choice(folds, len(folds), replace=True)
        pieces = [group[group.fold_id == fold] for fold in sampled]
        sample = pd.concat(pieces, ignore_index=True)
        x = sample.centroid_distance - sample.groupby(sample.index // 5).centroid_distance.transform("mean")
        y = sample.class_recall - sample.groupby(sample.index // 5).class_recall.transform("mean")
        boot.append(float(np.dot(x, y) / max(np.dot(x, x), 1e-12)))
    permutations = []
    base = group.copy()
    for _ in range(10000):
        shuffled = base.groupby("fold_id").class_recall.transform(
            lambda values: rng.permutation(values.to_numpy())
        )
        y = shuffled - shuffled.groupby(base.fold_id).transform("mean")
        permutations.append(float(np.corrcoef(centered_x, y)[0, 1]))
    p_value = float((1 + np.sum(np.abs(permutations) >= abs(pearson))) / (1 + len(permutations)))
    return {
        "n_fold_rotations": int(len(group)),
        "within_fold_slope": slope,
        "cluster_bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "within_fold_pearson": pearson,
        "within_fold_spearman": spearman,
        "within_fold_permutation_p": p_value,
    }


def _add_recalls(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.recall_by_class.map(
        lambda value: {part.split(":", 1)[0]: float(part.split(":", 1)[1]) for part in value.split("|")}
    )
    frame = frame.copy()
    for label in ("inner_race", "outer_race"):
        frame[label] = parsed.map(lambda values: values[label])
    return frame


def _parse_selected(value: str) -> dict[str, set[str]]:
    return {
        part.split("=", 1)[0]: set(part.split("=", 1)[1].split("|"))
        for part in str(value).split(";")
    }


if __name__ == "__main__":
    main()
