"""Audit SRT labels across practical accuracy margins.

The output separates the legacy-linear provisional readout from the final
capacity-aware readout, which requires both the legacy linear and two-layer
MLP manipulation checks to verify suppression.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.002, 0.003, 0.005, 0.01])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/srt_margin_sensitivity.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _paderborn_cases() + _hust_cases() + _synthetic_cases()
    rows: list[dict[str, str]] = []
    for case in cases:
        for epsilon in args.epsilons:
            linear_verified = case["linear_eta_low"] > 0
            mlp_available = not math.isnan(case["mlp_eta_low"])
            capacity_verified = linear_verified and (
                case["mlp_eta_low"] > 0 if mlp_available else case["direct_marker"]
            )
            rows.append(
                {
                    "dataset": case["dataset"],
                    "case_id": case["case_id"],
                    "n_pairs": str(case["n_pairs"]),
                    "epsilon_A": f"{epsilon:.3f}",
                    "delta_mean": f"{case['delta_mean']:.9f}",
                    "delta_ci_low": f"{case['delta_low']:.9f}",
                    "delta_ci_high": f"{case['delta_high']:.9f}",
                    "linear_eta_ci_low": _format(case["linear_eta_low"]),
                    "linear_eta_ci_high": _format(case["linear_eta_high"]),
                    "mlp_eta_ci_low": _format(case["mlp_eta_low"]),
                    "mlp_eta_ci_high": _format(case["mlp_eta_high"]),
                    "primary_linear_readout": _classify(
                        case["delta_low"], case["delta_high"], epsilon, linear_verified
                    ),
                    "capacity_aware_readout": _classify(
                        case["delta_low"], case["delta_high"], epsilon, capacity_verified
                    ),
                    "capacity_rule": (
                        "direct_marker_probe"
                        if case["direct_marker"]
                        else "legacy_linear_and_mlp_2x"
                    ),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


def _paderborn_cases() -> list[dict[str, object]]:
    responses = _read(ROOT / "results/paderborn_matched_srt_summary.csv")
    capacity = _capacity_lookup()
    cases = []
    for row in responses:
        if row["method"] != "deepcoral":
            continue
        fold = row["fold_id"]
        linear = capacity[("Paderborn", fold, "1", "legacy_linear")]
        mlp = capacity[("Paderborn", fold, "1", "mlp_2x")]
        cases.append(
            _case(
                "Paderborn", fold.replace("leave_condition_", "c"), int(row["n_pairs"]),
                float(row["delta_accuracy_mean"]), float(row["delta_accuracy_ci_low"]),
                float(row["delta_accuracy_ci_high"]), linear, mlp,
            )
        )
    return cases


def _hust_cases() -> list[dict[str, object]]:
    rows = _read(ROOT / "results/suppression_response_hust_conditional.csv")
    by_strength: dict[float, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_strength[float(row["lambda_domain"])][int(row["seed"])] = float(row["accuracy"])
    capacity = _capacity_lookup()
    cases = []
    for strength in (0.1, 0.3, 1.0):
        seeds = sorted(set(by_strength[0.0]) & set(by_strength[strength]))
        deltas = [by_strength[strength][seed] - by_strength[0.0][seed] for seed in seeds]
        low, high = _ci95(deltas)
        key_strength = f"{strength:g}"
        linear = capacity[("HUST", "fold0", key_strength, "legacy_linear")]
        mlp = capacity[("HUST", "fold0", key_strength, "mlp_2x")]
        cases.append(
            _case(
                "HUST", f"file_lambda_{key_strength}", len(deltas), statistics.mean(deltas),
                low, high, linear, mlp,
            )
        )
    return cases


def _synthetic_cases() -> list[dict[str, object]]:
    cases = []
    for row in _read(ROOT / "results/synthetic_srt_marker_summary.csv"):
        strength = float(row["lambda"])
        if strength <= 0:
            continue
        eta = {
            "eta_ci_low": row["eta_marker_ci_low"],
            "eta_ci_high": row["eta_marker_ci_high"],
        }
        case = _case(
            "Synthetic", f"marker_lambda_{strength:g}", int(row["n_pairs"]),
            float(row["delta_accuracy_mean"]), float(row["delta_accuracy_ci_low"]),
            float(row["delta_accuracy_ci_high"]), eta, None,
        )
        case["direct_marker"] = True
        cases.append(case)
    return cases


def _case(dataset, case_id, n_pairs, mean, low, high, linear, mlp):
    return {
        "dataset": dataset,
        "case_id": case_id,
        "n_pairs": n_pairs,
        "delta_mean": mean,
        "delta_low": low,
        "delta_high": high,
        "linear_eta_low": float(linear["eta_ci_low"]),
        "linear_eta_high": float(linear["eta_ci_high"]),
        "mlp_eta_low": float(mlp["eta_ci_low"]) if mlp else math.nan,
        "mlp_eta_high": float(mlp["eta_ci_high"]) if mlp else math.nan,
        "direct_marker": False,
    }


def _capacity_lookup() -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = _read(ROOT / "results/probe_capacity_sensitivity_summary.csv")
    return {
        (row["dataset"], row["fold_id"], row["active_strength"], row["probe_model"]): row
        for row in rows
    }


def _classify(low: float, high: float, epsilon: float, manipulation: bool) -> str:
    if not manipulation:
        return "inconclusive_manipulation"
    if low > epsilon:
        return "shortcut_response"
    if high < -epsilon:
        return "entanglement_response"
    if low >= -epsilon and high <= epsilon:
        return "neutral_response"
    return "inconclusive_response"


def _ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    half = float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def _format(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.9f}"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
