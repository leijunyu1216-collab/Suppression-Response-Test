"""Compute seed-paired descriptive tests used by the manuscript.

The output is deliberately descriptive: each comparison has three matched random
seeds, so p-values document uncertainty rather than support a high-power claim.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def _paired_rows(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> dict[str, object]:
    paired = candidate.merge(reference, on="seed", suffixes=("_candidate", "_reference"))
    diff = paired["accuracy_candidate"] - paired["accuracy_reference"]
    if len(diff) < 2:
        raise ValueError(f"{label}: fewer than two matched seeds")
    try:
        wilcoxon_p = float(wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "comparison": label,
        "n_pairs": len(diff),
        "seeds": "|".join(str(value) for value in sorted(paired["seed"].tolist())),
        "reference_mean": reference[reference["seed"].isin(paired["seed"])]["accuracy"].mean(),
        "candidate_mean": candidate[candidate["seed"].isin(paired["seed"])]["accuracy"].mean(),
        "mean_difference_candidate_minus_reference": diff.mean(),
        "difference_std": diff.std(ddof=1),
        "paired_t_p": float(ttest_rel(paired["accuracy_candidate"], paired["accuracy_reference"]).pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def main() -> None:
    wdcnn = pd.read_csv(ROOT / "results" / "wdcnn_formal_full.csv")
    fixed = pd.read_csv(ROOT / "results" / "lambda_sensitivity_fixed.csv")
    hust = pd.read_csv(ROOT / "results" / "suppression_response_hust_conditional.csv")
    rows: list[dict[str, object]] = []

    for fold in ("leave_condition_00", "leave_condition_03"):
        reference = wdcnn[
            (wdcnn["dataset"] == "Paderborn")
            & (wdcnn["protocol"] == "P2_condition_disjoint")
            & (wdcnn["fold_id"] == fold)
            & (wdcnn["model"] == "WDCNN")
        ][["seed", "accuracy"]]
        for model in ("DANN-src-condition-lam1-grlsched", "DeepCORAL-src-condition-lam1-rawcoral"):
            candidate = fixed[(fixed["fold_id"] == fold) & (fixed["model"] == model)][["seed", "accuracy"]]
            rows.append(_paired_rows(reference, candidate, f"Paderborn {fold}: {model} vs WDCNN"))

    base = hust[hust["lambda_domain"] == 0.0][["seed", "accuracy"]]
    for lam in (0.1, 0.3, 1.0):
        candidate = hust[hust["lambda_domain"] == lam][["seed", "accuracy"]]
        rows.append(_paired_rows(base, candidate, f"HUST P1: file adversary lambda={lam:g} vs base"))

    out = ROOT / "results" / "paper_stat_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
