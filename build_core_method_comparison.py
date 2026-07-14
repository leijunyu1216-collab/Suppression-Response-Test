from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_FOLDS = {
    "HUST": {"fold0", "leave_condition_04"},
    "Paderborn": {"fold0", "leave_condition_00"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WDCNN + DA core method comparison table.")
    parser.add_argument("--wdcnn-summary", type=Path, default=Path("results/wdcnn_formal_full_summary.csv"))
    parser.add_argument("--domain-summary", type=Path, default=Path("results/domain_baselines_core_summary.csv"))
    parser.add_argument(
        "--models",
        default="DANN-src-condition-lam0p1,DeepCORAL-src-condition-lam1e06",
        help="Comma-separated domain-baseline models to include.",
    )
    parser.add_argument("--summary-out", type=Path, default=Path("results/core_method_comparison_summary.csv"))
    parser.add_argument("--drop-out", type=Path, default=Path("results/core_method_comparison_drop.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wdcnn = pd.read_csv(args.wdcnn_summary)
    domain = pd.read_csv(args.domain_summary)
    models = {model.strip() for model in args.models.split(",") if model.strip()}

    wdcnn_core = _filter_core_rows(wdcnn)
    domain_core = _filter_core_rows(domain[domain["model"].isin(models)])
    combined = pd.concat([wdcnn_core, domain_core], ignore_index=True)
    combined = combined.sort_values(["dataset", "model", "protocol", "fold_id"]).reset_index(drop=True)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.summary_out, index=False)
    _make_drop_table(combined).to_csv(args.drop_out, index=False)
    print(f"wrote {len(combined)} rows to {args.summary_out}")
    print(f"wrote drop table to {args.drop_out}")


def _filter_core_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        if dataset in CORE_FOLDS and str(row["fold_id"]) in CORE_FOLDS[dataset]:
            keep.append(row)
    if not keep:
        return df.iloc[0:0].copy()
    return pd.DataFrame(keep)


def _make_drop_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        p0 = group[(group["protocol"] == "P0_random_window") & (group["fold_id"] == "fold0")]
        if p0.empty:
            continue
        p0_acc = float(p0.iloc[0]["accuracy_mean"])
        p0_f1 = float(p0.iloc[0]["macro_f1_mean"])
        for _, row in group.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "protocol": row["protocol"],
                    "fold_id": row["fold_id"],
                    "n_runs": row["n_runs"],
                    "accuracy_mean": row["accuracy_mean"],
                    "accuracy_std": row["accuracy_std"],
                    "macro_f1_mean": row["macro_f1_mean"],
                    "macro_f1_std": row["macro_f1_std"],
                    "accuracy_drop_vs_p0": p0_acc - float(row["accuracy_mean"]),
                    "macro_f1_drop_vs_p0": p0_f1 - float(row["macro_f1_mean"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "protocol", "fold_id"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
