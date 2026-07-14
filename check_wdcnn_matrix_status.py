from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from run_wdcnn_matrix import PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check WDCNN matrix completion status.")
    parser.add_argument("--input", type=Path, default=Path("results/wdcnn_formal_full.csv"))
    parser.add_argument("--preset", choices=sorted(PRESETS), action="append", required=True)
    parser.add_argument("--seeds", default="13,21,42")
    parser.add_argument("--missing-out", type=Path, default=Path("results/wdcnn_formal_full_missing.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    expected = []
    for preset_name in args.preset:
        preset = PRESETS[preset_name]
        for seed in seeds:
            for protocol, fold_id in preset["folds"]:
                expected.append(
                    {
                        "dataset": preset["dataset"],
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "seed": seed,
                    }
                )

    if args.input.exists() and args.input.stat().st_size > 0:
        actual_df = pd.read_csv(args.input)
        actual = {
            (row["dataset"], row["protocol"], row["fold_id"], int(row["seed"]))
            for _, row in actual_df.iterrows()
        }
    else:
        actual = set()

    missing = [
        row
        for row in expected
        if (row["dataset"], row["protocol"], row["fold_id"], row["seed"]) not in actual
    ]
    args.missing_out.parent.mkdir(parents=True, exist_ok=True)
    with args.missing_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "protocol", "fold_id", "seed"])
        writer.writeheader()
        writer.writerows(missing)

    print(f"expected={len(expected)} completed={len(expected) - len(missing)} missing={len(missing)}")
    print(f"missing list: {args.missing_out}")


if __name__ == "__main__":
    main()

