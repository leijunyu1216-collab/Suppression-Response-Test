from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CWRU harder-label window CSV using fault type plus fault size."
    )
    parser.add_argument("--windows", type=Path, default=Path("data/metadata/windows.csv"))
    parser.add_argument("--records", type=Path, default=Path("data/metadata/records.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/metadata/windows_cwru_harder.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = pd.read_csv(args.windows)
    records = pd.read_csv(args.records)

    cwru = windows[windows["dataset"] == "CWRU"].copy()
    if cwru.empty:
        raise SystemExit("No CWRU rows found in windows CSV.")

    record_sizes = records[records["dataset"] == "CWRU"][["record_id", "fault_size"]].drop_duplicates()
    cwru = cwru.merge(record_sizes, on="record_id", how="left", validate="many_to_one")
    if cwru["fault_size"].isna().any():
        missing = cwru.loc[cwru["fault_size"].isna(), "record_id"].drop_duplicates().head(10).tolist()
        raise SystemExit(f"Missing fault_size for records: {missing}")

    cwru["fault_label"] = [
        "normal" if label == "normal" else f"{label}_{str(size).replace('.', 'p')}"
        for label, size in zip(cwru["fault_label"], cwru["fault_size"])
    ]
    cwru = cwru.drop(columns=["fault_size"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cwru.to_csv(args.out, index=False)
    print(f"wrote {len(cwru)} CWRU hard-label windows to {args.out}")
    print("label counts:")
    print(cwru["fault_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
