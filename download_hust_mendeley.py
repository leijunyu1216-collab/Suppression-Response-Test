from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://data.mendeley.com/public-api/zip/cbv7jyx4p9/download/3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HUST Bearing from Mendeley Data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/HUST"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "hust_bearing_mendeley_v3.zip"
    if not zip_path.exists() or args.overwrite:
        print(f"download {URL} -> {zip_path}")
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with zip_path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
    else:
        print(f"skip existing {zip_path}")

    if args.no_extract:
        return
    extract_dir = args.out_dir / "mendeley_v3"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"extract {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    print("done")


if __name__ == "__main__":
    main()

