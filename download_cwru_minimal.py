from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


CWRU_BASE = "https://engineering.case.edu/sites/default/files"


MINIMAL_FILES = {
    # Normal baseline, loads 0-3.
    "normal/load_0/97.mat": "97.mat",
    "normal/load_1/98.mat": "98.mat",
    "normal/load_2/99.mat": "99.mat",
    "normal/load_3/100.mat": "100.mat",
    # 12 kHz drive-end faults, 0.007 inch, loads 0-3.
    "12k_de/IR007/load_0/105.mat": "105.mat",
    "12k_de/IR007/load_1/106.mat": "106.mat",
    "12k_de/IR007/load_2/107.mat": "107.mat",
    "12k_de/IR007/load_3/108.mat": "108.mat",
    "12k_de/B007/load_0/118.mat": "118.mat",
    "12k_de/B007/load_1/119.mat": "119.mat",
    "12k_de/B007/load_2/120.mat": "120.mat",
    "12k_de/B007/load_3/121.mat": "121.mat",
    "12k_de/OR007@6/load_0/130.mat": "130.mat",
    "12k_de/OR007@6/load_1/131.mat": "131.mat",
    "12k_de/OR007@6/load_2/132.mat": "132.mat",
    "12k_de/OR007@6/load_3/133.mat": "133.mat",
}


FULL_12K_DE_FILES = {
    **MINIMAL_FILES,
    # 0.007 inch, outer race additional positions.
    "12k_de/OR007@3/load_0/144.mat": "144.mat",
    "12k_de/OR007@3/load_1/145.mat": "145.mat",
    "12k_de/OR007@3/load_2/146.mat": "146.mat",
    "12k_de/OR007@3/load_3/147.mat": "147.mat",
    "12k_de/OR007@12/load_0/156.mat": "156.mat",
    "12k_de/OR007@12/load_1/158.mat": "158.mat",
    "12k_de/OR007@12/load_2/159.mat": "159.mat",
    "12k_de/OR007@12/load_3/160.mat": "160.mat",
    # 0.014 inch.
    "12k_de/IR014/load_0/169.mat": "169.mat",
    "12k_de/IR014/load_1/170.mat": "170.mat",
    "12k_de/IR014/load_2/171.mat": "171.mat",
    "12k_de/IR014/load_3/172.mat": "172.mat",
    "12k_de/B014/load_0/185.mat": "185.mat",
    "12k_de/B014/load_1/186.mat": "186.mat",
    "12k_de/B014/load_2/187.mat": "187.mat",
    "12k_de/B014/load_3/188.mat": "188.mat",
    "12k_de/OR014@6/load_0/197.mat": "197.mat",
    "12k_de/OR014@6/load_1/198.mat": "198.mat",
    "12k_de/OR014@6/load_2/199.mat": "199.mat",
    "12k_de/OR014@6/load_3/200.mat": "200.mat",
    # 0.021 inch.
    "12k_de/IR021/load_0/209.mat": "209.mat",
    "12k_de/IR021/load_1/210.mat": "210.mat",
    "12k_de/IR021/load_2/211.mat": "211.mat",
    "12k_de/IR021/load_3/212.mat": "212.mat",
    "12k_de/B021/load_0/222.mat": "222.mat",
    "12k_de/B021/load_1/223.mat": "223.mat",
    "12k_de/B021/load_2/224.mat": "224.mat",
    "12k_de/B021/load_3/225.mat": "225.mat",
    "12k_de/OR021@6/load_0/234.mat": "234.mat",
    "12k_de/OR021@6/load_1/235.mat": "235.mat",
    "12k_de/OR021@6/load_2/236.mat": "236.mat",
    "12k_de/OR021@6/load_3/237.mat": "237.mat",
    "12k_de/OR021@3/load_0/246.mat": "246.mat",
    "12k_de/OR021@3/load_1/247.mat": "247.mat",
    "12k_de/OR021@3/load_2/248.mat": "248.mat",
    "12k_de/OR021@3/load_3/249.mat": "249.mat",
    "12k_de/OR021@12/load_0/258.mat": "258.mat",
    "12k_de/OR021@12/load_1/259.mat": "259.mat",
    "12k_de/OR021@12/load_2/260.mat": "260.mat",
    "12k_de/OR021@12/load_3/261.mat": "261.mat",
    # 0.028 inch, inner and ball only.
    "12k_de/IR028/load_0/3001.mat": "3001.mat",
    "12k_de/IR028/load_1/3002.mat": "3002.mat",
    "12k_de/IR028/load_2/3003.mat": "3003.mat",
    "12k_de/IR028/load_3/3004.mat": "3004.mat",
    "12k_de/B028/load_0/3005.mat": "3005.mat",
    "12k_de/B028/load_1/3006.mat": "3006.mat",
    "12k_de/B028/load_2/3007.mat": "3007.mat",
    "12k_de/B028/load_3/3008.mat": "3008.mat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a minimal official CWRU subset.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/CWRU"))
    parser.add_argument("--subset", choices=["minimal", "full12k"], default="minimal")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = MINIMAL_FILES if args.subset == "minimal" else FULL_12K_DE_FILES
    downloaded = 0
    skipped = 0
    for rel_path, remote_name in files.items():
        target = args.out_dir / rel_path
        if target.exists() and target.stat().st_size > 0 and not args.overwrite:
            print(f"skip {target}")
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{CWRU_BASE}/{remote_name}"
        print(f"download {url} -> {target}")
        _download_with_retries(url, target, args.retries)
        downloaded += 1
    print(f"downloaded={downloaded} skipped={skipped} total={len(files)}")


def _download_with_retries(url: str, target: Path, retries: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if target.exists():
                target.unlink()
            urllib.request.urlretrieve(url, target)
            return
        except Exception as exc:
            last_error = exc
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if target.exists():
                target.unlink()
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"failed to download {url}") from last_error


if __name__ == "__main__":
    main()
