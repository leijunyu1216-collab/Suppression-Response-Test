"""Compute bearing-specific BPFO/BPFI envelope-harmonic profiles."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortcutfd.io import load_signal  # noqa: E402
from shortcutfd.paderborn import read_window_rpm  # noqa: E402
from shortcutfd.windows import read_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--rpm-map", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-per-bearing-condition", type=int, default=100)
    parser.add_argument("--harmonics", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--target-half-width-order", type=float, default=0.35)
    parser.add_argument("--background-half-width-order", type=float, default=1.5)
    parser.add_argument("--sampling-rate", type=float, default=64000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.background_half_width_order <= args.target_half_width_order:
        raise SystemExit("background width must exceed target width")
    windows = read_windows(args.windows)
    rpm = read_window_rpm(args.rpm_map)
    geometry = pd.read_csv(args.geometry).set_index("bearing_id")
    missing = sorted({window.bearing_id for window in windows} - set(geometry.index))
    if missing:
        raise SystemExit(f"geometry missing bearings: {missing}")

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for window in windows:
        if window.window_id in rpm:
            groups[(window.bearing_id, window.condition_id)].append(window)
    selected = []
    for key, values in sorted(groups.items()):
        selected.extend(_even_sample(values, args.windows_per_bearing_condition))

    by_path: dict[str, list] = defaultdict(list)
    for window in selected:
        by_path[window.path].append(window)
    observations: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)
    fft_resolution = args.sampling_rate / (selected[0].end - selected[0].start)
    for path_index, (path, path_windows) in enumerate(sorted(by_path.items()), start=1):
        signal = load_signal(Path(path), preferred_channel=path_windows[0].channel)
        batch = np.stack([_slice(signal, window.start, window.end) for window in path_windows])
        envelope = np.abs(hilbert(batch, axis=1))
        power = np.abs(np.fft.rfft(envelope, axis=1)) ** 2
        frequency = np.fft.rfftfreq(batch.shape[1], d=1.0 / args.sampling_rate)
        for index, window in enumerate(path_windows):
            shaft_hz = float(rpm[window.window_id]) / 60.0
            for mechanism in ("bpfo", "bpfi"):
                order = float(geometry.loc[window.bearing_id, f"{mechanism}_order"])
                values = [
                    _harmonic_snr(
                        power[index], frequency, shaft_hz, order * harmonic,
                        args.target_half_width_order, args.background_half_width_order,
                        fft_resolution,
                    )
                    for harmonic in args.harmonics
                ]
                observations[(window.bearing_id, window.condition_id, mechanism)].append(values)
        if path_index % 50 == 0:
            print(f"processed {path_index}/{len(by_path)} records", flush=True)

    rows = []
    label_by_bearing = {window.bearing_id: window.fault_label for window in windows}
    for (bearing, condition, mechanism), values in sorted(observations.items()):
        array = np.asarray(values, dtype=float)
        row = {
            "bearing_id": bearing,
            "fault_label": label_by_bearing[bearing],
            "condition_id": condition,
            "mechanism": mechanism,
            "n_windows": len(array),
            "manufacturer": geometry.loc[bearing, "manufacturer"],
            "characteristic_order": geometry.loc[bearing, f"{mechanism}_order"],
            "median_harmonic_snr_db": float(np.median(array)),
        }
        for position, harmonic in enumerate(args.harmonics):
            row[f"h{harmonic}_snr_db"] = float(np.median(array[:, position]))
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


def _even_sample(values: list, cap: int) -> list:
    values = sorted(values, key=lambda item: (item.record_id, item.window_order))
    if len(values) <= cap:
        return values
    indexes = np.linspace(0, len(values) - 1, cap).round().astype(int)
    return [values[index] for index in indexes]


def _slice(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    size = end - start
    value = signal[start:end].astype(np.float32, copy=False)
    if len(value) == size:
        return value
    output = np.zeros(size, dtype=np.float32)
    output[: len(value)] = value
    return output


def _harmonic_snr(
    power: np.ndarray,
    frequency: np.ndarray,
    shaft_hz: float,
    center_order: float,
    target_width_order: float,
    background_width_order: float,
    minimum_width_hz: float,
) -> float:
    center = center_order * shaft_hz
    target_width = max(target_width_order * shaft_hz, minimum_width_hz)
    background_width = max(background_width_order * shaft_hz, 2.5 * minimum_width_hz)
    target = np.abs(frequency - center) <= target_width
    background = (np.abs(frequency - center) <= background_width) & ~target
    target_energy = float(np.mean(power[target])) if target.any() else 0.0
    background_energy = float(np.mean(power[background])) if background.any() else 0.0
    return 10.0 * np.log10((target_energy + 1e-12) / (background_energy + 1e-12))


if __name__ == "__main__":
    main()
