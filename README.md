# Suppression-Response-Test
Measuring Protocol-Induced Shortcut Generalization in Bearing Fault Diagnosis: A Suppression-Response Test
# Suppression-Response Test for Bearing Fault Diagnosis

Code and results for the paper:

> **Measuring Protocol-Induced Shortcut Generalization in Bearing Fault Diagnosis: A Suppression-Response Test**
> Lei Junyu

This repository implements a measurement framework for evaluating whether high diagnostic accuracy in bearing fault diagnosis reflects genuine fault-discrimination ability or is inflated by evaluation-protocol shortcuts (temporal proximity, record identity, operating-condition leakage, or file-channel artifacts).

## What This Repository Contains

- **`src/shortcutfd/`** — Core library: data schema, windowing, deterministic protocol splits (P0–P4), and metadata I/O.
- **`scripts/`** — 79 experiment scripts covering the full pipeline: data download, metadata construction, model training, SRT execution, probe-capacity sensitivity, power analysis, and result summarization.
- **`results/`** — 126 CSV result files and 15 JSON validation files from all completed experiments.
- **`data/splits/`** — Deterministic split index files (window ID → train/test assignment) for every protocol and fold.
- **`data/metadata/`** — Window and record index files.
- **`reports/`** — Experiment protocols, acceptance criteria, and validation reports.
- **`paper/`** — LaTeX source of the manuscript.

## The Suppression-Response Test (SRT)

The SRT is a three-step measurement protocol for grouped vibration data:

1. **Detect** — Train a probe (linear or MLP) on frozen features to measure whether a candidate channel (e.g., operating condition, file identity) is decodable. Report the predictability gap Δ_k between base (λ=0) and suppressed (λ>0) checkpoints.

2. **Suppress** — Apply a measurement intervention (adversarial training, moment matching, or representation canonicalization) designed to reduce channel decodability.

3. **Respond** — Measure the paired accuracy change δ_k on the held-out strict-protocol test set.

A **manipulation check** (η_k > 0 with confidence interval excluding zero) is required before any causal attribution. The framework further requires **probe-capacity sensitivity**: both linear and MLP probes must agree on the manipulation direction for a result to be labeled capacity-robust.

### Response Classification

| Label | Condition | Interpretation |
|---|---|---|
| **shortcut** | δ_k > 0, η_k > 0 (both probes) | Suppression removed a shortcut; accuracy dropped because the shortcut was useful |
| **entanglement** | δ_k < 0, η_k > 0 (both probes) | Suppression removed task-relevant information entangled with the channel |
| **neutral** | \|δ_k\| < ε_A, η_k > 0 | Channel was removable without accuracy cost |
| **inconclusive** | η_k CI crosses zero | Manipulation not verified; no causal attribution possible |
| **canonicalization** | δ_can > 0, η_can > 0 | Label-free representation change improved accuracy while reducing nuisance |

### Protocol Matrix

| Protocol | Splitting Rule | Leakage Risk |
|---|---|---|
| P0 | Random window | Temporal proximity, record identity |
| P1 | Record-wise disjoint | Record identity |
| P2 | Condition-disjoint | Operating-condition confound |
| P3 | Bearing-disjoint | Bearing-individual confound |
| P4 | P2 ∩ P3 (joint) | Both condition and bearing |

## Dependencies

```
Python >= 3.9
numpy
scipy
scikit-learn
torch >= 1.12
```

Install with:

```bash
pip install numpy scipy scikit-learn torch
```

No GPU is required for reproduction; all experiments in the paper were run on CPU.

## Project Structure

```
.
├── src/shortcutfd/          # Core library
│   ├── schema.py            # Record dataclass and CSV I/O
│   ├── windows.py           # Window dataclass and indexing
│   ├── splits.py            # Split dataclass and protocol rules
│   ├── features.py          # Time/frequency feature extraction
│   ├── paderborn.py         # Paderborn-specific metadata parsing
│   ├── metadata_builders.py # Record/window index construction
│   └── io.py                # Signal loading utilities
│
├── scripts/                 # Experiment scripts (79 files)
│   ├── download_*.py        # Dataset download scripts
│   ├── build_metadata.py    # Build record index from raw data
│   ├── build_windows.py     # Build window index
│   ├── build_splits.py      # Generate P0/P1/P2 splits
│   ├── train_wdcnn.py       # WDCNN training (also defines ResNet1D)
│   ├── run_paderborn_matched_srt.py  # Main SRT runner
│   ├── run_paderborn_c00_power_extension.py  # 15-pair c00 extension
│   ├── probe_capacity_sensitivity.py  # Linear vs MLP probe comparison
│   ├── synthetic_srt_calibrated.py    # Synthetic marker positive control
│   ├── run_suppression_response_hust.py  # HUST file-channel SRT
│   ├── run_p3_*.py          # Paderborn P3 physical-bearing experiments
│   ├── summarize_*.py       # Result summarization scripts
│   └── analyze_*.py         # Sensitivity and power analyses
│
├── results/                 # Completed experiment results
│   ├── paderborn_matched_srt_*.csv         # WDCNN SRT (5 seeds × 4 folds)
│   ├── paderborn_matched_srt_resnet1d_*.csv # ResNet1D SRT (5 seeds × 4 folds)
│   ├── paderborn_matched_srt_power.csv     # c00 15-pair power analysis
│   ├── synthetic_srt_marker_summary.csv    # Synthetic positive control
│   ├── probe_capacity_sensitivity_*.csv    # Probe capacity comparison
│   ├── srt_margin_sensitivity.csv          # ε_A threshold sensitivity
│   ├── supplement_hust_delta_r_summary.csv # HUST file-channel SRT
│   ├── paderborn_inner_confirmatory_v1_*.csv # Inner-race confirmatory
│   ├── paderborn_balanced18_*.csv          # P3 physical-bearing results
│   └── *.json                              # Validation status files
│
├── data/
│   ├── splits/              # Deterministic split index files
│   └── metadata/            # Window and record index files
│
├── reports/                 # Experiment protocols and validation reports
└── paper/                   # LaTeX manuscript source
```

## Reproduction Guide

### Step 1: Download Datasets

Raw data must be downloaded separately. The scripts below download the subsets used in the paper:

```bash
# CWRU (small, ~50 MB)
python scripts/download_cwru_minimal.py --out-dir data/raw/CWRU

# HUST Mendeley v3 (~665 MB compressed)
python scripts/download_hust_mendeley.py --out-dir data/raw/HUST

# Paderborn pilot subset (requires --insecure for legacy certificate)
python scripts/download_paderborn_subset.py --out-dir data/raw/Paderborn --subset pilot --insecure
```

**Note**: The full Paderborn balanced18 subset (18 bearings) requires manual download from the Paderborn University website due to size and licensing.

### Step 2: Build Metadata and Splits

```bash
# Build record index
python scripts/build_metadata.py \
  --cwru-root data/raw/CWRU \
  --hust-root data/raw/HUST \
  --paderborn-root data/raw/Paderborn/extracted \
  --out data/metadata/records.csv

# Build window index (size=2048, stride=1024)
python scripts/build_windows.py \
  --records data/metadata/records.csv \
  --out data/metadata/windows.csv \
  --window-size 2048 --stride 1024 --max-windows-per-record 500

# Generate P0/P1/P2 splits (seed=13)
python scripts/build_splits.py \
  --windows data/metadata/windows.csv \
  --out-dir data/splits --seed 13
```

### Step 3: Run SRT Experiments

#### Paderborn Condition-Channel SRT (WDCNN, 5 seeds × 4 folds)

```bash
python scripts/run_paderborn_matched_srt.py \
  --backbone wdcnn \
  --seeds 7 13 21 42 101 \
  --folds leave_condition_00 leave_condition_01 leave_condition_02 leave_condition_03 \
  --methods deepcoral \
  --epochs 12 \
  --tag paderborn_matched_srt
```

#### Paderborn Condition-Channel SRT (ResNet1D, 5 seeds × 4 folds)

```bash
python scripts/run_paderborn_matched_srt.py \
  --backbone resnet1d \
  --seeds 7 13 21 42 101 \
  --folds leave_condition_00 leave_condition_01 leave_condition_02 leave_condition_03 \
  --methods deepcoral \
  --epochs 12 \
  --tag paderborn_matched_srt_resnet1d
```

#### c00 Power Extension (15 pairs)

```bash
python scripts/run_paderborn_c00_power_extension.py \
  --n-pairs 15 \
  --backbone wdcnn \
  --tag paderborn_matched_srt
```

#### HUST File-Channel SRT

```bash
python scripts/run_suppression_response_hust.py \
  --lambdas 0.0 0.1 0.3 1.0 \
  --seeds 13 21 42 \
  --protocol P1_record_wise
```

#### Synthetic Marker Positive Control

```bash
python scripts/synthetic_srt_calibrated.py \
  --lambdas 0.0 0.1 0.3 1.0 \
  --seeds 7 13 21 42 101 \
  --epochs 16
```

### Step 4: Analyze Results

```bash
# Probe-capacity sensitivity (linear vs MLP)
python scripts/probe_capacity_sensitivity.py \
  --checkpoint-dir results/checkpoints/paderborn_matched_srt \
  --out results/probe_capacity_sensitivity_summary.csv

# ε_A threshold sensitivity
python scripts/analyze_srt_margin_sensitivity.py \
  --margins 0.002 0.003 0.005 0.01 \
  --out results/srt_margin_sensitivity.csv

# Power analysis for c00
python scripts/summarize_srt_power.py \
  --tag paderborn_matched_srt \
  --fold leave_condition_00 \
  --out results/paderborn_matched_srt_power.csv
```

## Key Results

All results are stored in `results/` as CSV files. The most important files are:

| File | Content |
|---|---|
| `paderborn_matched_srt_summary.csv` | WDCNN SRT: 4 folds × 5 seeds, δ and η for each fold |
| `paderborn_matched_srt_resnet1d_summary.csv` | ResNet1D SRT: 4 folds × 5 seeds, cross-architecture comparison |
| `paderborn_matched_srt_power.csv` | c00 15-pair extension: power=0.828, p=0.007 |
| `synthetic_srt_marker_summary.csv` | Synthetic positive control: δ=+0.122, η=+0.171, shortcut response |
| `probe_capacity_sensitivity_summary.csv` | Linear vs MLP probe comparison for all checkpoints |
| `srt_margin_sensitivity.csv` | ε_A threshold sensitivity: 4 margins × 7 labels |
| `supplement_hust_delta_r_summary.csv` | HUST file-channel SRT: entanglement at λ=0.3 |
| `paderborn_inner_confirmatory_v1_per_bearing.csv` | Inner-race confirmatory: 9 bearings, null result |
| `paderborn_backbone_srt_comparison.csv` | WDCNN vs ResNet1D cross-architecture comparison |

### Summary of Main Findings

1. **Protocol gap**: WDCNN drops 23.28 pp on the hardest Paderborn P2 fold (c00).
2. **Synthetic positive control**: δ=+0.122, η=+0.171 — complete shortcut response with verified manipulation.
3. **HUST file-channel**: Capacity-robust entanglement response at λ=0.3 (δ=-0.090, both linear and MLP probes verify suppression).
4. **Paderborn c00 (15-pair)**: δ=-0.0196, p=0.007, power=0.828; linear suppression verified but MLP not — capacity-sensitive inconclusive, demonstrating the probe-capacity rule in action.
5. **Cross-architecture**: ResNet1D replication yields 8/8 capacity-aware inconclusive labels, with MLP probes revealing architecture-dependent encoding increase under DeepCORAL intervention.
6. **P3 canonicalization**: Low-frequency envelope representation improves unseen-bearing accuracy by 24.86 pp and macro-F1 by 26.85 pp.
7. **BPFO localization**: Strong outer-race frequency dependence confirmed; BPFI mechanism not supported as universal operator (mean -0.24 pp, p=0.566).

## Verification

Each major experiment has a corresponding `*_validation.json` file that checks internal consistency:

```bash
# Check WDCNN SRT validation
cat results/paderborn_matched_srt_resnet1d_validation.json

# Check synthetic marker validation
cat results/synthetic_srt_marker_validation.json

# Check probe capacity validation
cat results/probe_capacity_sensitivity_validation.json
```

All validation files report `"overall_status": "pass"`.

## Reproducibility Notes

- **Deterministic splits**: All splits are generated from fixed seeds and stored as CSV index files. Re-running `build_splits.py` with the same seed produces identical splits.
- **Per-window normalization**: Each window is normalized using its own mean and standard deviation at load time.
- **Checkpoint reproducibility**: Training scripts use `_seed_everything(seed)` to set Python, NumPy, and PyTorch random seeds. Results may vary slightly across PyTorch versions due to CUDA non-determinism, but CPU results are fully reproducible.
- **Probe construction**: Linear probes use sklearn LogisticRegression; MLP probes use a 2-layer ReLU network with hidden widths [64, 32], trained with Adam and early stopping. Three MLP initialization seeds (13, 21, 42) are averaged per checkpoint.

## Citation

```bibtex
@article{lei2026srt,
  title={Measuring Protocol-Induced Shortcut Generalization in Bearing Fault Diagnosis: A Suppression-Response Test},
  author={Lei, Junyu},
  year={2026},
  journal={Measurement},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Lei Junyu — China University of Geosciences, Department of Computer science
Email: [Leijunyu1216@gmail.com,2718889330@qq.com]
