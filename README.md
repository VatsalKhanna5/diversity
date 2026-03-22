# SCIRS Research Framework

A reproducible simulation and reporting framework for evaluating Spatial-Coordinate Interleaved Rotated Signaling (SCIRS) against classical diversity baselines in low-dimensional MIMO/MISO settings.

## 1) Scope and Purpose

This repository is designed for publication-oriented experimentation. It emphasizes:

- strict experiment reproducibility (config + seed logging)
- modularity across modulation, channel, scheme, and receiver layers
- statistically meaningful BER/SER evaluation
- artifact traceability from raw logs to paper figures/tables

Core compared schemes:

- SISO
- MRC (2x1, 3x1)
- Alamouti (2x1)
- SCIRS (2x1, 3x1)

## 2) Repository Layout

- `configs/`: YAML configurations for default and experiment-specific runs
- `src/modulation/`: QPSK / 16QAM mappers and constellation definitions
- `src/channel/`: Rayleigh and correlated channel models, AWGN
- `src/schemes/`: MRC, Alamouti, SCIRS encoding/decoding blocks
- `src/pipeline/`: simulation runner, experiment orchestration, reporting helpers
- `experiments/`: standalone experiment entry scripts
- `results/`: generated raw logs, processed numeric outputs, and plots
- `paper/`: publication-ready figures/tables and packaging manifest

## 3) Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4) Reproducibility Controls

Primary controls are in `configs/base_config.yaml`:

- `seeds`: list of random seeds for multi-run averaging
- `n_symbols`: target symbols per SNR point
- `max_errors`: early-stop threshold (minimum useful errors)
- `snr_db`: SNR range grid
- `parallel.enabled`, `parallel.workers`: multiprocessing switch

Each experiment stores:

- raw per-run logs in `results/raw/ber_logs/*.json`
- aggregate per-SNR CSV in `results/processed/csv/*.csv`
- NumPy arrays in `results/processed/numpy/*.npz`
- generated BER plots in `results/plots/ber_curves/*.png`

## 5) Simulation Methodology

### 5.1 Statistical policy

- BER/SER estimated per SNR point
- mean aggregation across seeds
- 95% confidence interval reported for BER (`ber_ci95`)
- early stopping active once sufficient error events are collected

### 5.2 Channel policy

- flat Rayleigh fading (default)
- correlated channel stress test via Toeplitz model:
  `R_ij = rho^|i-j|`

### 5.3 Scheme notes

- MRC and Alamouti are implemented as classical baselines
- SCIRS variants use coordinate interleaving + rotation + ML block detection

## 6) Running Experiments

### 6.1 Individual experiments

```bash
python experiments/exp_01_baseline_mrc.py
python experiments/exp_03_scirs_2x1.py
python experiments/exp_04_scirs_3x1.py
```

### 6.2 Full batch run

```bash
python experiments/run_all.py
```

Optional flags:

```bash
python experiments/run_all.py --final-plots
python experiments/run_all.py --paper-pack
python experiments/run_all.py --final-plots --paper-pack --strict-pack
```

## 7) Experiment Inventory

- `exp_01_baseline_mrc.py`: baseline BER comparison (SISO/MRC/Alamouti)
- `exp_02_alamouti.py`: focused Alamouti run
- `exp_03_scirs_2x1.py`: SCIRS 2x1 BER/SER evaluation
- `exp_04_scirs_3x1.py`: flagship SCIRS 3x1 vs MRC 3x1 comparison
- `exp_05_correlated_channels.py`: robustness under `rho={0.0,0.5,0.9}`
- `exp_06_rotation_sweep.py`: angle sweep and best-theta identification
- `exp_07_complexity_analysis.py`: ML vs sphere runtime benchmark
- `exp_08_constellation_visualization.py`: constellation geometry figure
- `final_paper_plots.py`: polished, paper-ready composite plots

## 8) Final Paper Artifacts

### 8.1 Generate polished figures

```bash
python experiments/final_paper_plots.py
```

Expected outputs (if source logs exist):

- `paper/figures/Fig10_main_ber_comparison.png`
- `paper/figures/Fig11_scirs3x1_gain.png`
- `paper/figures/Fig12_correlation_robustness.png`
- `paper/figures/Fig13_rotation_sweep.png`
- `paper/figures/Fig14_complexity_comparison.png`

### 8.2 Package and verify paper assets

```bash
python experiments/paper_pack.py
python experiments/paper_pack.py --strict
```

- packaging manifest: `paper/manifest.json`

## 9) Quality and Interpretation Guidance

For research-grade confidence:

- increase `n_symbols` beyond baseline defaults for high-SNR tails
- ensure confidence intervals are narrow enough at target BER points
- avoid over-interpreting single-seed trends
- validate monotonic BER behavior before reporting coding-gain claims

## 10) Notes for Extension

Typical extension points:

- add new modulation in `src/modulation/`
- add new channel model in `src/channel/`
- add new scheme with compatible transmit/receive interface in `src/schemes/`
- add custom report logic in `src/pipeline/reporting.py`

## 11) Recommended Reporting Checklist

Before manuscript export:

1. run full experiment suite
2. regenerate final paper plots
3. verify table/figure consistency with raw logs
4. run strict paper pack to detect missing assets
5. archive exact config files used for each reported figure
