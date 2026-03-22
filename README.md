# SCIRS Research Framework

A reproducible, config-driven mini research framework for diversity/coding-gain experiments:
SISO, MRC, Alamouti, SCIRS-2x1, and SCIRS-3x1.

## What is included

- Theory/channel/modulation/scheme separation under `src/`
- Reproducible runs with logged configs + per-seed metrics
- Early stopping (`max_errors`) with target symbol budget (`n_symbols`)
- Raw JSON + CSV + NPZ outputs and publication-style plots
- Full experiment suite + one-command orchestration

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run one experiment

```bash
python experiments/exp_04_scirs_3x1.py
```

## Run all experiments

```bash
python experiments/run_all.py
# or run and automatically collect paper assets
python experiments/run_all.py --paper-pack
```

## Experiment map

- `exp_01_baseline_mrc.py`: SISO, MRC(2x1/3x1), Alamouti(2x1)
- `exp_02_alamouti.py`: focused Alamouti baseline
- `exp_03_scirs_2x1.py`: SCIRS 2x1 curve
- `exp_04_scirs_3x1.py`: flagship SCIRS 3x1 vs MRC 3x1 + table
- `exp_05_correlated_channels.py`: rho in {0.0, 0.5, 0.9}
- `exp_06_rotation_sweep.py`: theta sweep in [0, 45] deg
- `exp_07_complexity_analysis.py`: ML vs sphere runtime
- `exp_08_constellation_visualization.py`: original/rotated QPSK plots

## Output layout

- Raw logs: `results/raw/ber_logs`
- Processed CSV: `results/processed/csv`
- Processed numpy: `results/processed/numpy`
- BER plots: `results/plots/ber_curves`
- Comparison plots: `results/plots/comparison`
- Constellations: `results/plots/constellation`
- Paper tables: `paper/tables`

## Reproducibility knobs

See `configs/base_config.yaml`:

- `seeds`: multi-seed averaging
- `n_symbols`: symbols per SNR point
- `max_errors`: early stopping threshold
- `parallel.enabled/workers`: optional multiprocessing

## Paper Pack

Collect publication assets with standardized names:

```bash
python experiments/paper_pack.py
# strict mode: fail if any expected file is missing
python experiments/paper_pack.py --strict
```

Manifest is written to `paper/manifest.json`.
