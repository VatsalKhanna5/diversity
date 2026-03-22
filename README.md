# SCIRS Research Framework

This repository implements a research-grade, reproducible Monte Carlo simulator for diversity systems over flat Rayleigh fading, including a mathematically correct implementation of SCIRS (Spatial-Coordinate Interleaved Rotated Signaling) with **joint ML detection**.

## 1. Research Objective

Evaluate BER/SER performance of:

- SISO
- MRC (2x1, 3x1)
- Alamouti (2x1)
- SCIRS (2x1, 3x1)

under:

- i.i.d. Rayleigh fading
- correlated Rayleigh fading (`rho = 0, 0.5, 0.9`)

for SNR range `0..30 dB`.

## 2. Mathematical Model

For an `Lx1` MISO channel:

- channel vector: `H = [h1, ..., hL]`
- each `hi ~ CN(0,1)`
- noise: `n ~ CN(0, N0)`
- received signal: `y = Hx + n`

Perfect CSI is assumed at the receiver.

## 3. Critical SCIRS Rule

SCIRS is **jointly encoded and jointly decoded**. It is not repetition coding.

- Do not decode symbol coordinates independently.
- Do not decode per symbol after rotation.
- Perform ML detection over the **full transmitted symbol vector**.

If implemented incorrectly, BER can saturate near random-guess levels.

## 4. Modulation

QPSK Gray mapping is implemented exactly as:

- `(0,0) -> (1 + j)/sqrt(2)`
- `(0,1) -> (-1 + j)/sqrt(2)`
- `(1,1) -> (-1 - j)/sqrt(2)`
- `(1,0) -> (1 - j)/sqrt(2)`

Average symbol energy is normalized to 1.

## 5. Scheme Implementations

### 5.1 SISO

- model: `y = h s + n`
- detection: ML over QPSK constellation

### 5.2 MRC (`Lx1`)

- same symbol transmitted from all antennas
- receive model: `y = sum_i h_i * s + n`
- equivalent channel: `h_eff = sum_i h_i / sqrt(L)`
- equalize then ML slice

### 5.3 Alamouti (2x1)

- standard orthogonal STBC:
  - time 1: `[s1, s2]`
  - time 2: `[-conj(s2), conj(s1)]`
- linear combiner with known closed-form decoder

### 5.4 SCIRS 2x1

- symbol vector: `s = [s1, s2]^T`
- rotated vector: `x = G s`, where
  - `theta = 0.5 * arctan(2)`
  - `G = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`
- single-slot transmission across two antennas
- joint ML over all `4^2 = 16` QPSK pairs

### 5.5 SCIRS 3x1

- symbol vector: `s = [s1, s2, s3]^T`
- rotated using deterministic orthonormal `3x3` matrix
- single-slot transmission across three antennas
- joint ML over all `4^3 = 64` QPSK triples

## 6. Statistics and Reliability

Per SNR point:

- simulate up to `n_symbols` (default `1e5`)
- early-stop only after collecting at least `max_errors` (default `200`) bit errors
- aggregate over multiple seeds
- report BER mean, SER mean, BER standard deviation, and BER 95% CI

## 7. Correlated Channel Model

Correlation matrix:

- `R_ij = rho^|i-j|`

Channel generation uses Cholesky factorization of `R`.

## 8. Sanity Checks

Built-in checks verify:

- high-SNR BER is low for all schemes
- BER does not collapse to random-guess region

Run:

```bash
python experiments/exp_10_sanity_checks.py
```

## 9. Repository Structure

- `configs/` configuration files
- `src/modulation/` QPSK / 16QAM mapping
- `src/channel/` Rayleigh + correlated channel models
- `src/schemes/` Alamouti + SCIRS kernels
- `src/pipeline/` simulation + experiment execution
- `experiments/` reproducible experiment entrypoints
- `results/` raw/processed output artifacts
- `paper/` publication-ready figures and tables

## 10. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 11. Run Experiments

### Full suite

```bash
python experiments/run_all.py
```

### Full suite + final figures + packaging

```bash
python experiments/run_all.py --final-plots --paper-pack --strict-pack
```

## 12. Experiment Scripts

- `exp_01_baseline_mrc.py`: baseline SISO/MRC/Alamouti
- `exp_02_alamouti.py`: Alamouti-only reference
- `exp_03_scirs_2x1.py`: SCIRS 2x1
- `exp_04_scirs_3x1.py`: SCIRS 3x1 vs MRC 3x1
- `exp_05_correlated_channels.py`: correlated stress test (MRC + SCIRS)
- `exp_06_rotation_sweep.py`: BER vs rotation angle
- `exp_07_complexity_analysis.py`: detector runtime comparison
- `exp_08_constellation_visualization.py`: constellation geometry
- `exp_09_main_comparison.py`: all core schemes on one BER figure
- `exp_10_sanity_checks.py`: implementation sanity verification
- `final_paper_plots.py`: polished paper figures from stored CSV/JSON

## 13. Output Artifacts

Generated artifacts include:

- raw logs: `results/raw/ber_logs/*.json`
- aggregate curves: `results/processed/csv/*.csv`
- numpy bundles: `results/processed/numpy/*.npz`
- BER curves: `results/plots/ber_curves/*.png`
- comparison figures: `results/plots/comparison/*.png`
- paper figures: `paper/figures/*.png`
- paper tables: `paper/tables/*.md`, `paper/tables/*.json`
- pack manifest: `paper/manifest.json`

## 14. Reproducibility Checklist

Before reporting final numbers:

1. lock config files used for each figure
2. run sanity checks
3. regenerate all experiments with fixed seeds
4. regenerate final paper plots
5. run strict paper packaging
6. archive raw JSON logs with manuscript
