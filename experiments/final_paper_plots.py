import _bootstrap  # noqa: F401

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_style import apply_publication_style


def _load_csv_curve(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None

    snr = []
    ber = []
    ci = []
    scheme = None
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr.append(float(row["snr_db"]))
            ber.append(float(row["ber"]))
            ci.append(float(row.get("ber_ci95", 0.0)))
            scheme = row["scheme"]

    return {"snr": np.asarray(snr), "ber": np.asarray(ber), "ci": np.asarray(ci), "scheme": scheme or p.stem}


def _plot_curves(curves: list[dict], title: str, out_path: str, y_min: float = 1e-6) -> None:
    apply_publication_style()
    plt.figure(figsize=(7.5, 5.3))

    for c in curves:
        plt.semilogy(c["snr"], c["ber"], marker="o", lw=2, label=c["scheme"])
        lo = np.clip(c["ber"] - c["ci"], 1e-10, None)
        hi = np.clip(c["ber"] + c["ci"], 1e-10, None)
        plt.fill_between(c["snr"], lo, hi, alpha=0.12)

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.ylim(bottom=y_min)
    plt.legend(frameon=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    curves_main = []
    for path in [
        "results/processed/csv/exp09_siso.csv",
        "results/processed/csv/exp09_mrc_2x1.csv",
        "results/processed/csv/exp09_mrc_3x1.csv",
        "results/processed/csv/exp09_alamouti_2x1.csv",
        "results/processed/csv/exp09_scirs_2x1.csv",
        "results/processed/csv/exp09_scirs_3x1.csv",
    ]:
        c = _load_csv_curve(path)
        if c is not None:
            curves_main.append(c)

    if curves_main:
        _plot_curves(curves_main, "BER Comparison Across Baseline and SCIRS Schemes", "paper/figures/Fig10_main_ber_comparison.png")

    curves_gain = []
    for path in [
        "results/processed/csv/exp09_mrc_3x1.csv",
        "results/processed/csv/exp09_scirs_3x1.csv",
    ]:
        c = _load_csv_curve(path)
        if c is not None:
            curves_gain.append(c)
    if curves_gain:
        _plot_curves(curves_gain, "SCIRS 3x1 vs MRC 3x1", "paper/figures/Fig11_scirs3x1_gain.png", y_min=1e-7)

    curves_corr = []
    for path in [
        "results/processed/csv/exp05_scirs3x1_rho_0.0.csv",
        "results/processed/csv/exp05_scirs3x1_rho_0.5.csv",
        "results/processed/csv/exp05_scirs3x1_rho_0.9.csv",
    ]:
        c = _load_csv_curve(path)
        if c is not None:
            curves_corr.append(c)
    if curves_corr:
        _plot_curves(curves_corr, "SCIRS 3x1 Robustness Under Correlated Fading", "paper/figures/Fig12_correlation_robustness.png")

    rot = Path("results/raw/ber_logs/exp06_rotation_sweep.json")
    if rot.exists():
        data = json.loads(rot.read_text(encoding="utf-8"))
        apply_publication_style()
        plt.figure(figsize=(7.2, 4.8))
        theta = np.asarray(data["theta_deg"], dtype=float)
        ber = np.asarray(data["ber"], dtype=float)
        plt.plot(theta, ber, marker="o", lw=2)
        best = int(np.argmin(ber))
        plt.scatter([theta[best]], [ber[best]], color="red", zorder=4, label=f"Best theta={theta[best]:.1f} deg")
        plt.xlabel("Rotation angle (degrees)")
        plt.ylabel("BER at fixed SNR")
        plt.title("Rotation Angle Sweep")
        plt.legend()
        plt.savefig("paper/figures/Fig13_rotation_sweep.png")
        plt.close()

    comp = Path("results/raw/ber_logs/exp07_complexity_runtime.json")
    if comp.exists():
        data = json.loads(comp.read_text(encoding="utf-8"))
        apply_publication_style()
        x = np.arange(len(data["modulations"]))
        w = 0.32
        plt.figure(figsize=(7.0, 4.8))
        plt.bar(x - w / 2, data["ml_runtime_s"], width=w, label="ML")
        plt.bar(x + w / 2, data["sphere_runtime_s"], width=w, label="Sphere")
        plt.xticks(x, data["modulations"])
        plt.ylabel("Runtime (seconds)")
        plt.title("Detection Complexity")
        plt.legend()
        plt.savefig("paper/figures/Fig14_complexity_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()
