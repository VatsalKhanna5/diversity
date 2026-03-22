import _bootstrap  # noqa: F401

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.experiment_runner import run_experiment_config
from src.utils.config_loader import load_config
from src.utils.logger import save_json


if __name__ == "__main__":
    base = load_config("configs/scirs_2x1.yaml")
    snr_probe = 10
    thetas = np.arange(0, 46, 3)
    ber = []

    for theta in thetas:
        cfg = deepcopy(base)
        cfg["snr_db"] = {"start": snr_probe, "stop": snr_probe, "step": 1}
        cfg["rotation"] = {"theta_deg": float(theta)}
        result = run_experiment_config(cfg, output_tag=f"exp06_theta_{theta}")
        ber.append(result["ber"][0])

    best_idx = int(np.argmin(ber))
    summary = {
        "snr_db": snr_probe,
        "theta_deg": thetas.tolist(),
        "ber": [float(x) for x in ber],
        "best_theta_deg": float(thetas[best_idx]),
        "best_ber": float(ber[best_idx]),
    }
    save_json("results/raw/ber_logs/exp06_rotation_sweep.json", summary)

    plt.figure(figsize=(7, 5))
    plt.plot(thetas, ber, marker="o")
    plt.xlabel("Rotation angle (deg)")
    plt.ylabel(f"BER at {snr_probe} dB")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/comparison/exp06_rotation_sweep.png", dpi=240)
    plt.close()
