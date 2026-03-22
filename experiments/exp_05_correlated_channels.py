import _bootstrap  # noqa: F401

from copy import deepcopy

from src.pipeline.experiment_runner import run_experiment_config
from src.pipeline.reporting import comparison_plot
from src.utils.config_loader import load_config


if __name__ == "__main__":
    base_scirs = load_config("configs/scirs_3x1.yaml")
    base_mrc = load_config("configs/base_config.yaml")
    base_mrc["scheme"] = "mrc_3x1"

    rhos = [0.0, 0.5, 0.9]
    curves = []

    for rho in rhos:
        cfg = deepcopy(base_mrc)
        cfg["channel"] = {"type": "correlated", "correlation_rho": rho}
        out = run_experiment_config(cfg, output_tag=f"exp05_mrc3x1_rho_{rho}")
        out["scheme"] = f"MRC_3x1 rho={rho}"
        curves.append(out)

    for rho in rhos:
        cfg = deepcopy(base_scirs)
        cfg["channel"] = {"type": "correlated", "correlation_rho": rho}
        out = run_experiment_config(cfg, output_tag=f"exp05_scirs3x1_rho_{rho}")
        out["scheme"] = f"SCIRS_3x1 rho={rho}"
        curves.append(out)

    comparison_plot(curves, "Correlated Channel Stress Test (MRC vs SCIRS)", "results/plots/comparison/exp05_correlated.png")
