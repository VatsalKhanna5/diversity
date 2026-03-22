import _bootstrap  # noqa: F401

from copy import deepcopy

from src.pipeline.experiment_runner import run_experiment_config
from src.pipeline.reporting import comparison_plot
from src.utils.config_loader import load_config
from src.utils.logger import save_json


if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    schemes = ["siso", "mrc_2x1", "mrc_3x1", "alamouti_2x1", "scirs_2x1", "scirs_3x1"]
    curves = []

    for scheme in schemes:
        run_cfg = deepcopy(cfg)
        run_cfg["scheme"] = scheme
        if scheme == "scirs_2x1":
            run_cfg["rotation"] = {"theta_deg": 31.7175}
        curves.append(run_experiment_config(run_cfg, output_tag=f"exp09_{scheme}"))

    comparison_plot(curves, "Main BER Comparison", "results/plots/comparison/exp09_main_comparison.png")
    save_json("results/raw/ber_logs/exp09_main_comparison.json", {"curves": curves})
