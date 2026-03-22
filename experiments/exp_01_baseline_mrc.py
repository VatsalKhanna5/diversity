import _bootstrap  # noqa: F401

from copy import deepcopy

from src.pipeline.experiment_runner import run_experiment_config
from src.pipeline.reporting import comparison_plot, make_summary_table
from src.utils.config_loader import load_config
from src.utils.logger import save_json


if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    schemes = ["siso", "mrc_2x1", "mrc_3x1", "alamouti_2x1"]

    curves = []
    for scheme in schemes:
        run_cfg = deepcopy(cfg)
        run_cfg["scheme"] = scheme
        curves.append(run_experiment_config(run_cfg, output_tag=f"baseline_{scheme}"))

    comparison_plot(curves, "Baseline Diversity Schemes", "results/plots/comparison/exp01_baseline_compare.png")
    make_summary_table(curves, target_ber=1e-3, out_markdown="paper/tables/exp01_baseline_table.md")
    save_json("results/raw/ber_logs/exp01_baseline_bundle.json", {"curves": curves})
