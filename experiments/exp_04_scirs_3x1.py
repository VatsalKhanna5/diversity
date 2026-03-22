import _bootstrap  # noqa: F401

from copy import deepcopy

from src.pipeline.experiment_runner import run_experiment, run_experiment_config
from src.pipeline.reporting import comparison_plot, make_summary_table
from src.utils.config_loader import load_config


if __name__ == "__main__":
    scirs_curve = run_experiment("configs/scirs_3x1.yaml", output_tag="exp04_scirs_3x1")

    mrc_cfg = deepcopy(load_config("configs/base_config.yaml"))
    mrc_cfg["scheme"] = "mrc_3x1"
    mrc_curve = run_experiment_config(mrc_cfg, output_tag="exp04_mrc_3x1")

    curves = [mrc_curve, scirs_curve]
    comparison_plot(curves, "SCIRS 3x1 vs MRC 3x1", "results/plots/comparison/exp04_scirs3x1_vs_mrc3x1.png")
    make_summary_table(curves, target_ber=1e-4, out_markdown="paper/tables/exp04_scirs3x1_gain.md")
