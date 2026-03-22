import _bootstrap  # noqa: F401

from copy import deepcopy

from src.pipeline.experiment_runner import run_experiment_config
from src.utils.config_loader import load_config


if __name__ == "__main__":
    cfg = deepcopy(load_config("configs/base_config.yaml"))
    cfg["scheme"] = "alamouti_2x1"
    run_experiment_config(cfg, output_tag="exp02_alamouti")
