import _bootstrap  # noqa: F401

from src.pipeline.experiment_runner import run_experiment


if __name__ == "__main__":
    run_experiment("configs/scirs_2x1.yaml", output_tag="exp03_scirs_2x1")
