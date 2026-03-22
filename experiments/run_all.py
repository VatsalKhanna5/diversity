import _bootstrap  # noqa: F401

import importlib
from pathlib import Path


EXPERIMENT_MODULES = [
    "experiments.exp_01_baseline_mrc",
    "experiments.exp_02_alamouti",
    "experiments.exp_03_scirs_2x1",
    "experiments.exp_04_scirs_3x1",
    "experiments.exp_05_correlated_channels",
    "experiments.exp_06_rotation_sweep",
    "experiments.exp_07_complexity_analysis",
    "experiments.exp_08_constellation_visualization",
]


if __name__ == "__main__":
    for mod_name in EXPERIMENT_MODULES:
        print(f"[RUN] {mod_name}")
        mod = importlib.import_module(mod_name)
        path = Path(mod.__file__)
        namespace = {"__name__": "__main__", "__file__": str(path)}
        exec(path.read_text(encoding="utf-8"), namespace)
