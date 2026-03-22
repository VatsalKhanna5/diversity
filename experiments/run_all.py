import _bootstrap  # noqa: F401

import argparse
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


def _run_module_as_main(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    path = Path(mod.__file__)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    exec(path.read_text(encoding="utf-8"), namespace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all SCIRS experiments.")
    parser.add_argument("--paper-pack", action="store_true", help="Copy generated artifacts into paper-ready paths.")
    parser.add_argument("--strict-pack", action="store_true", help="With --paper-pack, fail if artifacts are missing.")
    parser.add_argument("--final-plots", action="store_true", help="Generate polished final paper plots from result files.")
    args = parser.parse_args()

    for mod_name in EXPERIMENT_MODULES:
        print(f"[RUN] {mod_name}")
        _run_module_as_main(mod_name)

    if args.final_plots:
        from experiments.final_paper_plots import main as final_plots_main

        print("[RUN] experiments.final_paper_plots")
        final_plots_main()

    if args.paper_pack:
        from experiments.paper_pack import pack_artifacts

        result = pack_artifacts(strict=args.strict_pack)
        print(f"[PACK] copied={len(result['copied'])} missing={len(result['missing'])}")
