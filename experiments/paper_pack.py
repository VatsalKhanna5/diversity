import _bootstrap  # noqa: F401

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FIGURE_MAP = {
    "results/plots/comparison/exp01_baseline_compare.png": "paper/figures/Fig01_baseline_diversity.png",
    "results/plots/ber_curves/exp03_scirs_2x1.png": "paper/figures/Fig02_scirs_2x1_ber.png",
    "results/plots/comparison/exp04_scirs3x1_vs_mrc3x1.png": "paper/figures/Fig03_scirs3x1_vs_mrc3x1.png",
    "results/plots/comparison/exp05_correlated.png": "paper/figures/Fig04_correlated_channel_robustness.png",
    "results/plots/comparison/exp06_rotation_sweep.png": "paper/figures/Fig05_rotation_sweep.png",
    "results/plots/comparison/exp07_complexity_runtime.png": "paper/figures/Fig06_runtime_complexity.png",
    "results/plots/constellation/exp08_qpsk_rotation.png": "paper/figures/Fig07_qpsk_rotation_constellation.png",
}

TABLE_MAP = {
    "paper/tables/exp01_baseline_table.md": "paper/tables/Table01_baseline_snr_targets.md",
    "paper/tables/exp01_baseline_table.json": "paper/tables/Table01_baseline_snr_targets.json",
    "paper/tables/exp04_scirs3x1_gain.md": "paper/tables/Table02_scirs3x1_gain.md",
    "paper/tables/exp04_scirs3x1_gain.json": "paper/tables/Table02_scirs3x1_gain.json",
}


def _copy_map(path_map: dict[str, str], strict: bool, kind: str) -> tuple[list[dict], list[dict]]:
    copied = []
    missing = []

    for src_raw, dst_raw in path_map.items():
        src = Path(src_raw)
        dst = Path(dst_raw)

        if not src.exists():
            missing.append({"kind": kind, "source": str(src), "target": str(dst)})
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"kind": kind, "source": str(src), "target": str(dst)})

    if strict and missing:
        missing_list = "\n".join(f"- {x['source']}" for x in missing)
        raise FileNotFoundError(f"Missing required {kind} artifacts:\n{missing_list}")

    return copied, missing


def pack_artifacts(strict: bool = False) -> dict:
    copied_figures, missing_figures = _copy_map(FIGURE_MAP, strict=strict, kind="figure")
    copied_tables, missing_tables = _copy_map(TABLE_MAP, strict=strict, kind="table")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "copied": copied_figures + copied_tables,
        "missing": missing_figures + missing_tables,
        "strict": strict,
    }

    manifest = Path("paper/manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect plots/tables into paper-ready paths.")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected artifact is missing.")
    args = parser.parse_args()

    payload = pack_artifacts(strict=args.strict)
    print(f"Paper pack complete: {len(payload['copied'])} copied, {len(payload['missing'])} missing")


if __name__ == "__main__":
    main()
