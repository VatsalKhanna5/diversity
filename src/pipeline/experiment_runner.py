from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.evaluator import make_snr_grid, save_csv
from src.pipeline.simulate import run_single_snr
from src.utils.config_loader import load_config
from src.utils.logger import build_logger, save_json
from src.utils.plot_style import apply_publication_style


def _run_job(cfg: dict, snr: float, seed: int) -> dict:
    return run_single_snr(cfg, snr, seed)


def _plot_curve(payload: dict, out_path: Path) -> None:
    apply_publication_style()

    snr = np.asarray(payload["snr_db"], dtype=float)
    ber = np.asarray(payload["ber"], dtype=float)
    ci95 = np.asarray(payload["ber_ci95"], dtype=float)

    plt.figure()
    plt.semilogy(snr, ber, marker="o", lw=2, label=payload["scheme"])
    lower = np.clip(ber - ci95, 1e-8, None)
    upper = np.clip(ber + ci95, 1e-8, None)
    plt.fill_between(snr, lower, upper, alpha=0.18, label="95% CI")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title(f"BER vs SNR: {payload['scheme']}")
    plt.ylim(bottom=1e-7)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def run_experiment_config(cfg: dict, output_tag: str | None = None) -> dict:
    logger = build_logger()
    snr_cfg = cfg.get("snr_db", {"start": 0, "stop": 30, "step": 2})
    snr_grid = make_snr_grid(int(snr_cfg["start"]), int(snr_cfg["stop"]), int(snr_cfg["step"]))
    seeds = list(cfg.get("seeds", [int(cfg.get("seed", 42))]))

    jobs = [(float(snr), int(seed)) for snr in snr_grid for seed in seeds]
    parallel = bool(cfg.get("parallel", {}).get("enabled", False))
    workers = int(cfg.get("parallel", {}).get("workers", max((os.cpu_count() or 2) // 2, 1)))

    t0 = time.perf_counter()
    per_run: list[dict] = []

    if parallel and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_job, cfg, snr, seed) for snr, seed in jobs]
            for fut in futures:
                per_run.append(fut.result())
    else:
        for snr, seed in jobs:
            per_run.append(_run_job(cfg, snr, seed))

    all_rows: list[dict] = []
    for snr in snr_grid:
        snr_runs = [r for r in per_run if abs(r["snr_db"] - float(snr)) < 1e-9]
        ber_samples = np.asarray([r["ber"] for r in snr_runs], dtype=float)
        ser_samples = np.asarray([r["ser"] for r in snr_runs], dtype=float)

        ber_mean = float(np.mean(ber_samples))
        ser_mean = float(np.mean(ser_samples))
        ber_std = float(np.std(ber_samples, ddof=1)) if len(ber_samples) > 1 else 0.0
        ser_std = float(np.std(ser_samples, ddof=1)) if len(ser_samples) > 1 else 0.0
        ci95 = 1.96 * ber_std / np.sqrt(max(len(ber_samples), 1))

        row = {
            "snr_db": float(snr),
            "ber": ber_mean,
            "ser": ser_mean,
            "ber_std": ber_std,
            "ser_std": ser_std,
            "ber_ci95": float(ci95),
            "bit_errors": int(np.sum([r["bit_errors"] for r in snr_runs])),
            "total_bits": int(np.sum([r["total_bits"] for r in snr_runs])),
            "scheme": cfg.get("scheme", "unknown"),
            "n_symbols": int(cfg.get("n_symbols", 100000)),
            "seeds": "|".join(str(s) for s in seeds),
        }
        all_rows.append(row)
        logger.info(
            "scheme=%s snr=%.1f ber=%.3e +/- %.1e",
            row["scheme"],
            snr,
            row["ber"],
            row["ber_ci95"],
        )

    runtime_s = time.perf_counter() - t0
    tag = output_tag or cfg.get("scheme", "experiment")

    payload = {
        "snr_db": [r["snr_db"] for r in all_rows],
        "ber": [r["ber"] for r in all_rows],
        "ser": [r["ser"] for r in all_rows],
        "ber_ci95": [r["ber_ci95"] for r in all_rows],
        "scheme": cfg.get("scheme", "unknown"),
        "config": cfg,
        "runtime_s": float(runtime_s),
        "seed_runs": per_run,
    }

    raw_path = Path("results/raw/ber_logs") / f"{tag}.json"
    csv_path = Path("results/processed/csv") / f"{tag}.csv"
    npy_path = Path("results/processed/numpy") / f"{tag}.npz"
    plot_path = Path("results/plots/ber_curves") / f"{tag}.png"

    save_json(raw_path, payload)
    save_csv(csv_path, all_rows)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npy_path,
        snr_db=np.asarray(payload["snr_db"], dtype=float),
        ber=np.asarray(payload["ber"], dtype=float),
        ser=np.asarray(payload["ser"], dtype=float),
        ber_ci95=np.asarray(payload["ber_ci95"], dtype=float),
    )
    _plot_curve(payload, plot_path)

    logger.info(
        "saved json=%s csv=%s npz=%s plot=%s runtime=%.2fs",
        raw_path,
        csv_path,
        npy_path,
        plot_path,
        runtime_s,
    )
    return payload


def run_experiment(config_path: str, output_tag: str | None = None) -> dict:
    return run_experiment_config(load_config(config_path), output_tag=output_tag)
