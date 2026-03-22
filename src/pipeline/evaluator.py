from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.receiver.metrics import ber, ser


def make_snr_grid(start: int, stop: int, step: int) -> np.ndarray:
    return np.arange(start, stop + step, step)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(tx_bits: np.ndarray, rx_bits: np.ndarray, tx_syms: np.ndarray, rx_syms: np.ndarray) -> dict[str, float]:
    return {"ber": ber(tx_bits, rx_bits), "ser": ser(tx_syms, rx_syms)}
