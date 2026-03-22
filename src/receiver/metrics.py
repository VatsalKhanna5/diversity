from __future__ import annotations

import numpy as np


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return float(np.mean(tx_bits != rx_bits))


def ser(tx_symbols: np.ndarray, rx_symbols: np.ndarray, atol: float = 1e-6) -> float:
    return float(np.mean(np.abs(tx_symbols - rx_symbols) > atol))
