import numpy as np


def transmit(symbols: np.ndarray, n_tx: int) -> np.ndarray:
    # Repetition over transmit branches with power normalization.
    return np.repeat(symbols[:, None], n_tx, axis=1) / np.sqrt(n_tx)


def detect(rx: np.ndarray, h: np.ndarray) -> np.ndarray:
    w = np.conj(h)
    denom = np.sum(np.abs(h) ** 2, axis=1) + 1e-12
    return np.sum(w * rx, axis=1) / denom
