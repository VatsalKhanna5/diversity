from __future__ import annotations

import itertools

import numpy as np


class MLDetector:
    def __init__(self, constellation: np.ndarray, n_tx: int = 1) -> None:
        self.constellation = np.asarray(constellation)
        self.n_tx = n_tx
        candidates = list(itertools.product(self.constellation, repeat=n_tx))
        self.symbol_vectors = np.asarray(candidates, dtype=complex)

    def detect(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        # Vectorized exhaustive metric over all candidates.
        pred = h[:, None, :] * self.symbol_vectors[None, :, :]
        pred_sum = np.sum(pred, axis=-1)
        metric = np.abs(y[:, None] - pred_sum) ** 2
        best_idx = np.argmin(metric, axis=1)
        return self.symbol_vectors[best_idx]
