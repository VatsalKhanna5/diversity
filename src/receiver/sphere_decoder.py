from __future__ import annotations

import numpy as np

from src.receiver.ml_detector import MLDetector


class SphereDecoder:
    """Proxy placeholder: bounded-search approximation built on ML candidates."""

    def __init__(self, constellation: np.ndarray, n_tx: int = 1, k: int = 8) -> None:
        self.ml = MLDetector(constellation, n_tx=n_tx)
        self.k = k

    def detect(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        pred = h[:, None, :] * self.ml.symbol_vectors[None, :, :]
        pred_sum = np.sum(pred, axis=-1)
        metric = np.abs(y[:, None] - pred_sum) ** 2

        shortlist = np.argpartition(metric, kth=min(self.k, metric.shape[1] - 1), axis=1)[:, : self.k]
        best_local = np.argmin(np.take_along_axis(metric, shortlist, axis=1), axis=1)
        idx = shortlist[np.arange(len(best_local)), best_local]
        return self.ml.symbol_vectors[idx]
