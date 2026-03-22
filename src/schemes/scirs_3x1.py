from __future__ import annotations

import itertools

import numpy as np


# Deterministic orthonormal rotation from QR of fixed matrix.
_BASE = np.array(
    [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
)
_Q, _ = np.linalg.qr(_BASE)
G3 = _Q


def encode(symbol_triplets: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    g = G3 if rotation is None else rotation
    return symbol_triplets @ g.T


def candidate_codebook(constellation: np.ndarray, rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(list(itertools.product(constellation, repeat=3)), dtype=complex)
    x = encode(groups, rotation=rotation)
    return groups, x


def ml_detect(y: np.ndarray, h: np.ndarray, constellation: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    groups, x_codebook = candidate_codebook(constellation, rotation=rotation)
    pred = np.sum(h[:, None, :] * x_codebook[None, :, :], axis=2)
    metric = np.abs(y[:, None] - pred) ** 2
    idx = np.argmin(metric, axis=1)
    return groups[idx]
