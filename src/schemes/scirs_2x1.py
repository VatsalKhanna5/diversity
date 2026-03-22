from __future__ import annotations

import itertools

import numpy as np


THETA_OPT = 0.5 * np.arctan(2.0)


def rotation_matrix(theta_rad: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]],
        dtype=float,
    )


def encode(symbol_pairs: np.ndarray, theta_rad: float) -> np.ndarray:
    # symbol_pairs: [n, 2]
    g = rotation_matrix(theta_rad)
    return symbol_pairs @ g.T


def candidate_codebook(constellation: np.ndarray, theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(list(itertools.product(constellation, repeat=2)), dtype=complex)
    x = encode(pairs, theta_rad)
    return pairs, x


def ml_detect(y: np.ndarray, h: np.ndarray, constellation: np.ndarray, theta_rad: float) -> np.ndarray:
    pairs, x_codebook = candidate_codebook(constellation, theta_rad)
    # pred: [n_samples, n_candidates]
    pred = np.sum(h[:, None, :] * x_codebook[None, :, :], axis=2)
    metric = np.abs(y[:, None] - pred) ** 2
    idx = np.argmin(metric, axis=1)
    return pairs[idx]
