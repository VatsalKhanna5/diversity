from __future__ import annotations

import itertools

import numpy as np

from src.schemes.rotation import rotation_matrix_3x3


def _interleave_triplet(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1 = np.real(s1) + 1j * np.imag(s2)
    v2 = np.real(s2) + 1j * np.imag(s3)
    v3 = np.real(s3) + 1j * np.imag(s1)
    return v1, v2, v3


def _deinterleave_triplet(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1 = np.real(v1) + 1j * np.imag(v3)
    s2 = np.real(v2) + 1j * np.imag(v1)
    s3 = np.real(v3) + 1j * np.imag(v2)
    return s1, s2, s3


def transmit(symbols: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 35.0)))
    g = rotation_matrix_3x3(theta)

    n_groups = len(symbols) // 3
    trimmed = symbols[: 3 * n_groups]
    groups = trimmed.reshape(-1, 3)

    v1, v2, v3 = _interleave_triplet(groups[:, 0], groups[:, 1], groups[:, 2])
    v = np.column_stack([v1, v2, v3])
    x = v @ g.T
    x = x / np.sqrt(np.mean(np.sum(np.abs(x) ** 2, axis=1)))

    meta = {
        "theta": theta,
        "n_used_symbols": int(3 * n_groups),
        "rotation": g,
    }
    return x, meta


def _candidate_tx(constellation: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    g = rotation_matrix_3x3(theta)
    groups = np.asarray(list(itertools.product(constellation, repeat=3)), dtype=complex)
    v1, v2, v3 = _interleave_triplet(groups[:, 0], groups[:, 1], groups[:, 2])
    v = np.column_stack([v1, v2, v3])
    x = v @ g.T
    x = x / np.sqrt(np.mean(np.sum(np.abs(x) ** 2, axis=1)))
    return groups, x


def receive(rx_signal: np.ndarray, channel: np.ndarray, config: dict, constellation: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 35.0)))
    groups, x_candidates = _candidate_tx(constellation, theta)

    pred = np.sum(channel[:, None, :] * x_candidates[None, :, :], axis=2)
    metric = np.abs(rx_signal[:, None] - pred) ** 2
    idx = np.argmin(metric, axis=1)
    s_hat = groups[idx]
    return s_hat.reshape(-1)
