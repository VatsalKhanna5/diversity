from __future__ import annotations

import itertools

import numpy as np

from src.schemes.rotation import rotation_matrix_2d


def _interleave_pair(s1: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v1 = np.real(s1) + 1j * np.imag(s2)
    v2 = np.real(s2) + 1j * np.imag(s1)
    return v1, v2


def _deinterleave_pair(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.real(v1) + 1j * np.imag(v2)
    s2 = np.real(v2) + 1j * np.imag(v1)
    return s1, s2


def transmit(symbols: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 31.7175)))
    g = rotation_matrix_2d(theta)

    n_pairs = len(symbols) // 2
    trimmed = symbols[: 2 * n_pairs]
    pairs = trimmed.reshape(-1, 2)

    v1, v2 = _interleave_pair(pairs[:, 0], pairs[:, 1])
    v = np.column_stack([v1, v2])
    x = v @ g.T
    x = x / np.sqrt(np.mean(np.sum(np.abs(x) ** 2, axis=1)))

    meta = {
        "theta": theta,
        "n_used_symbols": int(2 * n_pairs),
        "rotation": g,
    }
    return x, meta


def _candidate_tx(constellation: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    g = rotation_matrix_2d(theta)
    pairs = np.asarray(list(itertools.product(constellation, repeat=2)), dtype=complex)
    v1, v2 = _interleave_pair(pairs[:, 0], pairs[:, 1])
    v = np.column_stack([v1, v2])
    x = v @ g.T
    x = x / np.sqrt(np.mean(np.sum(np.abs(x) ** 2, axis=1)))
    return pairs, x


def receive(rx_signal: np.ndarray, channel: np.ndarray, config: dict, constellation: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 31.7175)))
    pairs, x_candidates = _candidate_tx(constellation, theta)

    pred = np.sum(channel[:, None, :] * x_candidates[None, :, :], axis=2)
    metric = np.abs(rx_signal[:, None] - pred) ** 2
    idx = np.argmin(metric, axis=1)
    s_hat = pairs[idx]
    return s_hat.reshape(-1)
