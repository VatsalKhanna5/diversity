from __future__ import annotations

import itertools

import numpy as np

from src.schemes.rotation import rotate_iq


def _ciod_map(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u1 = np.real(r1) + 1j * np.imag(r2)
    u2 = np.real(r2) + 1j * np.imag(r1)
    return u1, u2


def transmit(symbols: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 31.7175)))

    n_blocks = len(symbols) // 2
    used = symbols[: 2 * n_blocks].reshape(-1, 2)
    r1 = rotate_iq(used[:, 0], theta)
    r2 = rotate_iq(used[:, 1], theta)

    u1, u2 = _ciod_map(r1, r2)

    tx = np.zeros((n_blocks, 2, 2), dtype=complex)
    tx[:, 0, 0] = u1
    tx[:, 1, 1] = u2

    norm = np.sqrt(np.mean(np.abs(tx) ** 2) * tx.shape[2])
    tx /= max(norm, 1e-12)

    return tx, {"n_used_symbols": int(2 * n_blocks), "theta_deg": np.rad2deg(theta)}


def _candidate_blocks(constellation: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(list(itertools.product(constellation, repeat=2)), dtype=complex)
    r1 = rotate_iq(pairs[:, 0], theta)
    r2 = rotate_iq(pairs[:, 1], theta)
    u1, u2 = _ciod_map(r1, r2)

    tx = np.zeros((len(pairs), 2, 2), dtype=complex)
    tx[:, 0, 0] = u1
    tx[:, 1, 1] = u2
    norm = np.sqrt(np.mean(np.abs(tx) ** 2) * tx.shape[2])
    tx /= max(norm, 1e-12)

    return pairs, tx


def receive(rx_signal: np.ndarray, channel: np.ndarray, config: dict, constellation: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 31.7175)))
    pairs, tx_candidates = _candidate_blocks(constellation, theta)

    # rx_signal: [n_blocks, 2], channel: [n_blocks, 2], tx_candidates: [n_cand, 2, 2]
    pred = np.sum(channel[:, None, None, :] * tx_candidates[None, :, :, :], axis=3)
    metric = np.sum(np.abs(rx_signal[:, None, :] - pred) ** 2, axis=2)

    idx = np.argmin(metric, axis=1)
    return pairs[idx].reshape(-1)
