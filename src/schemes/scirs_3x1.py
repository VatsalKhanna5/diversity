from __future__ import annotations

import itertools

import numpy as np

from src.schemes.rotation import rotate_iq


def _ciod3_map(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = np.real(r1) + 1j * np.imag(r2)
    u2 = np.real(r2) + 1j * np.imag(r3)
    u3 = np.real(r3) + 1j * np.imag(r1)
    return u1, u2, u3


def transmit(symbols: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 35.0)))

    n_blocks = len(symbols) // 3
    used = symbols[: 3 * n_blocks].reshape(-1, 3)
    r1 = rotate_iq(used[:, 0], theta)
    r2 = rotate_iq(used[:, 1], theta)
    r3 = rotate_iq(used[:, 2], theta)

    u1, u2, u3 = _ciod3_map(r1, r2, r3)

    tx = np.zeros((n_blocks, 3, 3), dtype=complex)
    tx[:, 0, 0] = u1
    tx[:, 1, 1] = u2
    tx[:, 2, 2] = u3

    norm = np.sqrt(np.mean(np.abs(tx) ** 2) * tx.shape[2])
    tx /= max(norm, 1e-12)

    return tx, {"n_used_symbols": int(3 * n_blocks), "theta_deg": np.rad2deg(theta)}


def _candidate_blocks(constellation: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(list(itertools.product(constellation, repeat=3)), dtype=complex)
    r1 = rotate_iq(groups[:, 0], theta)
    r2 = rotate_iq(groups[:, 1], theta)
    r3 = rotate_iq(groups[:, 2], theta)

    u1, u2, u3 = _ciod3_map(r1, r2, r3)

    tx = np.zeros((len(groups), 3, 3), dtype=complex)
    tx[:, 0, 0] = u1
    tx[:, 1, 1] = u2
    tx[:, 2, 2] = u3
    norm = np.sqrt(np.mean(np.abs(tx) ** 2) * tx.shape[2])
    tx /= max(norm, 1e-12)

    return groups, tx


def receive(rx_signal: np.ndarray, channel: np.ndarray, config: dict, constellation: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(float(config.get("rotation", {}).get("theta_deg", 35.0)))
    groups, tx_candidates = _candidate_blocks(constellation, theta)

    pred = np.sum(channel[:, None, None, :] * tx_candidates[None, :, :, :], axis=3)
    metric = np.sum(np.abs(rx_signal[:, None, :] - pred) ** 2, axis=2)

    idx = np.argmin(metric, axis=1)
    return groups[idx].reshape(-1)
