from __future__ import annotations

import numpy as np


_SQRT2 = np.sqrt(2.0)

# Gray mapping requested by the spec.
_BITS = np.array(
    [
        [0, 0],  # +1 + j
        [0, 1],  # -1 + j
        [1, 1],  # -1 - j
        [1, 0],  # +1 - j
    ],
    dtype=np.uint8,
)
_CONSTELLATION = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=complex) / _SQRT2


def constellation() -> np.ndarray:
    return _CONSTELLATION.copy()


def modulate(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 2)
    idx = np.where(
        (b[:, 0] == 0) & (b[:, 1] == 0),
        0,
        np.where((b[:, 0] == 0) & (b[:, 1] == 1), 1, np.where((b[:, 0] == 1) & (b[:, 1] == 1), 2, 3)),
    )
    return _CONSTELLATION[idx]


def demodulate(symbols: np.ndarray) -> np.ndarray:
    metric = np.abs(symbols[:, None] - _CONSTELLATION[None, :]) ** 2
    idx = np.argmin(metric, axis=1)
    return _BITS[idx].reshape(-1)
