import numpy as np


_LEVELS = np.array([-3, -1, 1, 3], dtype=float)
_NORM = np.sqrt(10.0)


def constellation() -> np.ndarray:
    grid = _LEVELS[:, None] + 1j * _LEVELS[None, :]
    return grid.reshape(-1) / _NORM


def _pair_to_level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    idx = (msb.astype(int) << 1) | lsb.astype(int)
    gray_to_bin = np.array([0, 1, 3, 2])
    return _LEVELS[gray_to_bin[idx]]


def modulate(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 4)
    i = _pair_to_level(b[:, 0], b[:, 1])
    q = _pair_to_level(b[:, 2], b[:, 3])
    return (i + 1j * q) / _NORM


def demodulate(symbols: np.ndarray) -> np.ndarray:
    s = symbols * _NORM
    i_idx = np.argmin(np.abs(np.real(s)[:, None] - _LEVELS[None, :]), axis=1)
    q_idx = np.argmin(np.abs(np.imag(s)[:, None] - _LEVELS[None, :]), axis=1)

    bin_to_gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    ib = bin_to_gray[i_idx]
    qb = bin_to_gray[q_idx]
    return np.column_stack([ib[:, 0], ib[:, 1], qb[:, 0], qb[:, 1]]).reshape(-1)
