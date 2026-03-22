import numpy as np


SQRT2 = np.sqrt(2.0)


def constellation() -> np.ndarray:
    return np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=complex) / SQRT2


def modulate(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 2)
    sym = (2 * b[:, 0] - 1) + 1j * (2 * b[:, 1] - 1)
    return sym / SQRT2


def demodulate(symbols: np.ndarray) -> np.ndarray:
    bits_i = (np.real(symbols) > 0).astype(np.uint8)
    bits_q = (np.imag(symbols) > 0).astype(np.uint8)
    return np.stack([bits_i, bits_q], axis=1).reshape(-1)
