import numpy as np


def encode(symbols: np.ndarray) -> np.ndarray:
    pairs = symbols.reshape(-1, 2)
    s1, s2 = pairs[:, 0], pairs[:, 1]

    x_t1 = np.column_stack([s1, s2])
    x_t2 = np.column_stack([-np.conj(s2), np.conj(s1)])
    return np.stack([x_t1, x_t2], axis=1)


def detect(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    # y: [n_blocks, 2 time slots], h: [n_blocks, 2 tx]
    y1, y2 = y[:, 0], y[:, 1]
    h1, h2 = h[:, 0], h[:, 1]

    z1 = np.conj(h1) * y1 + h2 * np.conj(y2)
    z2 = np.conj(h2) * y1 - h1 * np.conj(y2)
    denom = np.abs(h1) ** 2 + np.abs(h2) ** 2 + 1e-12
    s_hat = np.column_stack([z1 / denom, z2 / denom])
    return s_hat.reshape(-1)
