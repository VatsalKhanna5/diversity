import numpy as np


def _toeplitz_correlation(n_tx: int, rho: float) -> np.ndarray:
    idx = np.arange(n_tx)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample(n_samples: int, n_tx: int, rho: float) -> np.ndarray:
    r = _toeplitz_correlation(n_tx, rho)
    l = np.linalg.cholesky(r)
    w = (np.random.randn(n_samples, n_tx) + 1j * np.random.randn(n_samples, n_tx)) / np.sqrt(2.0)
    return w @ l.T
