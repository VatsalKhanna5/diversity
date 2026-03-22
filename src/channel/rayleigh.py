import numpy as np


def sample(n_samples: int, n_tx: int) -> np.ndarray:
    return (np.random.randn(n_samples, n_tx) + 1j * np.random.randn(n_samples, n_tx)) / np.sqrt(2.0)
