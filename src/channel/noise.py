import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, symbol_energy: float = 1.0) -> np.ndarray:
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = symbol_energy / (2.0 * snr_linear)
    noise = np.sqrt(noise_var) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise
