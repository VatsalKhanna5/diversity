from __future__ import annotations

import numpy as np

from src.channel import correlated, noise, rayleigh
from src.modulation.mapper import Mapper
from src.schemes import alamouti, mrc, scirs_2x1, scirs_3x1


def _channel_sample(cfg: dict, n_samples: int, n_tx: int) -> np.ndarray:
    ch = cfg.get("channel", {})
    if ch.get("type", "rayleigh").lower() == "correlated":
        return correlated.sample(n_samples, n_tx, float(ch.get("correlation_rho", 0.5)))
    return rayleigh.sample(n_samples, n_tx)


def _run_siso(syms: np.ndarray, cfg: dict, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    h = _channel_sample(cfg, len(syms), 1)
    y = np.sum(h * syms[:, None], axis=1)
    y_n = noise.add_awgn(y, snr_db)
    s_hat = y_n / (h[:, 0] + 1e-12)
    return s_hat, syms


def _run_mrc(syms: np.ndarray, cfg: dict, snr_db: float, n_tx: int) -> tuple[np.ndarray, np.ndarray]:
    h = _channel_sample(cfg, len(syms), n_tx)
    x = mrc.transmit(syms, n_tx=n_tx)
    y_b = h * x
    y_b_n = noise.add_awgn(y_b, snr_db)
    s_hat = mrc.detect(y_b_n, h)
    return s_hat, syms


def _run_alamouti(syms: np.ndarray, cfg: dict, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    n_used = (len(syms) // 2) * 2
    trimmed = syms[:n_used]
    blocks = trimmed.reshape(-1, 2)

    h = _channel_sample(cfg, len(blocks), 2)
    x = alamouti.encode(trimmed)
    y = np.sum(h[:, None, :] * x, axis=2)
    y_n = noise.add_awgn(y, snr_db)
    s_hat = alamouti.detect(y_n, h)
    return s_hat, trimmed


def _run_scirs_2x1(syms: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx, meta = scirs_2x1.transmit(syms, cfg)
    h = _channel_sample(cfg, tx.shape[0], 2)
    y = np.sum(h[:, None, :] * tx, axis=2)
    y_n = noise.add_awgn(y, snr_db)
    s_hat = scirs_2x1.receive(y_n, h, cfg, constellation)
    return s_hat, syms[: meta["n_used_symbols"]]


def _run_scirs_3x1(syms: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx, meta = scirs_3x1.transmit(syms, cfg)
    h = _channel_sample(cfg, tx.shape[0], 3)
    y = np.sum(h[:, None, :] * tx, axis=2)
    y_n = noise.add_awgn(y, snr_db)
    s_hat = scirs_3x1.receive(y_n, h, cfg, constellation)
    return s_hat, syms[: meta["n_used_symbols"]]


def run_single_snr(cfg: dict, snr_db: float, seed: int) -> dict:
    np.random.seed(seed)
    mapper = Mapper(cfg.get("modulation", "qpsk"))

    n_symbols = int(cfg.get("n_symbols", 100000))
    max_errors = int(cfg.get("max_errors", 200))
    batch_size = int(cfg.get("batch_size", 5000))
    scheme = cfg.get("scheme", "siso").lower()

    total_bits = 0
    bit_errors = 0
    total_syms = 0
    sym_errors = 0

    while total_syms < n_symbols and bit_errors < max_errors:
        n_batch = min(batch_size, n_symbols - total_syms)
        bits = np.random.randint(0, 2, size=n_batch * mapper.bits_per_symbol, dtype=np.uint8)
        syms = mapper.modulate(bits)

        if scheme == "siso":
            s_hat, syms_used = _run_siso(syms, cfg, snr_db)
        elif scheme == "mrc_2x1":
            s_hat, syms_used = _run_mrc(syms, cfg, snr_db, n_tx=2)
        elif scheme == "mrc_3x1":
            s_hat, syms_used = _run_mrc(syms, cfg, snr_db, n_tx=3)
        elif scheme == "alamouti_2x1":
            s_hat, syms_used = _run_alamouti(syms, cfg, snr_db)
        elif scheme == "scirs_2x1":
            s_hat, syms_used = _run_scirs_2x1(syms, cfg, snr_db, mapper.constellation)
        elif scheme == "scirs_3x1":
            s_hat, syms_used = _run_scirs_3x1(syms, cfg, snr_db, mapper.constellation)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        bits_used = mapper.demodulate(syms_used)
        rx_bits = mapper.demodulate(s_hat)
        decided_syms = mapper.modulate(rx_bits)

        bit_errors += int(np.sum(bits_used != rx_bits))
        total_bits += int(bits_used.size)

        sym_errors += int(np.sum(np.abs(syms_used - decided_syms) > 1e-12))
        total_syms += int(len(syms_used))

    return {
        "snr_db": float(snr_db),
        "ber": bit_errors / max(total_bits, 1),
        "ser": sym_errors / max(total_syms, 1),
        "bit_errors": int(bit_errors),
        "total_bits": int(total_bits),
        "symbol_errors": int(sym_errors),
        "total_symbols": int(total_syms),
        "seed": int(seed),
        "scheme": scheme,
    }
