from __future__ import annotations

import numpy as np

from src.channel import correlated, noise, rayleigh
from src.modulation.mapper import Mapper
from src.schemes import alamouti, scirs_2x1, scirs_3x1


def _channel_sample(cfg: dict, n_samples: int, n_tx: int) -> np.ndarray:
    ch = cfg.get("channel", {})
    if ch.get("type", "rayleigh").lower() == "correlated":
        rho = float(ch.get("correlation_rho", 0.0))
        return correlated.sample(n_samples, n_tx, rho)
    return rayleigh.sample(n_samples, n_tx)


def _ml_slicer(y_eq: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    metric = np.abs(y_eq[:, None] - constellation[None, :]) ** 2
    idx = np.argmin(metric, axis=1)
    return constellation[idx]


def _run_siso(symbols: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = _channel_sample(cfg, len(symbols), 1)
    y = h[:, 0] * symbols
    y_n = noise.add_awgn(y, snr_db)
    y_eq = y_n / (h[:, 0] + 1e-12)
    s_hat = _ml_slicer(y_eq, constellation)
    return s_hat, symbols


def _run_mrc(symbols: np.ndarray, cfg: dict, snr_db: float, n_tx: int, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = _channel_sample(cfg, len(symbols), n_tx)
    x = np.repeat(symbols[:, None], n_tx, axis=1) / np.sqrt(n_tx)
    # Lx1 MISO receive model with one scalar output per symbol.
    y = np.sum(h * x, axis=1)
    y_n = noise.add_awgn(y, snr_db)

    heff = np.sum(h, axis=1) / np.sqrt(n_tx)
    y_eq = y_n / (heff + 1e-12)
    s_hat = _ml_slicer(y_eq, constellation)
    return s_hat, symbols


def _run_alamouti(symbols: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_used = (len(symbols) // 2) * 2
    s = symbols[:n_used]
    blocks = s.reshape(-1, 2)

    h = _channel_sample(cfg, len(blocks), 2)
    x = alamouti.encode(s)

    y = np.sum(h[:, None, :] * x, axis=2)
    y_n = noise.add_awgn(y, snr_db)
    s_eq = alamouti.detect(y_n, h)
    s_hat = _ml_slicer(s_eq, constellation)
    return s_hat, s


def _run_scirs_2x1(symbols: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = float(cfg.get("rotation", {}).get("theta_rad", scirs_2x1.THETA_OPT))
    if "theta_deg" in cfg.get("rotation", {}):
        theta = np.deg2rad(float(cfg["rotation"]["theta_deg"]))

    n_used = (len(symbols) // 2) * 2
    s = symbols[:n_used].reshape(-1, 2)
    h = _channel_sample(cfg, len(s), 2)

    x = scirs_2x1.encode(s, theta)
    y = np.sum(h * x, axis=1)
    y_n = noise.add_awgn(y, snr_db)

    s_hat_blocks = scirs_2x1.ml_detect(y_n, h, constellation, theta)
    s_hat = s_hat_blocks.reshape(-1)
    return s_hat, s.reshape(-1)


def _run_scirs_3x1(symbols: np.ndarray, cfg: dict, snr_db: float, constellation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_used = (len(symbols) // 3) * 3
    s = symbols[:n_used].reshape(-1, 3)
    h = _channel_sample(cfg, len(s), 3)

    x = scirs_3x1.encode(s)
    y = np.sum(h * x, axis=1)
    y_n = noise.add_awgn(y, snr_db)

    s_hat_blocks = scirs_3x1.ml_detect(y_n, h, constellation)
    s_hat = s_hat_blocks.reshape(-1)
    return s_hat, s.reshape(-1)


def run_single_snr(cfg: dict, snr_db: float, seed: int) -> dict:
    np.random.seed(seed)
    mapper = Mapper(cfg.get("modulation", "qpsk"))
    constellation = mapper.constellation

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
            s_hat, s_ref = _run_siso(syms, cfg, snr_db, constellation)
        elif scheme == "mrc_2x1":
            s_hat, s_ref = _run_mrc(syms, cfg, snr_db, n_tx=2, constellation=constellation)
        elif scheme == "mrc_3x1":
            s_hat, s_ref = _run_mrc(syms, cfg, snr_db, n_tx=3, constellation=constellation)
        elif scheme == "alamouti_2x1":
            s_hat, s_ref = _run_alamouti(syms, cfg, snr_db, constellation)
        elif scheme == "scirs_2x1":
            s_hat, s_ref = _run_scirs_2x1(syms, cfg, snr_db, constellation)
        elif scheme == "scirs_3x1":
            s_hat, s_ref = _run_scirs_3x1(syms, cfg, snr_db, constellation)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        bits_ref = mapper.demodulate(s_ref)
        bits_hat = mapper.demodulate(s_hat)
        syms_hat_hard = mapper.modulate(bits_hat)

        bit_errors += int(np.sum(bits_ref != bits_hat))
        total_bits += int(bits_ref.size)
        sym_errors += int(np.sum(np.abs(s_ref - syms_hat_hard) > 1e-12))
        total_syms += int(len(s_ref))

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


def run_sanity_checks(seed: int = 1234) -> dict[str, bool]:
    cfg = {
        "modulation": "qpsk",
        "n_symbols": 2000,
        "max_errors": 2000,
        "batch_size": 2000,
        "channel": {"type": "rayleigh"},
        "rotation": {"theta_rad": scirs_2x1.THETA_OPT},
    }

    checks = {}

    # High-SNR sanity (near-zero BER expected).
    for scheme in ["siso", "mrc_2x1", "mrc_3x1", "alamouti_2x1", "scirs_2x1", "scirs_3x1"]:
        cfg["scheme"] = scheme
        out = run_single_snr(cfg, snr_db=30.0, seed=seed)
        checks[f"{scheme}_high_snr_ok"] = out["ber"] < 1e-3
        checks[f"{scheme}_not_random_guess"] = out["ber"] < 0.2

    return checks
