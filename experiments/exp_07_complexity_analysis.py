import _bootstrap  # noqa: F401

import time

import matplotlib.pyplot as plt
import numpy as np

from src.modulation.mapper import Mapper
from src.receiver.ml_detector import MLDetector
from src.receiver.sphere_decoder import SphereDecoder
from src.utils.logger import save_json


if __name__ == "__main__":
    mapper_names = ["qpsk", "16qam"]
    runtimes_ml = []
    runtimes_sd = []

    n_samples = 8000
    for name in mapper_names:
        mapper = Mapper(name)
        const = mapper.constellation

        picks = np.random.randint(0, len(const), size=(n_samples, 2))
        s_vec = const[picks]
        h = (np.random.randn(n_samples, 2) + 1j * np.random.randn(n_samples, 2)) / np.sqrt(2)
        y = np.sum(h * s_vec, axis=1)

        ml = MLDetector(constellation=const, n_tx=2)
        sd = SphereDecoder(constellation=const, n_tx=2, k=8)

        t0 = time.perf_counter()
        _ = ml.detect(y, h)
        runtimes_ml.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        _ = sd.detect(y, h)
        runtimes_sd.append(time.perf_counter() - t1)

    save_json(
        "results/raw/ber_logs/exp07_complexity_runtime.json",
        {
            "modulations": mapper_names,
            "ml_runtime_s": runtimes_ml,
            "sphere_runtime_s": runtimes_sd,
        },
    )

    x = np.arange(len(mapper_names))
    w = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - w / 2, runtimes_ml, w, label="ML")
    plt.bar(x + w / 2, runtimes_sd, w, label="Sphere")
    plt.xticks(x, mapper_names)
    plt.ylabel("Runtime (s)")
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/comparison/exp07_complexity_runtime.png", dpi=240)
    plt.close()
