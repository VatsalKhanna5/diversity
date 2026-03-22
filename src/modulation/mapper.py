from __future__ import annotations

from typing import Callable

import numpy as np

from src.modulation import qam16, qpsk


class Mapper:
    def __init__(self, name: str) -> None:
        key = name.lower()
        table: dict[str, tuple[int, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[], np.ndarray]]]
        table = {
            "qpsk": (2, qpsk.modulate, qpsk.demodulate, qpsk.constellation),
            "16qam": (4, qam16.modulate, qam16.demodulate, qam16.constellation),
            "qam16": (4, qam16.modulate, qam16.demodulate, qam16.constellation),
        }
        if key not in table:
            raise ValueError(f"Unsupported modulation: {name}")

        self.bits_per_symbol, self.modulate, self.demodulate, self._constellation_fn = table[key]

    @property
    def constellation(self) -> np.ndarray:
        return self._constellation_fn()
