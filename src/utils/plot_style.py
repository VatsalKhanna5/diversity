from __future__ import annotations

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 5.0),
            "figure.dpi": 140,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.alpha": 0.32,
            "grid.linestyle": "--",
            "legend.fontsize": 10,
            "font.size": 11,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
