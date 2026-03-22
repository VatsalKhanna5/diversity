import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from src.modulation.mapper import Mapper
from src.schemes.rotation import rotate_iq


if __name__ == "__main__":
    mapper = Mapper("qpsk")
    theta_deg = 31.7175
    theta = np.deg2rad(theta_deg)

    original = mapper.constellation
    rotated = rotate_iq(original, theta)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pts, title in [
        (axes[0], original, "Original QPSK"),
        (axes[1], rotated, f"Rotated QPSK ({theta_deg:.2f} deg)"),
    ]:
        ax.scatter(np.real(pts), np.imag(pts), s=60)
        for p in pts:
            ax.annotate(f"({p.real:.2f},{p.imag:.2f})", (p.real, p.imag), fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("results/plots/constellation/exp08_qpsk_rotation.png", dpi=260)
    plt.close()
