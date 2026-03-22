from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import save_json


def comparison_plot(curves: list[dict], title: str, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.2))
    for c in curves:
        plt.semilogy(c["snr_db"], c["ber"], marker="o", label=c["scheme"])
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=260)
    plt.close()


def make_summary_table(curves: list[dict], target_ber: float, out_markdown: str | Path) -> None:
    rows = []
    for c in curves:
        snr = np.asarray(c["snr_db"], dtype=float)
        ber = np.asarray(c["ber"], dtype=float)
        if np.all(ber > target_ber):
            req_snr = float("nan")
        else:
            req_snr = float(np.interp(np.log10(target_ber), np.log10(ber[::-1]), snr[::-1]))
        rows.append({"scheme": c["scheme"], "snr_at_target_ber": req_snr})

    rows = sorted(rows, key=lambda x: (np.isnan(x["snr_at_target_ber"]), x["snr_at_target_ber"]))

    md = ["| Scheme | SNR @ BER target (dB) |", "|---|---:|"]
    for r in rows:
        value = "N/A" if np.isnan(r["snr_at_target_ber"]) else f"{r['snr_at_target_ber']:.2f}"
        md.append(f"| {r['scheme']} | {value} |")

    out = Path(out_markdown)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")

    save_json(out.with_suffix(".json"), {"target_ber": target_ber, "rows": rows})
