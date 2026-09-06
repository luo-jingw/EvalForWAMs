"""Render the FastWAM VRAM breakdown with the T5 text encoder shown as a
GHOSTED (hatched, translucent) segment: it occupies real memory but is NOT
counted in the accounted total, because it is offloadable offline (precompute
the text embeddings once and inject them as `context`; see NO_CALIBRATION.md /
the T5-offload note). The accounted peak is thus the quantization-relevant
working set (transformer weights + VAE + KV + activation), with T5 drawn behind
it as "occupied but excluded".

Reads the measure_memory.py output (measured_kv_cache.json) and writes a PNG.
Standalone (not the shared calc_cross_ckpt memory_breakdown) so the ghosting is
fully controlled here.

CLI:
    python -m ptqeval.wam.fastwam.plot_memory_breakdown \\
        --measured_kv_cache results/fastwam/fastwam_w4a4/summary/measured_kv_cache.json \\
        --title "FastWAM W4A4 VRAM (T5 offloadable)" \\
        --output results/fastwam/fastwam_w4a4/cross_summary/plots/memory_breakdown.png
"""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colors match the shared calc_cross_ckpt memory_breakdown legend.
_C_TEXT = "#7B5CB8"   # T5 (ghosted)
_C_XFMR = "#2F7AB5"   # transformer weights
_C_KV = "#E8973A"     # KV cache
_C_VAE = "#3FA05A"    # VAE
_C_ACT = "#9AA0A6"    # activations + scratch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured_kv_cache", required=True)
    ap.add_argument("--title", default="FastWAM W4A4 VRAM breakdown")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    s = json.load(open(args.measured_kv_cache))["samples"]
    gb = lambda mb: float(mb) / 1024.0
    xfmr = gb(s["transformer_weight_mb"])
    vae = gb(s["vae_weight_mb"])
    kv = gb(s["kv_cache_mb"])
    act = gb(s["activation_peak_mb"])
    t5 = gb(s["text_encoder_weight_mb"])

    accounted = xfmr + vae + kv + act          # counted total (T5 excluded)
    full = accounted + t5                       # what is actually resident now

    # Counted (solid) segments, left to right.
    counted = [
        ("Transformer weights", xfmr, _C_XFMR),
        ("VAE", vae, _C_VAE),
        ("KV cache", kv, _C_KV),
        ("Activations + scratch", act, _C_ACT),
    ]

    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    y = 0.0
    left = 0.0
    # Value labels go OUTSIDE the bar, into the legend (label — value / %).
    for label, val, color in counted:
        pct = f"  ({val / accounted * 100:.0f}%)" if val / accounted > 0.02 else ""
        ax.barh(y, val, left=left, height=0.62, color=color,
                edgecolor="white", linewidth=0.8, zorder=3,
                label=f"{label} — {val:.1f} GB{pct}")
        left += val

    # Ghosted T5: translucent + hatched, appended after the counted block.
    ax.barh(y, t5, left=left, height=0.62, facecolor=_C_TEXT, alpha=0.22,
            edgecolor=_C_TEXT, linewidth=1.4, hatch="////", zorder=2,
            label=f"T5 text encoder — {t5:.1f} GB  (not counted)")

    # Reference lines (labelled outside/above the bar, not on the bars).
    ax.axvline(accounted, color="#333333", lw=2.0, zorder=5)
    ax.text(accounted, 0.44, f"accounted peak {accounted:.1f} GB (T5 offloaded)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#333333")
    ax.axvline(full, color=_C_TEXT, lw=1.6, linestyle="--", alpha=0.6, zorder=5)
    ax.text(full, 0.44, f"resident now {full:.1f} GB", ha="center", va="bottom",
            fontsize=9, color=_C_TEXT, alpha=0.85)

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlim(0, full * 1.03)
    ax.set_xlabel("VRAM (GB)")
    ax.set_title(args.title, fontsize=12, pad=26)
    # Legend OUTSIDE the axes (right), carrying all the value labels.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              framealpha=0.95, ncol=1, title="segments (GB)", title_fontsize=9)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(0.02, 0.02,
             "Solid = accounted working set (quantization-relevant). Hatched = T5, "
             "physically resident but excluded — offloadable offline (precompute "
             "text embeddings, inject as context).",
             ha="left", va="bottom", fontsize=8.3, color="#555555")
    fig.subplots_adjust(left=0.045, right=0.70, bottom=0.30, top=0.80)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"  accounted (T5 offloaded) = {accounted:.2f} GB ; resident now = {full:.2f} GB "
          f"(T5 = {t5:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
