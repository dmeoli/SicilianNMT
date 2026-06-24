#!/usr/bin/env python3
"""Generate the results charts for the HLT slide deck, in the Sicilian palette
(Mediterranean blue + complementary gold). Outputs to docs/slides/img/.

    python docs/slides/make_charts.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0E4D8C"        # Mediterranean / majolica blue (primary)
NAVY = "#072373"        # deep blue
GOLD = "#D69E2E"        # Sicilian gold/ochre (complement)
INK = "#282828"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
# light -> dark blue ramp for the sequential progression
blue_ramp = LinearSegmentedColormap.from_list("scn", ["#BcD3EA", BLUE, NAVY])


def progression():
    labels = ["floor\n(copy)", "Sockeye\nbaseline", "+ lever B", "+ more\ndata",
              "+ lever D", "NLLB-1.3B\nzero-shot", "NLLB-1.3B\nbidir LoRA"]
    bleu = [5.27, 5.54, 7.24, 9.79, 10.85, 29.00, 31.33]
    n = len(bleu)
    colors = [blue_ramp(i / (n - 1)) for i in range(n)]
    colors[-1] = GOLD                      # our best result pops in gold
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(range(n), bleu, color=colors, width=0.72, zorder=3)
    # Wdowiak published baseline (his own test set) as a reference line
    ax.axhline(29.1, ls="--", lw=1.4, color=INK, zorder=2)
    ax.text(0.1, 29.1 + 0.7, "Wdowiak baseline 29.1 (his test set)",
            fontsize=10.5, color=INK)
    for b, v in zip(bars, bleu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if v == bleu[-1] else "normal",
                color=GOLD if v == bleu[-1] else INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("BLEU  (scn$\\to$en, frozen test)")
    ax.set_ylim(0, 35)
    ax.set_title("From a 6.6M from-scratch model to a fine-tuned NLLB-1.3B",
                 fontsize=12.5, color=NAVY, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "bleu_progression.png", dpi=200)
    plt.close(fig)


def directions():
    cats = ["scn$\\to$en", "en$\\to$scn"]
    zero = [29.00, 9.89]
    ft = [31.33, 18.65]
    x = range(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar([i - w / 2 for i in x], zero, w, label="zero-shot", color=BLUE, zorder=3)
    ax.bar([i + w / 2 for i in x], ft, w, label="bidir LoRA", color=GOLD, zorder=3)
    for i, (z, f) in enumerate(zip(zero, ft)):
        ax.text(i - w / 2, z + 0.4, f"{z:.1f}", ha="center", fontsize=10.5, color=INK)
        ax.text(i + w / 2, f + 0.4, f"{f:.1f}", ha="center", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("BLEU (frozen test)")
    ax.set_ylim(0, 36)
    ax.set_title("Bidirectional fine-tune\nrescues the weak direction",
                 fontsize=12, color=NAVY)
    ax.legend(frameon=False, fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "bleu_directions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    progression()
    directions()
    print("wrote bleu_progression.png and bleu_directions.png to", OUT)
