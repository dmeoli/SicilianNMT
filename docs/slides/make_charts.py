#!/usr/bin/env python3
"""Generate the results charts for the HLT slide deck, in the Sicilian palette
(Mediterranean blue + complementary gold), once in English and once in Italian
(the latter with the suffix _it). Every number is the one of the paper
(docs/paper/tables). Outputs to docs/slides/img/.

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

# the labels of each chart, in the two languages
T = {
    "en": {
        "prog_labels": ["Sockeye\nbaseline", "+ lever B", "+ lever D", "NLLB-1.3B\nout of the box",
                        "bidir.\nLoRA", "+ back-\ntransl.", "staged", "final"],
        "prog_y": "BLEU  (scn$\\to$en, our test set)",
        "prog_title": "From a 6.7M model trained from scratch to the staged NLLB-1.3B",
        "prog_note": "Sockeye rows in the tokenized space, NLLB rows on raw text",
        "dir_legend": ["out of the box", "bidir. LoRA (web set)", "final (staged)"],
        "dir_y": "BLEU (our test set)",
        "dir_title": "Both directions, out of the box to final",
        "tri_legend": ["out of the box", "trilingual adapter"],
        "tri_y": "BLEU",
        "tri_title": "One trilingual adapter (web set): gains and forgetting",
        "tri_note": "scn-en: our test set; it-scn: WikiMatrix; it-en: FLORES-200 (not trained)",
        "sota_legend": ["Wdowiak, reverse training", "ours, final (NLLB + LoRA, staged)"],
        "sota_title": "On his test set, in his tokenized space",
        "sota_note": "121 hand-selected lines (AS38-39); his runs chose checkpoints on them, ours did not",
    },
    "it": {
        "prog_labels": ["Sockeye\nbaseline", "+ leva B", "+ leva D", "NLLB-1.3B\nsenza adatt.",
                        "LoRA\nbidir.", "+ back-\ntransl.", "a stadi", "finale"],
        "prog_y": "BLEU  (scn$\\to$en, nostro test set)",
        "prog_title": "Da un modello da 6,7M addestrato da zero a NLLB-1.3B a stadi",
        "prog_note": "righe Sockeye nello spazio tokenizzato, righe NLLB sul testo grezzo",
        "dir_legend": ["senza adattamento", "LoRA bidir. (set web)", "finale (a stadi)"],
        "dir_y": "BLEU (nostro test set)",
        "dir_title": "Le due direzioni, dal modello di base al finale",
        "tri_legend": ["senza adattamento", "adapter trilingue"],
        "tri_y": "BLEU",
        "tri_title": "Un adapter trilingue (set web): guadagni e forgetting",
        "tri_note": "scn-en: nostro test set; it-scn: WikiMatrix; it-en: FLORES-200 (non addestrate)",
        "sota_legend": ["Wdowiak, reverse training", "nostro, finale (NLLB + LoRA, a stadi)"],
        "sota_title": "Sul suo test set, nel suo spazio tokenizzato",
        "sota_note": "121 righe scelte a mano (AS38-39); i suoi modelli vi hanno scelto i checkpoint, i nostri no",
    },
}


def _finish(fig, ax, name, lang, **kw):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    fig.tight_layout()
    suffix = "" if lang == "en" else "_" + lang
    fig.savefig(OUT / f"{name}{suffix}.png", dpi=200, **kw)
    plt.close(fig)


def progression(lang):
    t = T[lang]
    bleu = [8.51, 10.94, 10.18, 29.00, 31.27, 31.60, 32.40, 36.01]
    n = len(bleu)
    colors = [blue_ramp(i / (n - 1)) for i in range(n)]
    colors[-1] = GOLD                      # the final model in gold
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bars = ax.bar(range(n), bleu, color=colors, width=0.72, zorder=3)
    for b, v in zip(bars, bleu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if v == bleu[-1] else "normal",
                color=GOLD if v == bleu[-1] else INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels(t["prog_labels"], fontsize=9.5)
    ax.set_ylabel(t["prog_y"])
    ax.set_ylim(0, 40)
    ax.set_title(t["prog_title"], fontsize=12.5, color=NAVY, pad=10)
    fig.text(0.5, -0.01, t["prog_note"], ha="center", fontsize=9, color="#777777")
    _finish(fig, ax, "bleu_progression", lang, bbox_inches="tight")


def directions(lang):
    t = T[lang]
    cats = ["scn$\\to$en", "en$\\to$scn"]
    series = [[29.00, 9.90], [31.27, 19.19], [36.01, 26.34]]
    colors = [BLUE, "#7FA6CF", GOLD]
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for k, (vals, c, lab) in enumerate(zip(series, colors, t["dir_legend"])):
        xs = [i + (k - 1) * w for i in range(len(cats))]
        ax.bar(xs, vals, w, label=lab, color=c, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=9.5, color=INK,
                    fontweight="bold" if k == 2 else "normal")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel(t["dir_y"])
    ax.set_ylim(0, 44)
    ax.set_title(t["dir_title"], fontsize=12, color=NAVY)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _finish(fig, ax, "bleu_directions", lang)


def trilingual(lang):
    t = T[lang]
    cats = ["scn$\\to$en", "en$\\to$scn", "it$\\to$scn", "scn$\\to$it", "it$\\to$en", "en$\\to$it"]
    zero = [29.00, 9.90, 17.57, 42.23, 34.42, 29.42]
    ft = [30.75, 17.49, 26.97, 43.47, 27.14, 28.77]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar([i - w / 2 for i in x], zero, w, label=t["tri_legend"][0], color=BLUE, zorder=3)
    ax.bar([i + w / 2 for i in x], ft, w, label=t["tri_legend"][1], color=GOLD, zorder=3)
    for i, (z, f) in enumerate(zip(zero, ft)):
        ax.text(i - w / 2, z + 0.5, f"{z:.1f}", ha="center", fontsize=9.5, color=INK)
        ax.text(i + w / 2, f + 0.5, f"{f:.1f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.axvspan(3.5, 5.5, color="#F3E9D2", zorder=0)   # the directions it was not trained on
    ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.set_ylabel(t["tri_y"])
    ax.set_ylim(0, 50)
    ax.set_title(t["tri_title"], fontsize=12.5, color=NAVY)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.text(0.5, -0.01, t["tri_note"], ha="center", fontsize=9, color="#777777")
    _finish(fig, ax, "bleu_trilingual", lang, bbox_inches="tight")


def vs_sota(lang):
    t = T[lang]
    cats = ["en$\\to$scn", "scn$\\to$en", "it$\\to$scn", "scn$\\to$it", "it$\\to$en", "en$\\to$it"]
    his = [45.1, 48.6, 61.4, 62.9, 48.2, 46.7]        # reverse training, third stage
    ours = [49.25, 59.34, 61.19, 60.30, 58.42, 48.77]  # final model of the ablation
    x = range(len(cats))
    w, d = 0.36, 0.2                     # a small gap between the two bars
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([i - d for i in x], his, w, label=t["sota_legend"][0], color=NAVY, zorder=3)
    ax.bar([i + d for i in x], ours, w, label=t["sota_legend"][1], color=GOLD, zorder=3)
    for i, (h, o) in enumerate(zip(his, ours)):
        ax.text(i - d, h + 0.6, f"{h:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + d, o + 0.6, f"{o:.2f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("BLEU"); ax.set_ylim(0, 78)
    ax.set_title(t["sota_title"], fontsize=12.5, color=NAVY, pad=8)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper left")
    fig.text(0.5, -0.01, t["sota_note"], ha="center", fontsize=8.5, color="#777777")
    _finish(fig, ax, "bleu_vs_sota", lang, bbox_inches="tight")


if __name__ == "__main__":
    for lang in T:
        progression(lang)
        directions(lang)
        trilingual(lang)
        vs_sota(lang)
    print("wrote the charts in", ", ".join(T), "to", OUT)
