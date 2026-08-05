#!/usr/bin/env python3
"""Test Eryk's suggestion: roll several short verses into one line before aligning.

Poetry was our weakest genre (mean LaBSE 0.635, recall 0.80 vs 0.91 for standard prose;
see labse_by_text.py). Eryk (2026-08): "don't be afraid to roll several verses of poetry
into a single line -> better recall AND precision," because short verses carry little
context for the embedding.

We simulate rolling on his gold: consecutive hand-aligned pairs stay parallel when
concatenated, so for window size k we merge k consecutive pairs (Sicilian joined, English
joined), re-embed with LaBSE and measure mean cosine and the fraction clearing our 0.40
threshold. If rolling helps, poetry's numbers should climb with k, and more than prose's
(whose lines are already full sentences -> the control).

    python experiments/extraction/poetry_rolling.py
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data/external/eryk/extract-text_r06_donato/parallels/AS41-AS42_aligned_v3-dtk_scores.csv"
SENT_THRESHOLD = 0.40

import sys
sys.path.insert(0, str((REPO / "experiments/extraction").resolve()))
from analyze_standard_vs_nonstandard import GENRE


def windows(pairs: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
    """Non-overlapping merges of k consecutive pairs (Sicilian joined, English joined)."""
    out = []
    for i in range(0, len(pairs), k):
        chunk = pairs[i:i + k]
        out.append((" ".join(s for s, _ in chunk), " ".join(e for _, e in chunk)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/analysis")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    by_text: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in csv.DictReader(GOLD.open(encoding="utf-8"), delimiter="\t"):
        if r["sicilian"].strip() and r["english"].strip():
            by_text[r["file"]].append((r["sicilian"], r["english"]))

    print("loading LaBSE ...", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE")

    def score(units: list[tuple[str, str]]) -> tuple[float, float]:
        scn = model.encode([s for s, _ in units], normalize_embeddings=True, batch_size=64)
        en = model.encode([e for _, e in units], normalize_embeddings=True, batch_size=64)
        cos = (scn * en).sum(axis=1)
        return float(cos.mean()), float((cos >= SENT_THRESHOLD).mean())

    print(f"\nRolling k consecutive verses/sentences, mean LaBSE cos and recall@{SENT_THRESHOLD}:\n")
    hdr = f"{'genre':<16}" + "".join(f"{'k=' + str(k) + ' cos':>11}{'rec':>6}" for k in args.ks)
    print(hdr)
    print("-" * len(hdr))

    focus = ["poetry", "prose", "standard-prose", "folk-tale"]
    summary = {}
    for genre in focus:
        texts = [t for t in by_text if GENRE.get(t, (t, "?"))[1] == genre]
        if not texts:
            continue
        line = f"{genre:<16}"
        summary[genre] = {}
        for k in args.ks:
            units = []
            for t in texts:
                units += windows(by_text[t], k)
            cos, rec = score(units)
            summary[genre][k] = (cos, rec, len(units))
            line += f"{cos:>11.3f}{rec:>6.2f}"
        print(line)

    print(f"\n(units per genre at k=1 -> k={args.ks[-1]}):")
    for g in summary:
        print(f"  {g:<16} " + " -> ".join(str(summary[g][k][2]) for k in args.ks))

    if "poetry" in summary and "prose" in summary:
        p = summary["poetry"]
        dp = p[args.ks[-1]][1] - p[args.ks[0]][1]
        dpc = p[args.ks[-1]][0] - p[args.ks[0]][0]
        print(f"\nVerdict: rolling poetry k={args.ks[0]}->{args.ks[-1]} changes mean cos by "
              f"{dpc:+.3f} and recall by {dp:+.2f}. "
              f"{'Helps' if dp > 0.02 else 'No clear gain'} — "
              f"{'as Eryk predicted.' if dp > 0.02 else 'contrary to the suggestion on this gold.'}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "poetry_rolling.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["genre", "k", "mean_cos", "recall@0.40", "n_units"])
        for g in summary:
            for k in args.ks:
                c, r, n = summary[g][k]
                w.writerow([g, k, f"{c:.4f}", f"{r:.4f}", n])
    print(f"\nwrote {args.out / 'poetry_rolling.tsv'}")


if __name__ == "__main__":
    main()
