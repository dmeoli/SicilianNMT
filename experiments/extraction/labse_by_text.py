#!/usr/bin/env python3
"""Direct test of our extractor's engine: LaBSE similarity of Eryk's GOLD pairs, per text.

The standard-vs-non-standard first pass (analyze_standard_vs_nonstandard.py) used HIS
model's signals. This uses OURS: LaBSE is the embedding our aligner scores pairs with,
so the LaBSE cosine of an already-correct (hand-aligned) pair measures how confidently
our extractor could have recovered it. If non-standard texts (dialect poetry, folk
tales) have systematically lower LaBSE similarity on correct pairs, our threshold-based
aligner will under-recover them -> Eryk's hypothesis holds at the embedding level, and
per-genre thresholds / a Sicilian-adapted encoder are warranted.

Also reports, per text, the recall at our production sentence threshold (0.40): the
fraction of gold pairs whose LaBSE cosine clears it (i.e. would survive alignment).

    python experiments/extraction/labse_by_text.py            # needs gold in data/external/eryk/
"""
from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data/external/eryk/extract-text_r06_donato/parallels/AS41-AS42_aligned_v3-dtk_scores.csv"
SENT_THRESHOLD = 0.40   # our tuned production sentence-similarity threshold

# reuse the genre labels + non-standardness proxies from the first-pass analysis
import sys
sys.path.insert(0, str((REPO / "experiments/extraction").resolve()))
from analyze_standard_vs_nonstandard import GENRE, load_dieli, tokens, pearson, spearman


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=GOLD)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/analysis")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.scores.open(encoding="utf-8"), delimiter="\t")
            if r["sicilian"].strip() and r["english"].strip()]
    print(f"gold pairs: {len(rows)}   loading LaBSE ...", flush=True)

    model = SentenceTransformer("sentence-transformers/LaBSE")
    scn = model.encode([r["sicilian"] for r in rows], normalize_embeddings=True,
                       batch_size=64, show_progress_bar=True)
    en = model.encode([r["english"] for r in rows], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=True)
    cos = (scn * en).sum(axis=1)   # cosine of each true pair

    dieli = load_dieli()
    agg: dict[str, dict] = defaultdict(lambda: {"cos": [], "oov": [], "clear": 0, "n": 0})
    for r, c in zip(rows, cos):
        toks = tokens(r["sicilian"])
        a = agg[r["file"]]
        a["n"] += 1
        a["cos"].append(float(c))
        a["oov"].append(sum(t not in dieli for t in toks) / len(toks) if toks else 0.0)
        a["clear"] += int(c >= SENT_THRESHOLD)

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    table = []
    for t, a in agg.items():
        title, genre = GENRE.get(t, (t, "?"))
        table.append({"id": t, "title": title, "genre": genre, "n": a["n"],
                      "oov": mean(a["oov"]), "labse_cos": mean(a["cos"]),
                      "recall@0.40": a["clear"] / a["n"]})
    table.sort(key=lambda d: d["labse_cos"])   # hardest (lowest LaBSE) first

    hdr = f"{'text':<42}{'genre':<15}{'n':>4}{'oov':>7}{'LaBSE':>8}{'rec@.40':>9}"
    print(f"\nPer-text LaBSE similarity of gold pairs (lowest = hardest to align first)\n")
    print(hdr)
    print("-" * len(hdr))
    for d in table:
        print(f"{d['title'][:41]:<42}{d['genre']:<15}{d['n']:>4}{d['oov']:>7.2f}"
              f"{d['labse_cos']:>8.3f}{d['recall@0.40']:>9.2f}")

    oov = [d["oov"] for d in table]
    lab = [d["labse_cos"] for d in table]
    print("\nCorrelation across the 18 texts:")
    print(f"  oov_dieli vs LaBSE cos:  Pearson {pearson(oov, lab):+.2f}  Spearman {spearman(oov, lab):+.2f}")
    print(f"\nPair-level ({len(rows):,} pairs)  oov vs LaBSE cos:  "
          f"Pearson {pearson([sum(t not in dieli for t in tokens(r['sicilian'])) / max(len(tokens(r['sicilian'])), 1) for r in rows], list(map(float, cos))):+.2f}")

    print("\nMean LaBSE cos / recall@0.40 by genre:")
    by_g: dict[str, list] = defaultdict(list)
    for d in table:
        by_g[d["genre"]].append((d["labse_cos"], d["recall@0.40"]))
    for g, vs in sorted(by_g.items(), key=lambda kv: mean([x[0] for x in kv[1]])):
        print(f"  {g:<15} LaBSE {mean([x[0] for x in vs]):.3f}   "
              f"recall {mean([x[1] for x in vs]):.2f}   ({len(vs)} text{'s' if len(vs) > 1 else ''})")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "labse_by_text.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), delimiter="\t")
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
