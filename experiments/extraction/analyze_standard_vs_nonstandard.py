#!/usr/bin/env python3
"""Does our extractor handle NON-standard Sicilian worse than standard Sicilian?

Eryk Wdowiak's hypothesis (2026-08): comparison with his hand-alignments should
favour standard Sicilian and, especially, Prof. Cipolla's texts, while the dialect
poetry and folk tales in Arba Sicula (non-standard Sicilian) may be harder.

We test it on his hand-aligned AS41-42 gold (18 texts, 1,373 pairs). For each text
we measure two OBJECTIVE non-standardness proxies from the Sicilian side and relate
them to two alignment-difficulty signals he already computed:

  non-standardness proxies (higher = less standard):
    oov_dieli   fraction of Sicilian tokens absent from the Dieli standard lexicon
    norm_div    fraction of tokens changed by std orthographic normalization

  difficulty signals (from his pipeline, higher = easier):
    se_score    his Sockeye Sicilian model's score for the pair (sockeye-score)
    ha_score    hunalign alignment confidence

If non-standard texts show systematically lower se_score / ha_score, the hypothesis
holds and per-genre corrective measures are warranted. Pure stdlib (no ML deps).

    python experiments/extraction/analyze_standard_vs_nonstandard.py
"""
from __future__ import annotations
import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data/external/eryk/extract-text_r06_donato/parallels/AS41-AS42_aligned_v3-dtk_scores.csv"
DIELI = REPO / "vocab/dieli-scen-dict.txt"

# 18 hand-aligned texts labelled from the AS41/42 tables of contents (title — author, genre).
GENRE = {
    "as41-p036": ("Puleo (article)", "prose"),
    "as41-p042": ("Nuttata di cauru — Santiaco", "poetry"),
    "as41-p048": ("Lu Don Chisciotti Sicilianu — G. Cipolla", "standard-prose"),
    "as41-p066": ("Lu Pani — G. Basile", "prose"),
    "as41-p096": ("Sicilian Artist of Universal Vision — Fusco", "prose"),
    "as41-p116": ("Benedetta Lino", "prose"),
    "as41-p126": ("review (Cocuzza / Buscemi)", "prose"),
    "as42-p034": ("Carlo Puleo", "prose"),
    "as42-p040": ("Lu ròggiu di lu varveri — G. Pitrè", "folk-tale"),
    "as42-p046": ("A so disposizioni — M. Scalabrino", "poetry"),
    "as42-p056": ("U Cannitu — A. Di Pietro", "poetry"),
    "as42-p062": ("I borghi chiù beddi di la Sicilia", "prose"),
    "as42-p074": ("The Story of Antonino Sciascia", "prose"),
    "as42-p080": ("Dedalu — R. Tripodi", "poetry"),
    "as42-p122": ("Giuseppe Sciacca", "prose"),
    "as42-p130": ("The Story of Catallu Valenti", "prose"),
    "as42-p136": ("Sasizza, pipi e cipudda — B. Lino", "prose"),
    "as42-p140": ("Ricenzioni — N. Provenzano", "prose"),
}

_WORD = re.compile(r"[a-zàèéìòùáíóúäöüâêîôûçñ'’]+")


def tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower().replace("’", "'"))


def load_dieli() -> set[str]:
    """Standard Sicilian word set: the RHS ('english @ sicilian phrase') of the Dieli dict."""
    lex: set[str] = set()
    for line in DIELI.read_text(encoding="utf-8", errors="replace").splitlines():
        if "@" in line:
            lex.update(tokens(line.split("@", 1)[1]))
    return lex


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else float("nan")


def _ranks(vs: list[float]) -> list[float]:
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    rank = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    return rank


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=GOLD)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/analysis")
    args = ap.parse_args()

    if not args.scores.exists():
        sys.exit(f"gold scores not found: {args.scores}\n"
                 f"copy Eryk's extract-text_r06_donato into data/external/eryk/ first.")

    dieli = load_dieli()
    print(f"Dieli standard lexicon: {len(dieli):,} word forms\n")

    rows = list(csv.DictReader(args.scores.open(encoding="utf-8"), delimiter="\t"))
    # per-text accumulators
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "oov": [], "nd": [], "se": [],
                                                "ha": [], "w": []})
    pair_oov, pair_se = [], []      # pair-level, for a higher-powered correlation
    dropped_inf = 0
    for r in rows:
        toks = tokens(r["sicilian"])
        if not toks:
            continue
        oov = sum(t not in dieli for t in toks) / len(toks)
        nd = norm_divergence(r["sicilian"])
        a = agg[r["file"]]
        a["n"] += 1
        a["oov"].append(oov)
        a["nd"].append(nd)
        a["w"].append(len(toks))
        try:
            se = float(r["se_score"])
        except ValueError:
            se = float("inf")
        if math.isfinite(se):
            a["se"].append(se)
            pair_oov.append(oov)
            pair_se.append(se)
        else:
            dropped_inf += 1
        try:
            a["ha"].append(float(r["score"]))
        except ValueError:
            pass

    def mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else float("nan")

    texts = sorted(agg)
    table = []
    for t in texts:
        a = agg[t]
        title, genre = GENRE.get(t, (t, "?"))
        table.append({"id": t, "title": title, "genre": genre, "n": a["n"],
                      "oov": mean(a["oov"]), "norm_div": mean(a["nd"]),
                      "words": mean(a["w"]), "se_score": mean(a["se"]),
                      "ha_score": mean(a["ha"])})

    table.sort(key=lambda d: d["oov"], reverse=True)
    print(f"Per-text (sorted by OOV, most non-standard first). {dropped_inf} pairs "
          f"dropped for non-finite se_score.\n")
    hdr = f"{'text':<42}{'genre':<15}{'n':>4}{'oov':>7}{'ndiv':>7}{'wrds':>6}{'se':>7}{'ha':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d in table:
        print(f"{d['title'][:41]:<42}{d['genre']:<15}{d['n']:>4}{d['oov']:>7.2f}"
              f"{d['norm_div']:>7.2f}{d['words']:>6.1f}{d['se_score']:>7.2f}{d['ha_score']:>7.2f}")

    # correlations across the 18 texts
    oov = [d["oov"] for d in table]
    nd = [d["norm_div"] for d in table]
    se = [d["se_score"] for d in table]
    ha = [d["ha_score"] for d in table]
    print("\nCorrelations across the 18 texts (negative = non-standard -> harder):")
    for name, xs in (("oov_dieli", oov), ("norm_div", nd)):
        print(f"  {name:<10} vs se_score:  Pearson {pearson(xs, se):+.2f}  Spearman {spearman(xs, se):+.2f}")
        print(f"  {name:<10} vs ha_score:  Pearson {pearson(xs, ha):+.2f}  Spearman {spearman(xs, ha):+.2f}")
    print(f"\nPair-level ({len(pair_oov):,} pairs)  oov_dieli vs se_score:  "
          f"Pearson {pearson(pair_oov, pair_se):+.2f}  Spearman {spearman(pair_oov, pair_se):+.2f}")

    # genre means
    print("\nMean se_score by genre:")
    by_g: dict[str, list[float]] = defaultdict(list)
    for d in table:
        if math.isfinite(d["se_score"]):
            by_g[d["genre"]].append(d["se_score"])
    for g, vs in sorted(by_g.items(), key=lambda kv: mean(kv[1])):
        print(f"  {g:<15} {mean(vs):6.2f}   ({len(vs)} text{'s' if len(vs) > 1 else ''})")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "standard_vs_nonstandard.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), delimiter="\t")
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    # normalizer lives in the dataset experiments; import lazily so --help works anywhere
    sys.path.insert(0, str((REPO / "experiments/dataset").resolve()))
    from normalize_scn import normalize

    def norm_divergence(s: str) -> float:
        a = tokens(s)
        b = tokens(normalize(s, "std"))
        if not a:
            return 0.0
        m = min(len(a), len(b))
        changed = sum(a[i] != b[i] for i in range(m)) + abs(len(a) - len(b))
        return changed / len(a)

    main()
