#!/usr/bin/env python3
"""Quality inspection of the Napizia 'Good-Sicilian-in-NLLB' 50k curated subset.

Reports score/length distributions, duplicates and noise flags so we can choose
our own cut threshold before mixing this source into the training corpus.

    python experiments/dataset/inspect_50k.py [path_to_50k.csv.gz]
"""
from __future__ import annotations
import csv
import gzip
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT = (REPO / "data/external/Good-Sicilian-in-NLLB/Napizia-scored-NLLB.en-scn/"
           "Napizia-scored-NLLB-en-scn_50k-w-scores.csv.gz")
SCN_STOP = {w.strip().lower() for w in (REPO / "vocab/stopwords_scn.txt").read_text().splitlines()
            if w.strip() and not w.startswith("#")}


def main(path: Path) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw_lines = sum(1 for _ in f) - 1
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", engine="python",
                         quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if len(df) < raw_lines:
        print(f"(skipped {raw_lines - len(df):,} malformed lines of {raw_lines:,})\n")
    en, sc = df["english"].astype(str), df["sicilian"].astype(str)
    enw = en.str.split().map(len)
    scw = sc.str.split().map(len)
    ratio = scw / enw.clip(lower=1)

    def q(s):
        return " ".join(f"{p}%={s.quantile(p/100):.3f}" for p in (5, 25, 50, 75, 95))

    print(f"== {path.name}: {len(df):,} rows ==\n")
    print("SCORES")
    print("  napizia (low=better):", q(df["napizia_score"]),
          f"| min={df.napizia_score.min():.3f} max={df.napizia_score.max():.3f}")
    print("  nllb    (high=better):", q(df["nllb_score"]))
    print("\nLENGTHS (words)")
    print("  english:", q(enw.astype(float)), f"| max={enw.max()}")
    print("  sicilian:", q(scw.astype(float)), f"| max={scw.max()}")
    print("  ratio scn/en:", q(ratio))

    print("\nDUPLICATES / NOISE")
    dup_pair = df.duplicated(subset=["english", "sicilian"]).sum()
    print(f"  exact dup (en,scn) : {dup_pair:,}  ({dup_pair/len(df):.1%})")
    print(f"  dup english        : {df.duplicated('english').sum():,}")
    print(f"  dup sicilian       : {df.duplicated('sicilian').sum():,}")
    print(f"  en == scn (copy)   : {(en.str.strip()==sc.str.strip()).sum():,}")
    short = (enw < 3) | (scw < 3)
    print(f"  very short (<3 wd) : {short.sum():,}  ({short.mean():.1%})")
    paren_num = sc.str.contains(r"\(\d+\)$", regex=True)
    print(f"  trailing '(N)' frag: {paren_num.sum():,}  ({paren_num.mean():.1%})")
    skew = (ratio > 3) | (ratio < 1/3)
    print(f"  length ratio skew  : {skew.sum():,}  ({skew.mean():.1%})")
    no_scn_stop = ~sc.str.lower().apply(lambda t: any(w in SCN_STOP for w in t.split()))
    long_enough = scw >= 5
    susp = no_scn_stop & long_enough
    print(f"  scn>=5wd w/o any SCN stopword (LID smell): {susp.sum():,}  ({susp.mean():.1%})")

    def show(title, frame):
        print(f"\n{title}")
        for _, r in frame.iterrows():
            print(f"  [{r.napizia_score:.3f}] {r.english[:70]!r}  ->  {r.sicilian[:70]!r}")

    show("BEST 6 (lowest napizia score):", df.nsmallest(6, "napizia_score"))
    show("WORST 6 in the 50k (highest napizia score):", df.nlargest(6, "napizia_score"))
    show("RANDOM 6 (seed=0):", df.sample(6, random_state=0))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
