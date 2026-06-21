#!/usr/bin/env python3
"""Build a clean Sicilian-English subset from Napizia's scored NLLB data.

Stages (each reported): napizia-score threshold -> Corsican filter -> exact-pair
dedup -> Sicilian-side dedup (keep best score). Reads the full scored CSV so the
threshold captures every eligible pair (not just the published top-50k).

    python experiments/dataset/build_clean_nllb.py --threshold 2.0 \
        --out data/processed/nllb_clean

Writes: pairs.tsv (en, scn, napizia), train.scn, train.en.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "data/external/Good-Sicilian-in-NLLB/Napizia-scored-NLLB-en-scn_all-w-scores.csv"

# Strong Corsican (co) markers absent from standard literary Sicilian.
# scn uses è/havi/chi/supra/ô where co uses hè/hà/chì/nant'/à u; "ghj" is a co digraph.
CORSICAN = re.compile(
    r"\b(h[èe]|hà|chì|gh?jè|ghjornu|inde|nantu?\s*à|à\s+[uia]\b|hanu|sò|ghjunghje)\b|ghj",
    re.IGNORECASE,
)


def is_corsican(text: str) -> bool:
    return bool(CORSICAN.search(text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="keep pairs with napizia score < threshold (low=better)")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/nllb_clean")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sicilianu": "scn", "english": "en", "napizia": "score"})
    df = df[["en", "scn", "score"]].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["en", "scn", "score"])
    n0 = len(df)

    df = df[df["score"] < args.threshold]
    n1 = len(df)

    df = df[~df["scn"].astype(str).map(is_corsican)]
    n2 = len(df)

    df = df.drop_duplicates(subset=["en", "scn"])
    n3 = len(df)

    df = df.sort_values("score").drop_duplicates(subset=["scn"], keep="first")
    n4 = len(df)

    print(f"loaded valid rows          : {n0:,}")
    print(f"after napizia < {args.threshold:<10}: {n1:,}  (-{n0-n1:,})")
    print(f"after Corsican filter      : {n2:,}  (-{n1-n2:,})")
    print(f"after exact-pair dedup     : {n3:,}  (-{n2-n3:,})")
    print(f"after Sicilian-side dedup  : {n4:,}  (-{n3-n4:,})   <-- FINAL")

    enw = df["en"].str.split().map(len)
    scw = df["scn"].str.split().map(len)
    print(f"\nfinal lengths (words): en median {enw.median():.0f} / scn median {scw.median():.0f}"
          f" | <3 words: {((enw<3)|(scw<3)).mean():.1%}")
    print(f"score range kept: {df.score.min():.3f} .. {df.score.max():.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "pairs.tsv", sep="\t", index=False)
    df["scn"].to_csv(args.out / "train.scn", index=False, header=False)
    df["en"].to_csv(args.out / "train.en", index=False, header=False)
    print(f"\nwrote pairs.tsv, train.scn, train.en to {args.out}/")


if __name__ == "__main__":
    main()
