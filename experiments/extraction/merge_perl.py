#!/usr/bin/env python3
"""Merge the legacy Perl/hunalign extraction into our Arba Sicula corpus (union).

For the issues the original pipeline processed (as27-31) we have hunalign output
(extract-text/aligned/asNN_ha.csv). This adds the Perl pairs that our LaBSE pipeline
did NOT already produce (deduped on normalized Sicilian), enriching those issues.
Final cleaning/splitting is left to experiments/dataset/assemble.py.

    python experiments/extraction/merge_perl.py    # rewrites data/processed/arbasicula/corpus.tsv
"""
from __future__ import annotations
import csv
import re
import unicodedata
from pathlib import Path

csv.field_size_limit(10 ** 7)  # some extracted "sentences" (junk pages) are very long

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/processed/arbasicula/corpus.tsv"
ALIGNED = REPO / "extract-text/aligned"
ISSUES = ["as27", "as28", "as29", "as30", "as31"]
MIN_SCORE = 0.5


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    rows = list(csv.reader(CORPUS.open(encoding="utf-8"), delimiter="\t"))
    header, ours = rows[0], rows[1:]
    seen = {key(r[1]) for r in ours if len(r) >= 2}
    print(f"our corpus: {len(ours)} pairs")

    added = 0
    for issue in ISSUES:
        f = ALIGNED / f"{issue}_ha.csv"
        if not f.exists():
            continue
        for r in csv.reader(f.open(encoding="utf-8"), delimiter="\t"):
            if len(r) < 2 or not r[0].strip() or not r[1].strip():
                continue
            try:
                score = float(r[2]) if len(r) > 2 and r[2] else 0.0
            except ValueError:
                score = 0.0
            scn, en = r[0].strip(), r[1].strip()
            if score < MIN_SCORE or len(scn.split()) < 3 or len(en.split()) < 3:
                continue
            k = key(scn)
            if k in seen:
                continue
            seen.add(k)
            ours.append([issue + "_perl", scn, en])
            added += 1

    with CORPUS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(ours)
    print(f"added {added} Perl-unique pairs (as27-31, score>{MIN_SCORE}) -> {len(ours)} total")
    print(f"wrote {CORPUS}")


if __name__ == "__main__":
    main()
