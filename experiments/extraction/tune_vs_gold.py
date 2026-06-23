#!/usr/bin/env python3
"""Tune the extraction/alignment thresholds against Eryk's hand-aligned AS41-42 gold.

Eryk aligned AS41-42 by hand (1,378 pairs). We use that as a benchmark: embed each
issue's candidate page-pairs ONCE, then sweep (page-sim, sent-sim) thresholds and
measure precision/recall of our automatic pairs against the gold (exact normalized
Sicilian-sentence match). Picks the threshold that best recovers his manual work.

    python experiments/extraction/tune_vs_gold.py
"""
from __future__ import annotations
import re
import sys
import unicodedata
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pages import classify_document, parallel_pairs, load_scn_stopwords
from align_sentences import page_sentences, align

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data/external/eryk/extract-text_r06_donato/parallels/AS41-AS42_aligned_v0-raw.sc"
ISSUES = ["as41", "as42"]


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    goldkeys = {key(s) for s in GOLD.read_text(encoding="utf-8").splitlines() if s.strip()}
    print(f"gold pairs: {len(goldkeys)}", flush=True)

    model = SentenceTransformer("sentence-transformers/LaBSE")
    scn_stop = load_scn_stopwords()

    # embed every candidate facing-pair ONCE, cache (scn, en, sim, page_sim)
    cache = []
    for issue in ISSUES:
        pdf = REPO / f"extract-text/as-issues/{issue}.pdf"
        pages = classify_document(pdf, scn_stop)
        doc = fitz.open(pdf)
        for a, b in parallel_pairs(pages):
            scn, en = page_sentences(doc, a, "it"), page_sentences(doc, b, "en")
            if len(scn) < 2 or len(en) < 2:
                continue
            es = model.encode(scn, normalize_embeddings=True)
            ee = model.encode(en, normalize_embeddings=True)
            sim = es @ ee.T
            ps = float(np.maximum(sim.max(axis=1).mean(), sim.max(axis=0).mean()))
            cache.append((scn, en, sim, ps))
        doc.close()
    print(f"cached candidate pairs: {len(cache)}\n", flush=True)

    print(f"{'page>=':>7}{'sent>=':>7}{'pairs':>7}{'match':>7}{'prec':>7}{'rec':>7}{'F1':>7}")
    for mps in (0.62, 0.58, 0.55, 0.50):
        for mss in (0.50, 0.45, 0.40, 0.35):
            ours = []
            for scn, en, sim, ps in cache:
                if ps < mps:
                    continue
                for si, sj, ei, ej, op in align(sim):
                    if op in ("1-0", "0-1"):
                        continue
                    if float(sim[si:sj, ei:ej].mean()) < mss:
                        continue
                    ours.append(" ".join(scn[si:sj]))
            ok = {key(s) for s in ours}
            inter = ok & goldkeys
            prec = len(inter) / max(len(ok), 1)
            rec = len(inter) / max(len(goldkeys), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"{mps:>7.2f}{mss:>7.2f}{len(ours):>7}{len(inter):>7}"
                  f"{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
