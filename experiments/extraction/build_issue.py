#!/usr/bin/env python3
"""Build a parallel corpus from one Arba Sicula issue, end to end (CPU).

Pipeline: PyMuPDF extract -> page-language classify -> facing SC/EN candidate
pairs -> LaBSE page-level confirmation (drops cover/index/non-parallel) ->
LaBSE sentence alignment per confirmed pair -> filtered scn-en sentence pairs.

    python experiments/extraction/build_issue.py extract-text/as-issues/as30.pdf \
        --out data/processed/as30

Reuses extract_pages.py and align_sentences.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from extract_pages import classify_document, parallel_pairs, load_scn_stopwords
from align_sentences import page_sentences, align


def process_issue(pdf: Path, model, scn_stop: set[str],
                  min_page_sim: float = 0.50, min_sent_sim: float = 0.40):
    """Return (out_scn, out_en, n_candidates, n_confirmed) for one issue PDF."""
    pages = classify_document(pdf, scn_stop)
    candidates = parallel_pairs(pages)
    doc = fitz.open(pdf)
    out_scn: list[str] = []
    out_en: list[str] = []
    confirmed = 0
    for a, b in candidates:
        scn = page_sentences(doc, a, "it")
        en = page_sentences(doc, b, "en")
        if len(scn) < 2 or len(en) < 2:
            continue
        es = model.encode(scn, normalize_embeddings=True)
        ee = model.encode(en, normalize_embeddings=True)
        sim = es @ ee.T
        # page-level confirmation: average of each side's best match
        page_sim = float(np.maximum(sim.max(axis=1).mean(), sim.max(axis=0).mean()))
        if page_sim < min_page_sim:
            continue
        confirmed += 1
        for si, sj, ei, ej, op in align(sim):
            if op in ("1-0", "0-1"):
                continue
            score = float(sim[si:sj, ei:ej].mean())
            if score < min_sent_sim:
                continue
            out_scn.append(" ".join(scn[si:sj]))
            out_en.append(" ".join(en[ei:ej]))
    doc.close()
    return out_scn, out_en, len(candidates), confirmed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-page-sim", type=float, default=0.50,
                    help="confirm a facing pair only if mean page cosine >= this")
    ap.add_argument("--min-sent-sim", type=float, default=0.40)
    args = ap.parse_args()

    model = SentenceTransformer("sentence-transformers/LaBSE")
    out_scn, out_en, n_cand, confirmed = process_issue(
        args.pdf, model, load_scn_stopwords(), args.min_page_sim, args.min_sent_sim)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "as.scn").write_text("\n".join(out_scn) + "\n", encoding="utf-8")
    (args.out / "as.en").write_text("\n".join(out_en) + "\n", encoding="utf-8")
    with open(args.out / "as.tsv", "w", encoding="utf-8") as f:
        for s, e in zip(out_scn, out_en):
            f.write(f"{s}\t{e}\n")

    print(f"{args.pdf.name}: candidate pairs {n_cand} -> "
          f"confirmed {confirmed} (page-sim>={args.min_page_sim}) -> "
          f"{len(out_scn)} aligned sentence pairs (sent-sim>={args.min_sent_sim})")
    print(f"wrote as.scn / as.en / as.tsv to {args.out}/")


if __name__ == "__main__":
    main()
