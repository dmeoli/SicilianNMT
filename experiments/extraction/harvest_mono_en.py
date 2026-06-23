#!/usr/bin/env python3
"""Harvest in-domain monolingual ENGLISH from the Arba Sicula PDFs, for back-translation.

Eryk Wdowiak's trick: text that does NOT align is not waste -- the unaligned English
sentences on confirmed parallel pages are clean, literary, in-domain English. We collect
them as a monolingual pool, then back-translate en->scn (with the bidirectional NLLB
adapter) to make synthetic (scn, en) pairs that augment scn->en training.

Reuses the validated parallel pipeline's page classification, confirmation and alignment
(does NOT touch build_issue.py). For every confirmed facing pair we keep the English
sentences that no kept alignment consumed, then dedup globally and against the existing
parallel English so the pool is genuinely NEW text.

    python experiments/extraction/harvest_mono_en.py \
        --issues extract-text/as-issues \
        --parallel data/processed/arbasicula/corpus.en \
        --out data/processed/arbasicula/mono_en.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from extract_pages import classify_document, parallel_pairs, load_scn_stopwords
from align_sentences import page_sentences, align, is_furniture

REPO = Path(__file__).resolve().parents[2]


def issue_mono_en(pdf: Path, model, scn_stop: set[str],
                  min_page_sim: float, min_sent_sim: float, min_chars: int) -> list[str]:
    """English sentences on confirmed pages that no kept alignment consumed."""
    pages = classify_document(pdf, scn_stop)
    doc = fitz.open(pdf)
    mono: list[str] = []
    for a, b in parallel_pairs(pages):
        scn = page_sentences(doc, a, "it")
        en = page_sentences(doc, b, "en")
        if len(scn) < 2 or len(en) < 2:
            continue
        es = model.encode(scn, normalize_embeddings=True)
        ee = model.encode(en, normalize_embeddings=True)
        sim = es @ ee.T
        page_sim = float(np.maximum(sim.max(axis=1).mean(), sim.max(axis=0).mean()))
        if page_sim < min_page_sim:
            continue
        used: set[int] = set()
        for si, sj, ei, ej, op in align(sim):
            if op in ("1-0", "0-1"):
                continue
            if float(sim[si:sj, ei:ej].mean()) < min_sent_sim:
                continue
            used.update(range(ei, ej))          # English consumed by a kept pair
        for k, s in enumerate(en):
            if k in used or is_furniture(s) or len(s) < min_chars:
                continue
            mono.append(s)
    doc.close()
    return mono


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issues", type=Path, default=REPO / "extract-text/as-issues")
    ap.add_argument("--parallel", type=Path, default=REPO / "data/processed/arbasicula/corpus.en",
                    help="existing parallel English; harvested text is deduped against it")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/arbasicula/mono_en.txt")
    ap.add_argument("--min-page-sim", type=float, default=0.50)
    ap.add_argument("--min-sent-sim", type=float, default=0.40)
    ap.add_argument("--min-chars", type=int, default=30,
                    help="drop short fragments (captions, page numbers) below this length")
    args = ap.parse_args()

    paired = set()
    if args.parallel.exists():
        paired = {ln.strip() for ln in args.parallel.read_text(encoding="utf-8").splitlines()}
    print(f"{len(paired):,} sentences already in the parallel English", flush=True)

    pdfs = sorted(args.issues.glob("*.pdf"))
    print(f"{len(pdfs)} issues found in {args.issues}", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE")
    scn_stop = load_scn_stopwords()

    seen: set[str] = set()
    pool: list[str] = []
    for pdf in pdfs:
        try:
            mono = issue_mono_en(pdf, model, scn_stop,
                                 args.min_page_sim, args.min_sent_sim, args.min_chars)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"  {pdf.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        new = 0
        for s in mono:
            if s in seen or s in paired:
                continue
            seen.add(s)
            pool.append(s)
            new += 1
        print(f"  {pdf.name}: {len(mono)} unaligned -> {new} new monolingual EN", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(pool) + "\n", encoding="utf-8")
    print(f"\nwrote {len(pool):,} monolingual English sentences to {args.out}")


if __name__ == "__main__":
    main()
