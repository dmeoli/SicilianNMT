#!/usr/bin/env python3
"""Build a parallel corpus from one Arba Sicula issue, end to end (CPU).

Pipeline: PyMuPDF extract -> page-language classify -> candidate (SC, EN) pages
adjacent on either side -> LaBSE page-level confirmation, each SC page keeping its
best-matching facing page (drops cover/index/non-parallel and the off-by-one pairs)
-> consecutive confirmed spreads grouped into blocks -> LaBSE sentence alignment per
block, restricted to facing pages +-1 -> filtered scn-en sentence pairs.

Aligning a block instead of each spread alone keeps the sentences that run over a
page break (they were cut in two and lost), while the facing-page band stops the
aligner from matching text of two different articles (E. Wdowiak, 2026-09).

    python experiments/extraction/build_issue.py extract-text/as-issues/as46.pdf \
        --out data/processed/as46

Reuses extract_pages.py and align_sentences.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from extract_pages import (classify_document, candidate_pairs, load_scn_stopwords,
                           page_label, printed_page_numbers)
from align_sentences import page_sentences, block_sentences, spread_band, align


def process_issue(pdf: Path, model, scn_stop: set[str],
                  min_page_sim: float = 0.50, min_sent_sim: float = 0.40,
                  report: list | None = None):
    """Return (out_scn, out_en, n_candidates, n_confirmed) for one issue PDF.

    If `report` is a list, one (scn_page, en_page, scn_index, en_index, page_sim)
    row per confirmed spread is appended to it (pages as printed numbers).
    """
    pages = classify_document(pdf, scn_stop)
    candidates = candidate_pairs(pages)
    doc = fitz.open(pdf)
    vec: dict[str, np.ndarray] = {}

    def embed(sents: list[str]) -> np.ndarray:
        new = list(dict.fromkeys(x for x in sents if x not in vec))
        if new:
            vec.update(zip(new, model.encode(new, normalize_embeddings=True)))
        return np.stack([vec[x] for x in sents])

    sents = {}
    scored = []
    for a, b in candidates:
        for i, lang in ((a, "it"), (b, "en")):
            if i not in sents:
                sents[i] = page_sentences(doc, i, lang)
        scn, en = sents[a], sents[b]
        if len(scn) < 2 or len(en) < 2:
            continue
        sim = embed(scn) @ embed(en).T
        # page-level confirmation: average of each side's best match
        page_sim = float(np.maximum(sim.max(axis=1).mean(), sim.max(axis=0).mean()))
        if page_sim >= min_page_sim:
            scored.append((page_sim, a, b))

    # every page in at most one spread, the most similar candidate first
    used: set[int] = set()
    spreads = []
    for page_sim, a, b in sorted(scored, reverse=True):
        if a not in used and b not in used:
            used.update((a, b))
            spreads.append((a, b, page_sim))
    spreads.sort()
    if report is not None:
        printed = printed_page_numbers(pdf)
        report.extend((page_label(printed, a), page_label(printed, b), a, b, ps)
                      for a, b, ps in spreads)

    blocks: list[list[tuple[int, int]]] = []
    for a, b, _ in spreads:
        if blocks and (a, b) == (blocks[-1][-1][0] + 2, blocks[-1][-1][1] + 2):
            blocks[-1].append((a, b))
        else:
            blocks.append([(a, b)])

    out_scn: list[str] = []
    out_en: list[str] = []
    for block in blocks:
        scn, scn_owner = block_sentences(doc, [a for a, _ in block], "it")
        en, en_owner = block_sentences(doc, [b for _, b in block], "en")
        if not scn or not en:
            continue
        sim = embed(scn) @ embed(en).T
        for si, sj, ei, ej, op in align(sim, band=spread_band(scn_owner, en_owner)):
            if op in ("1-0", "0-1"):
                continue
            score = float(sim[si:sj, ei:ej].mean())
            if score < min_sent_sim:
                continue
            out_scn.append(" ".join(scn[si:sj]))
            out_en.append(" ".join(en[ei:ej]))
    doc.close()
    return out_scn, out_en, len(candidates), len(spreads)


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
    spreads: list = []
    out_scn, out_en, n_cand, confirmed = process_issue(
        args.pdf, model, load_scn_stopwords(), args.min_page_sim, args.min_sent_sim,
        report=spreads)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "as.scn").write_text("\n".join(out_scn) + "\n", encoding="utf-8")
    (args.out / "as.en").write_text("\n".join(out_en) + "\n", encoding="utf-8")
    with open(args.out / "as.tsv", "w", encoding="utf-8") as f:
        for s, e in zip(out_scn, out_en):
            f.write(f"{s}\t{e}\n")
    with open(args.out / "pairs.tsv", "w", encoding="utf-8") as f:
        f.write("scn_page\ten_page\tscn_pdf_index\ten_pdf_index\tpage_sim\n")
        for sp, ep, a, b, ps in spreads:
            f.write(f"{sp}\t{ep}\t{a}\t{b}\t{ps:.3f}\n")

    print(f"{args.pdf.name}: candidate pairs {n_cand} -> "
          f"confirmed {confirmed} (page-sim>={args.min_page_sim}) -> "
          f"{len(out_scn)} aligned sentence pairs (sent-sim>={args.min_sent_sim})")
    print(f"wrote as.scn / as.en / as.tsv / pairs.tsv to {args.out}/")


if __name__ == "__main__":
    main()
