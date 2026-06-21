#!/usr/bin/env python3
"""Embedding-based sentence aligner (LaBSE + DP), a Vecalign-style prototype.

Given a Sicilian page and its facing English page of an Arba Sicula issue, it
segments both into sentences, embeds them with LaBSE and finds the monotonic
alignment (1-1, 1-2, 2-1, plus skips) maximizing cross-lingual similarity.
Replaces the legacy hunalign + Dieli-dictionary step.

    python experiments/extraction/align_sentences.py extract-text/as-issues/as30.pdf 42 43
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pysbd
from sentence_transformers import SentenceTransformer

HEADER_RE = re.compile(r"^\s*(arba sicula\b.*|\d{1,3})\s*$", re.IGNORECASE)

_SEGMENTERS: dict[str, pysbd.Segmenter] = {}


def _segmenter(lang: str) -> pysbd.Segmenter:
    if lang not in _SEGMENTERS:
        _SEGMENTERS[lang] = pysbd.Segmenter(language=lang, clean=False)
    return _SEGMENTERS[lang]


def is_furniture(s: str) -> bool:
    """Drop page furniture: mastheads, all-caps headers, stray tokens."""
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3 or len(s.split()) < 2:
        return True
    return sum(c.isupper() for c in letters) / len(letters) > 0.6


def page_sentences(doc: fitz.Document, idx: int, lang: str = "it") -> list[str]:
    """Sentence-segment a page's text (pysbd; lang 'it' for Sicilian, 'en' for English)."""
    lines = [ln for ln in doc[idx].get_text("text").splitlines() if not HEADER_RE.match(ln)]
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return [s.strip() for s in _segmenter(lang).segment(text)
            if len(s.strip()) > 1 and not is_furniture(s)]


def align(sim: np.ndarray, null_pen: float = 0.5):
    """Monotonic DP over a (n x m) cosine-similarity matrix. Returns aligned spans."""
    n, m = sim.shape
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG)
    bk: dict = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i, j]
            if cur == NEG:
                continue
            moves = []
            if i < n and j < m:
                moves.append((i + 1, j + 1, sim[i, j], "1-1"))
            if i < n and j + 1 < m:
                moves.append((i + 1, j + 2, (sim[i, j] + sim[i, j + 1]) / 2, "1-2"))
            if i + 1 < n and j < m:
                moves.append((i + 2, j + 1, (sim[i, j] + sim[i + 1, j]) / 2, "2-1"))
            if i < n:
                moves.append((i + 1, j, -null_pen, "1-0"))
            if j < m:
                moves.append((i, j + 1, -null_pen, "0-1"))
            for ni, nj, gain, op in moves:
                if cur + gain > dp[ni, nj]:
                    dp[ni, nj] = cur + gain
                    bk[(ni, nj)] = (i, j, op)
    path = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = bk[(i, j)]
        path.append((pi, i, pj, j, op))
        i, j = pi, pj
    return list(reversed(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("scn_page", type=int, help="0-indexed Sicilian page")
    ap.add_argument("en_page", type=int, help="0-indexed English page")
    ap.add_argument("--min-sim", type=float, default=0.5,
                    help="drop 1-1 alignments below this cosine (likely non-parallel)")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    scn = page_sentences(doc, args.scn_page, "it")
    en = page_sentences(doc, args.en_page, "en")
    print(f"scn sentences: {len(scn)} | en sentences: {len(en)}\n")

    model = SentenceTransformer("sentence-transformers/LaBSE")
    es = model.encode(scn, normalize_embeddings=True)
    ee = model.encode(en, normalize_embeddings=True)
    sim = es @ ee.T

    kept = dropped = 0
    for si, sj, ei, ej, op in align(sim):
        if op in ("1-0", "0-1"):
            continue
        s = " ".join(scn[si:sj])
        e = " ".join(en[ei:ej])
        score = float(sim[si:sj, ei:ej].mean())
        if op == "1-1" and score < args.min_sim:
            dropped += 1
            continue
        kept += 1
        print(f"[{op} {score:.2f}] {s[:75]}  ||  {e[:75]}")
    print(f"\nkept {kept} aligned pairs (dropped {dropped} below min-sim)")


if __name__ == "__main__":
    main()
