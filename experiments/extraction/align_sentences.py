#!/usr/bin/env python3
"""Embedding-based sentence aligner (LaBSE + DP), a Vecalign-style prototype.

Given a Sicilian page and its facing English page of an Arba Sicula issue, it
segments both into sentences, embeds them with LaBSE and finds the monotonic
alignment (1-1, 1-2, 2-1, plus skips) maximizing cross-lingual similarity.
Replaces the legacy hunalign + Dieli-dictionary step.

Pages are given as PRINTED page numbers (the ones on the paper copy); pass --index
to give 0-based pdf page indices instead.

    python experiments/extraction/align_sentences.py extract-text/as-issues/as46.pdf 60 61
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pysbd
from sentence_transformers import SentenceTransformer

from extract_pages import page_text, printed_page_numbers

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


_SOFT_HYPHEN_RE = re.compile(r"(\w)\xad\s*")
_EOL_HYPHEN_RE = re.compile(r"(\w)-[ \t]*\n(?=[a-zàèéìòùâêîôû])")


def _page_body(doc, idx: int) -> str:
    """Page text without furniture, lines joined and end-of-line hyphenation undone."""
    lines = [ln for ln in page_text(doc[idx]).splitlines() if not HEADER_RE.match(ln)]
    text = _SOFT_HYPHEN_RE.sub(r"\1", "\n".join(lines))
    text = _EOL_HYPHEN_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _segment(text: str, lang: str) -> list[str]:
    return [s.strip() for s in _segmenter(lang).segment(text)
            if len(s.strip()) > 1 and not is_furniture(s)]


def page_sentences(doc: fitz.Document, idx: int, lang: str = "it") -> list[str]:
    """Sentence-segment a page's text (pysbd; lang 'it' for Sicilian, 'en' for English)."""
    return _segment(_page_body(doc, idx), lang)


_OPEN_END_RE = re.compile(r"[^.!?;:…»\"”’')\]]$")
_LOWER_START_RE = re.compile(r"^[a-zàèéìòùâêîôû]")


def block_sentences(doc: fitz.Document, indices: list[int],
                    lang: str = "it") -> tuple[list[str], list[int]]:
    """Sentences of consecutive pages of one side, mended across page breaks.

    Each page is segmented on its own (the page break is a useful hard boundary:
    segmenting the whole block at once fuses unpunctuated verse into sentences that
    span pages). Then the last sentence of a page is joined to the first of the next
    one when it is visibly cut: no closing punctuation, and the continuation starts
    lowercase. Returns the sentences and, for each, the position in `indices` of the
    page where it starts.
    """
    sents: list[str] = []
    owner: list[int] = []
    for k, idx in enumerate(indices):
        page = page_sentences(doc, idx, lang)
        if (page and sents and owner[-1] == k - 1 and _OPEN_END_RE.search(sents[-1])
                and _LOWER_START_RE.match(page[0])):
            sents[-1] = f"{sents[-1]} {page[0]}"
            page = page[1:]
        sents.extend(page)
        owner.extend([k] * len(page))
    return sents, owner


def spread_band(scn_owner: list[int], en_owner: list[int], slack: int = 1):
    """Per DP row i, the allowed column range [lo, hi] (facing pages +- `slack`)."""
    n, m = len(scn_owner), len(en_owner)
    first = {}
    last = {}
    for j, o in enumerate(en_owner):
        first.setdefault(o, j)
        last[o] = j + 1
    top = max(en_owner, default=0)
    lo, hi = [], []
    for i in range(n + 1):
        o = scn_owner[min(i, n - 1)] if n else 0
        a = min((first[k] for k in range(max(0, o - slack), top + 1) if k in first), default=m)
        b = max((last[k] for k in range(0, o + slack + 1) if k in last), default=0)
        lo.append(a)
        hi.append(b)
    lo[0], hi[n] = 0, m
    for i in range(1, n + 1):              # both bounds non-decreasing
        lo[i] = max(lo[i], lo[i - 1])
    for i in range(n - 1, -1, -1):
        hi[i] = min(hi[i], hi[i + 1])
    for i in range(n + 1):                 # non-empty rows that overlap the previous
        lo[i] = min(lo[i], hi[i], hi[i - 1] if i else 0)
    return lo, hi


def align(sim: np.ndarray, null_pen: float = 0.5, band=None):
    """Monotonic DP over a (n x m) cosine-similarity matrix. Returns aligned spans.

    `band` = (lo, hi) from spread_band restricts row i to columns lo[i]..hi[i], so a
    block of many pages costs about as much as the pages aligned one by one.
    """
    n, m = sim.shape
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG)
    bk: dict = {}
    dp[0, 0] = 0.0
    lo, hi = band if band is not None else ([0] * (n + 1), [m] * (n + 1))
    for i in range(n + 1):
        for j in range(lo[i], hi[i] + 1):
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
    if (n, m) != (0, 0) and (n, m) not in bk:
        return align(sim, null_pen) if band is not None else []
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
    ap.add_argument("scn_page", type=int, help="Sicilian page (printed number)")
    ap.add_argument("en_page", type=int, help="English page (printed number)")
    ap.add_argument("--index", action="store_true",
                    help="pages are 0-based pdf indices, not printed numbers")
    ap.add_argument("--min-sim", type=float, default=0.5,
                    help="drop 1-1 alignments below this cosine (likely non-parallel)")
    args = ap.parse_args()

    scn_idx, en_idx = args.scn_page, args.en_page
    if not args.index:
        printed = printed_page_numbers(args.pdf)
        where = {p: i for i, p in enumerate(printed) if p is not None}
        missing = [p for p in (scn_idx, en_idx) if p not in where]
        if missing:
            ap.error(f"printed page(s) {missing} not found in {args.pdf.name}; "
                     "use --index with pdf page indices")
        scn_idx, en_idx = where[scn_idx], where[en_idx]
    doc = fitz.open(args.pdf)
    scn = page_sentences(doc, scn_idx, "it")
    en = page_sentences(doc, en_idx, "en")
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
