#!/usr/bin/env python3
"""Clean the aligned Arba Sicula pairs: strip page residue, drop the pairs that are
visibly wrong.

The filters follow Eryk's review of the full extraction (September 2026):

  * residue stripped from the text: the InDesign slug of AS43
    ("ArbaSicula Mast43.indd 9 8/5/2022 10:29:41 PM") and the "Return to the TOC"
    link of the digitised issues; a pair left empty is dropped;
  * length: the longer side has more than 1.6 times the words of the shorter one
    (a partial alignment, one side covering only part of the other), counted
    only when the longer side has at least 5 words;
  * table of contents: dot or bullet leaders of 6 or more ("......", "••••••");
  * bad OCR: 2 or more characters that never occur in the printed text
    (~ ¢ • · | \\ { } ^ _), or more than 25% of the lowercase English words
    missing from the system word list (/usr/share/dict); capitalised words are not
    counted, so names do not trip the filter. Without a word list this check is
    skipped and a warning is printed.

    from clean_pairs import clean_pairs
    kept, dropped = clean_pairs(pairs)     # pairs: [(scn, en), ...]
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

SLUG_RE = re.compile(r"ArbaSicula\s*Mast\d+\.indd\s+\d+"
                     r"(?:\s+\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M)?)*")
TOC_LINK_RE = re.compile(r"\bReturn to the TOC\b", re.IGNORECASE)
LEADER_RE = re.compile(r"\.{6,}|•{6,}")
JUNK_CHARS = set("~¢•·|\\{}^_")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
DICTS = (Path("/usr/share/dict/american-english"), Path("/usr/share/dict/british-english"),
         Path("/usr/share/dict/words"))

MAX_RATIO = 1.6
MIN_WORDS_FOR_RATIO = 5
MAX_EN_OOV = 0.25
MIN_EN_WORDS_FOR_OOV = 4

_EN_WORDS: set[str] | None = None


def english_words() -> set[str]:
    global _EN_WORDS
    if _EN_WORDS is None:
        _EN_WORDS = set()
        for p in DICTS:
            if p.exists():
                _EN_WORDS |= {w.strip().lower()
                              for w in p.read_text(encoding="utf-8", errors="ignore").split()}
        if not _EN_WORDS:
            print("clean_pairs: no word list in /usr/share/dict, English OCR check skipped",
                  file=sys.stderr)
    return _EN_WORDS


def strip_residue(text: str) -> str:
    text = SLUG_RE.sub(" ", text)
    text = TOC_LINK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def en_oov(text: str) -> float | None:
    """Share of lowercase English words not in the word list (None if too few)."""
    vocab = english_words()
    if not vocab:
        return None
    low = [re.sub(r"['’]s$", "", w.lower()) for w in WORD_RE.findall(text)
           if w[0].islower() and len(w) > 1]
    if len(low) < MIN_EN_WORDS_FOR_OOV:
        return None
    return sum(w not in vocab for w in low) / len(low)


def reject_reason(scn: str, en: str) -> str | None:
    """Why the (already stripped) pair must be dropped, or None to keep it."""
    if not scn or not en:
        return "empty"
    ws, we = len(scn.split()), len(en.split())
    if max(ws, we) >= MIN_WORDS_FOR_RATIO and max(ws, we) > MAX_RATIO * min(ws, we):
        return "length-ratio"
    if LEADER_RE.search(scn) or LEADER_RE.search(en):
        return "toc-leader"
    if sum(c in JUNK_CHARS for c in scn + en) >= 2:
        return "ocr-junk-chars"
    oov = en_oov(en)
    if oov is not None and oov > MAX_EN_OOV:
        return "ocr-english"
    return None


def clean_pairs(pairs):
    """Strip residue from every pair and drop the rejected ones.

    Accepts (scn, en) tuples or longer tuples whose last 2 fields are scn and en
    (the other fields are carried through). Returns (kept, Counter of reasons).
    """
    kept, why = [], Counter()
    for p in pairs:
        *head, scn, en = p
        s, e = strip_residue(scn), strip_residue(en)
        if (s, e) != (scn, en):
            why["stripped:residue"] += 1
        r = reject_reason(s, e)
        if r:
            why[f"dropped:{r}"] += 1
            continue
        kept.append((*head, s, e))
    return kept, why
