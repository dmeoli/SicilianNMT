#!/usr/bin/env python3
"""Modern PDF extraction + page-language classification for Arba Sicula issues.

Prototype (Phase 3) replacing the legacy pdflatex-rotate-crop + manual page-range
pipeline (extract-text/01_extract-text.pl). PyMuPDF extracts clean text directly
from the original PDFs — no LaTeX, no 180° rotation, no viewport crop. We then
classify each page as Sicilian (SC) / English (EN) / OTHER with a stopword-ratio
heuristic and pair facing SC/EN pages into candidate parallel articles.

Finding on as30: this recovers all 29 hand-selected Sicilian pages AND surfaces
genuine parallel pages the manual process skipped (e.g. the pp.30-31 poem).

Usage:
    python extract_pages.py path/to/as30.pdf --out out_dir
    python extract_pages.py path/to/as30.pdf --report-only

Sentence-level alignment (LaBSE/SONAR + vecalign) is the next stage, not here.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF

# Sicilian stopwords shipped in the repo; English function words inline.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCN_STOPWORDS_PATH = REPO_ROOT / "vocab" / "stopwords_scn.txt"
ENG_STOPWORDS = set(
    "the of and to a in that is was he for it with as his on be at by i this had "
    "not are but from or her she you we they him my me all".split()
)

WORD_RE = re.compile(r"[a-zA-Zàèéìòùâêîôûäëïöü'‘’ʼ]+")
# Page furniture to drop: running header and bare page numbers.
HEADER_RE = re.compile(r"^\s*(arba sicula\b.*|\d{1,3})\s*$", re.IGNORECASE)


def load_scn_stopwords(path: Path = SCN_STOPWORDS_PATH) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def strip_furniture(text: str) -> str:
    """Drop running headers ("Arba Sicula XXX") and standalone page numbers."""
    return "\n".join(ln for ln in text.splitlines() if not HEADER_RE.match(ln))


def classify(text: str, scn: set[str], eng: set[str], min_tokens: int = 40,
             min_ratio: float = 0.04) -> str:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if len(toks) < min_tokens:
        return "OTHER"
    s = sum(t in scn for t in toks) / len(toks)
    e = sum(t in eng for t in toks) / len(toks)
    if max(s, e) < min_ratio:
        return "OTHER"
    return "SC" if s > e else "EN"


def classify_document(pdf_path: Path, scn: set[str]) -> list[tuple[int, str, str]]:
    """Return [(page_index, label, clean_text), ...] for every page."""
    doc = fitz.open(pdf_path)
    out = []
    for i in range(doc.page_count):
        clean = strip_furniture(doc[i].get_text("text"))
        out.append((i, classify(clean, scn, ENG_STOPWORDS), clean))
    doc.close()
    return out


def parallel_pairs(pages: list[tuple[int, str, str]]) -> list[tuple[int, int]]:
    """Facing-page heuristic: an SC page immediately followed by an EN page.

    NOTE: facing-page adjacency is necessary but not sufficient — the next stage
    must confirm the two pages are mutual translations via cross-lingual
    sentence embeddings before trusting a pair.
    """
    label = {i: lab for i, lab, _ in pages}
    return [(i, i + 1) for i, lab, _ in pages
            if lab == "SC" and label.get(i + 1) == "EN"]


def _fmt_ranges(nums: list[int]) -> str:
    nums = sorted(nums)
    spans: list[list[int]] = []
    for x in nums:
        if spans and x == spans[-1][1] + 1:
            spans[-1][1] = x
        else:
            spans.append([x, x])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path, help="Arba Sicula issue PDF")
    ap.add_argument("--out", type=Path, help="dir to write sc.txt / en.txt / pairs.tsv")
    ap.add_argument("--report-only", action="store_true", help="print summary, write nothing")
    args = ap.parse_args()

    scn = load_scn_stopwords()
    pages = classify_document(args.pdf, scn)
    sc = [i for i, lab, _ in pages if lab == "SC"]
    en = [i for i, lab, _ in pages if lab == "EN"]
    pairs = parallel_pairs(pages)

    print(f"{args.pdf.name}: {len(pages)} pages | SC={len(sc)} EN={len(en)} "
          f"OTHER={len(pages) - len(sc) - len(en)} | candidate pairs={len(pairs)}")
    print("  SC pages :", _fmt_ranges(sc))
    print("  pairs    :", ", ".join(f"{a}/{b}" for a, b in pairs))

    if args.out and not args.report_only:
        args.out.mkdir(parents=True, exist_ok=True)
        text = {i: t for i, _, t in pages}
        (args.out / "sc.txt").write_text(
            "\n".join(text[a] for a, _ in pairs), encoding="utf-8")
        (args.out / "en.txt").write_text(
            "\n".join(text[b] for _, b in pairs), encoding="utf-8")
        (args.out / "pairs.tsv").write_text(
            "\n".join(f"{a}\t{b}" for a, b in pairs), encoding="utf-8")
        print(f"  wrote sc.txt, en.txt, pairs.tsv to {args.out}/")


if __name__ == "__main__":
    main()
