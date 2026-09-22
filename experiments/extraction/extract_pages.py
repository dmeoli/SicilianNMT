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

Page numbers: pairs are reported with the PRINTED page numbers (as30 p42/p43),
read from each page's running furniture, and the pdf index is kept alongside. The
two differ by an issue-dependent offset (as46: pdf index 59 = printed page 60), and
in the older phototypeset issues the offset even changes within an issue.

Sentence-level alignment (LaBSE + DP) is the next stage, see build_issue.py.
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
PAGENUM_RE = re.compile(r"^\s*(\d{1,3})\s*$")


def _glyph_lines(page, drop_phantom: bool) -> tuple[list[str], int]:
    """Text lines rebuilt from the glyphs, and the number of zero-advance spaces."""
    get = getattr(page, "get_text", None) or page.getText
    lines, phantom = [], 0
    for block in get("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span["chars"]]
            out = []
            for c, nxt in zip(chars, chars[1:] + [None]):
                if c["c"] == " " and nxt is not None and \
                        abs(nxt["bbox"][0] - c["bbox"][0]) < 0.3:
                    phantom += 1
                    if drop_phantom:
                        continue
                out.append(c["c"])
            lines.append("".join(out))
    return lines, phantom


_PHANTOM_DOCS: dict[str, bool] = {}


def _has_phantom_spaces(doc) -> bool:
    """True if at least 30% of the pages carry 10 or more zero-advance spaces.

    That is a property of the typesetting (AS22-23: 104 pages each); a scanned issue
    has a few such spaces by OCR accident, on at most 4 pages, mostly in captions and
    tables of contents, where dropping them would glue real words together.
    """
    key = doc.name
    if key not in _PHANTOM_DOCS:
        n = page_count(doc)
        hit = sum(_glyph_lines(doc[i], False)[1] >= 10 for i in range(n))
        _PHANTOM_DOCS[key] = hit >= 0.3 * n
    return _PHANTOM_DOCS[key]


def page_text(page) -> str:
    """Plain text of a page; `getText` is the pre-1.18 PyMuPDF spelling.

    Some typesetting (AS22-23) marks every hyphenation point with a space glyph of
    zero advance, which plain extraction turns into split words ("Bel lini",
    "cen tu ries"). In such documents the text is rebuilt from the glyphs, dropping
    every space whose next glyph starts at the same point, and the hyphens left
    inside a line ("civili- zation") are joined.
    """
    get = getattr(page, "get_text", None) or page.getText
    if not _has_phantom_spaces(page.parent):
        return get("text")
    lines, _ = _glyph_lines(page, True)
    return "\n".join(re.sub(r"([a-z])- (?=[a-z])", r"\1", ln) for ln in lines) + "\n"


def page_count(doc) -> int:
    return doc.page_count if hasattr(doc, "page_count") else doc.pageCount


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
    for i in range(page_count(doc)):
        clean = strip_furniture(page_text(doc[i]))
        out.append((i, classify(clean, scn, ENG_STOPWORDS), clean))
    doc.close()
    return out


def printed_page_numbers(pdf_path: Path, window: int = 2) -> list[int | None]:
    """Printed page number of every pdf page (None where it cannot be read).

    A bare number among the first/last 3 lines is a candidate; it is accepted only
    if a neighbouring page (within `window`) gives the same pdf-index offset, which
    rejects stray numbers in the body text. Pages without a readable number inherit
    the offset of the nearest accepted page within `window`.
    """
    doc = fitz.open(pdf_path)
    n = page_count(doc)
    cand: list[set[int]] = []
    for i in range(n):
        lines = [ln for ln in page_text(doc[i]).splitlines() if ln.strip()]
        cand.append({int(m.group(1)) - i for ln in lines[:3] + lines[-3:]
                     if (m := PAGENUM_RE.match(ln))})
    doc.close()
    offset: list[int | None] = [None] * n
    for i in range(n):
        near = set().union(*(cand[k] for k in range(max(0, i - window),
                                                   min(n, i + window + 1)) if k != i))
        both = cand[i] & near
        if len(both) == 1:
            offset[i] = both.pop()
    filled = list(offset)
    for i in range(n):
        if filled[i] is None:
            for d in range(1, window + 1):
                for k in (i - d, i + d):
                    if 0 <= k < n and offset[k] is not None and filled[i] is None:
                        filled[i] = offset[k]
    return [None if o is None else i + o for i, o in enumerate(filled)]


def candidate_pairs(pages: list[tuple[int, str, str]]) -> list[tuple[int, int]]:
    """Every (SC page, EN page) that are adjacent in the pdf, on either side.

    The paper copy puts the two versions on facing pages, but which neighbour is
    the facing one cannot be read off the labels: a misclassified page, or a stretch
    where the language order is swapped, turns the next-page rule into an off-by-one
    (as46: pdf pages 119/120 paired instead of 118/119). build_issue.py resolves
    each SC page to its best neighbour with LaBSE.
    """
    label = {i: lab for i, lab, _ in pages}
    return [(i, j) for i, lab, _ in pages if lab == "SC"
            for j in (i - 1, i + 1) if label.get(j) == "EN"]


def parallel_pairs(pages: list[tuple[int, str, str]],
                   printed: list[int | None] | None = None) -> list[tuple[int, int]]:
    """Label-only guess of the facing pairs (no embeddings), for quick reports.

    Where the printed number of the SC page is known, its facing page is the other
    half of the spread (even page on the left, odd on the right); otherwise the
    next page. NOTE: the early issues do not always follow the even-left rule, so
    the pipeline itself uses candidate_pairs + LaBSE instead of this guess.
    """
    label = {i: lab for i, lab, _ in pages}
    out = []
    for i, lab, _ in pages:
        if lab != "SC":
            continue
        j = i + 1
        if printed is not None and printed[i] is not None and printed[i] % 2:
            j = i - 1
        if label.get(j) == "EN":
            out.append((i, j))
    return out


def page_label(printed: list[int | None], i: int) -> str:
    """Printed page number as text, falling back to the pdf index (#59)."""
    return str(printed[i]) if printed[i] is not None else f"#{i}"


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
    printed = printed_page_numbers(args.pdf)
    pairs = parallel_pairs(pages, printed)

    print(f"{args.pdf.name}: {len(pages)} pages | SC={len(sc)} EN={len(en)} "
          f"OTHER={len(pages) - len(sc) - len(en)} | candidate pairs={len(pairs)}")
    print("  SC pages (pdf index):", _fmt_ranges(sc))
    print("  pairs (printed pages):",
          ", ".join(f"{page_label(printed, a)}/{page_label(printed, b)}" for a, b in pairs))

    if args.out and not args.report_only:
        args.out.mkdir(parents=True, exist_ok=True)
        text = {i: t for i, _, t in pages}
        (args.out / "sc.txt").write_text(
            "\n".join(text[a] for a, _ in pairs), encoding="utf-8")
        (args.out / "en.txt").write_text(
            "\n".join(text[b] for _, b in pairs), encoding="utf-8")
        (args.out / "pairs.tsv").write_text(
            "scn_page\ten_page\tscn_pdf_index\ten_pdf_index\n" +
            "".join(f"{page_label(printed, a)}\t{page_label(printed, b)}\t{a}\t{b}\n"
                    for a, b in pairs), encoding="utf-8")
        print(f"  wrote sc.txt, en.txt, pairs.tsv to {args.out}/")


if __name__ == "__main__":
    main()
