#!/usr/bin/env python3
"""Scrape the parallel Sicilian/English text of Eryk Wdowiak's Young Sicilian Manifesto.

The manifesto has two twin pages, "Manifestu d'un Giùvini Sicilianu" (Sicilian) and
"Young Sicilian Manifesto" (English), whose paragraphs correspond 1:1. Two details
break a purely positional pairing: (i) the site menu, the Buttitta epigraph and the
copyright line are identical on both pages, and (ii) the bilingual verse couplets
appear on both pages with the two languages in opposite order. We therefore pair
positionally, orient each pair by which side has English function words, take the
epigraph from the consecutive paragraphs of the identical blocks, and dedup.

    python experiments/extraction/scrape_manifesto.py --out data/processed/napizia_manifesto
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

from extract_pages import ENG_STOPWORDS, WORD_RE
from scrape_magazine import fetch, parse_article

BASE = "https://www.wdowiak.me/archive/sicilian"
PAGES = {"scn": f"{BASE}/giuvini-sicilianu.shtml", "en": f"{BASE}/young-sicilian.shtml"}
_DASH_RE = re.compile(r"^\s*[—–-]")          # attribution lines ("— Gnazziu Buttitta")
# extract_pages' list plus the function words of the short verse translations
ENGLISH = ENG_STOPWORDS | set("when then will do our their if so can what which more".split())


def english_ratio(text: str) -> float:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return sum(t in ENGLISH for t in toks) / len(toks)


def orient(x: str, y: str, margin: float = 0.1, min_words: int = 6,
           max_ratio: float = 2.5) -> tuple[str, str] | None:
    """Return (scn, en) if one of the two is clearly the English one, else None.

    Only the English side is tested: short Sicilian verses often contain none of our
    Sicilian stopwords, but the English line of a translated pair always has some.
    Wildly different lengths mean the positional pairing has slipped (one page has a
    paragraph the other lacks), so those pairs are dropped rather than guessed.
    """
    if _DASH_RE.match(x) or _DASH_RE.match(y) or \
            min(len(WORD_RE.findall(x)), len(WORD_RE.findall(y))) < min_words or \
            not 1 / max_ratio <= len(x) / len(y) <= max_ratio:
        return None           # attributions, site-menu entries, slipped pairs
    ex, ey = english_ratio(x), english_ratio(y)
    if ey - ex >= margin:
        return x, y
    if ex - ey >= margin:
        return y, x
    return None


def pair_pages(scn_paras: list[str], en_paras: list[str]) -> tuple[list[tuple[str, str]], bool]:
    """Oriented, deduplicated (scn, en) paragraph pairs, plus whether counts agreed."""
    pairs: list[tuple[str, str]] = []
    shared: list[str] = []                   # paragraphs identical on both pages, in order
    for x, y in zip(scn_paras, en_paras):
        if x == y:
            shared.append(x)
        elif (o := orient(x, y)) is not None:
            pairs.append(o)
    for x, y in zip(shared, shared[1:]):     # bilingual blocks shown on both pages
        o = orient(x, y)
        if o is not None and o[0] == x:
            pairs.append(o)
    return list(dict.fromkeys(pairs)), len(scn_paras) == len(en_paras)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/napizia_manifesto"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    paras = {lang: parse_article(fetch(url, args.delay))[0] for lang, url in PAGES.items()}
    pairs, aligned = pair_pages(paras["scn"], paras["en"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifesto.scn").write_text("\n".join(s for s, _ in pairs) + "\n", encoding="utf-8")
    (args.out / "manifesto.en").write_text("\n".join(e for _, e in pairs) + "\n", encoding="utf-8")
    flag = "" if aligned else "  [!] paragraph counts differ, check the pairing"
    print(f"paragraphs scn={len(paras['scn'])} en={len(paras['en'])} -> "
          f"{len(pairs)} scn/en pairs -> {args.out}{flag}")


if __name__ == "__main__":
    main()
