#!/usr/bin/env python3
"""Scrape parallel Sicilian/English/Italian text from the Napizia magazine.

Crawls the magazine archive, downloads each article's Sicilian page and its facing
translations, extracts the body paragraphs and pairs them positionally: the site
keeps the language versions paragraph-aligned 1:1, so equal paragraph counts give a
high-confidence pairing. Articles whose counts disagree are still written out and
flagged, to be reconciled by the same LaBSE sentence-alignment step as the Arba
Sicula PDFs (align_sentences.py).

COPYRIGHT: Maria Anna Manzella's poetry must NOT be scraped (agreed with Eryk
Wdowiak, 2026-08). The `mulinazzu` article is hard-excluded by slug, and as a safety
net any article whose author line contains "Manzella" is skipped too.

    python experiments/extraction/scrape_magazine.py --out data/processed/napizia_magazine
"""
from __future__ import annotations
import argparse
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://magazine.napizia.com"
ARCHIVE = f"{BASE}/vols/index.shtml"

# Per-language page file inside each article directory.
LANG_PAGE = {"scn": "index.shtml", "en": "index-en.shtml", "it": "index-it.shtml"}

# --- copyright exclusions (see module docstring) --------------------------------
EXCLUDE_SLUGS = {"mulinazzu"}          # "Lu Jornu ca lu Mulinazzu Chiancìu", M. A. Manzella
EXCLUDE_AUTHORS = ("manzella",)        # belt-and-suspenders: match the author line too

# Divs whose subtree is site chrome, not article text (matched by any class token).
BLOCK_CLASSES = {"navbar", "dropdown", "dropdown-content", "message",
                 "socialvanish", "socialappear", "socialicons", "footer"}

# Paragraphs that are boilerplate even inside the content column.
_JUNK_RE = re.compile(r"^\s*$|^copyright\b|^-->", re.IGNORECASE)
# Heading text that marks the end of the translated body (translator notes follow).
_NOTES_RE = re.compile(r"^\s*(noti|notes|note)\s*$", re.IGNORECASE)


def fetch(url: str, delay: float = 1.0) -> str:
    """GET a page politely (identifies itself, rate-limited by the caller's delay)."""
    req = urllib.request.Request(url, headers={"User-Agent": "SicilianNMT-scraper/1.0 "
                                               "(+https://github.com/dmeoli/SicilianNMT)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    time.sleep(delay)
    return html


def discover_articles(archive_html: str) -> list[tuple[str, str]]:
    """Return sorted unique (volume, slug) pairs from the archive's article links."""
    found = set(re.findall(r"/vols/(\d+)/([^/\"']+)/index[^\"']*\.shtml", archive_html))
    return sorted(found)


class _Article(HTMLParser):
    """Collect body <p> text, skipping chrome subtrees; also capture the <h2> author."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paras: list[str] = []
        self.author = ""
        self._div_classes: list[set[str]] = []   # stack of ancestor <div> class tokens
        self._buf: list[str] = []
        self._grab: str | None = None             # 'p' | 'h2' when inside a wanted tag
        self._stop = False                        # set once we reach the notes heading

    def _blocked(self) -> bool:
        return any(c & BLOCK_CLASSES for c in self._div_classes)

    def handle_starttag(self, tag: str, attrs: dict) -> None:
        a = dict(attrs)
        if tag == "div":
            self._div_classes.append(set((a.get("class") or "").split()))
        elif tag in ("h3", "h4") and not self._blocked():
            self._grab = "heading"
            self._buf = []
        elif tag == "h2" and not self._blocked():
            self._grab = "h2"
            self._buf = []
        elif tag == "p" and not self._blocked() and not self._stop:
            self._grab = "p"
            self._buf = []
        elif tag == "br" and self._grab:
            self._buf.append(" ")   # keep a paragraph on ONE line (line-aligned output)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._div_classes:
            self._div_classes.pop()
        elif tag == "h2" and self._grab == "h2":
            self.author = "".join(self._buf).strip()
            self._grab = None
        elif tag in ("h3", "h4") and self._grab == "heading":
            if _NOTES_RE.match("".join(self._buf).strip()):
                self._stop = True                 # translated body ends at the notes section
            self._grab = None
        elif tag == "p" and self._grab == "p":
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text and not _JUNK_RE.match(text):
                self.paras.append(text)
            self._grab = None

    def handle_data(self, data: str) -> None:
        if self._grab:
            self._buf.append(data)


def parse_article(html: str) -> tuple[list[str], str]:
    p = _Article()
    p.feed(html)
    return p.paras, p.author


def scrape(out_dir: Path, delay: float, langs: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    articles = discover_articles(fetch(ARCHIVE, delay))
    manifest: list[dict] = []
    combined: dict[str, list[str]] = {l: [] for l in langs}

    for vol, slug in articles:
        rec = {"vol": vol, "slug": slug, "status": None, "pairs": 0}
        if slug in EXCLUDE_SLUGS:
            rec["status"] = "skipped:manzella-slug"
            manifest.append(rec)
            print(f"  SKIP {vol}/{slug}  (copyright: Manzella slug)")
            continue

        pages: dict[str, list[str]] = {}
        author = ""
        for lang in langs:
            url = f"{BASE}/vols/{vol}/{slug}/{LANG_PAGE[lang]}"
            try:
                paras, a = parse_article(fetch(url, delay))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue                       # this language simply doesn't exist
                raise
            author = author or a
            if paras:
                pages[lang] = paras

        if any(k in author.lower() for k in EXCLUDE_AUTHORS):
            rec["status"] = "skipped:manzella-author"
            rec["author"] = author
            manifest.append(rec)
            print(f"  SKIP {vol}/{slug}  (copyright: author '{author}')")
            continue

        if "scn" not in pages or "en" not in pages:
            rec["status"] = "skipped:no-scn-en"
            manifest.append(rec)
            continue

        n = {l: len(p) for l, p in pages.items()}
        aligned = len({n[l] for l in ("scn", "en")}) == 1
        rec.update(author=author, counts=n, aligned=aligned)
        rec["status"] = "ok" if aligned else "ok:count-mismatch-needs-labse"
        rec["pairs"] = min(n["scn"], n["en"])

        for lang, paras in pages.items():
            (out_dir / f"{vol}-{slug}.{lang}").write_text("\n".join(paras) + "\n", encoding="utf-8")
            if aligned and lang in combined:
                combined[lang].extend(paras)
        manifest.append(rec)
        flag = "" if aligned else "  [!] count mismatch -> LaBSE"
        print(f"  OK   {vol}/{slug}  scn={n['scn']} en={n['en']}"
              f"{' it=' + str(n['it']) if 'it' in n else ''}{flag}")

    for lang, lines in combined.items():
        if lines:
            (out_dir / f"magazine.{lang}").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    ok = sum(1 for r in manifest if r["status"].startswith("ok"))
    skipped = sum(1 for r in manifest if r["status"].startswith("skipped"))
    pairs = sum(r["pairs"] for r in manifest)
    summary = {"articles": len(manifest), "ok": ok, "skipped": skipped,
               "aligned_paragraph_pairs": len(combined["scn"]), "total_pairs_estimate": pairs}
    print(f"\n{ok} articles OK, {skipped} skipped, "
          f"{len(combined['scn'])} paragraph-aligned scn/en pairs -> {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/napizia_magazine"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--langs", nargs="+", default=["scn", "en", "it"],
                    help="languages to fetch (scn and en required for a pair)")
    args = ap.parse_args()
    scrape(args.out, args.delay, args.langs)


if __name__ == "__main__":
    main()
