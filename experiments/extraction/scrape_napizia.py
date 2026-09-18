#!/usr/bin/env python3
"""Scrape the parallel Sicilian/English pages of the main Napizia website.

The site keeps a Sicilian and an English version of each page, paragraph by paragraph.
The page list is Eryk Wdowiak's own (email, 2026-09-18). Pairing works as for the
manifesto (scrape_manifesto.py): pair the two versions positionally, orient each pair
by which side carries the English function words, and drop the shared site menu.

    python experiments/extraction/scrape_napizia.py --out data/processed/napizia_site
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scrape_magazine import fetch, parse_article
from scrape_manifesto import pair_pages

BASE = "https://www.napizia.com"
PAGES = [
    ("napizia-main", "/index.shtml", "/index-en.shtml"),
    ("poets-page", "/pages/sicilian/poets-sc.shtml", "/pages/sicilian/poets.shtml"),
    ("introduction", "/pages/sicilian/intro-sc.shtml", "/pages/sicilian/intro-en.shtml"),
    ("sicilian-translator", "/pages/sicilian/translator-sc.shtml",
     "/pages/sicilian/translator.shtml"),
    ("find-a-word", "/pages/sicilian/trova-palora-sc.shtml",
     "/pages/sicilian/trova-palora.shtml"),
    ("boot-and-stem", "/pages/sicilian/sicilian-verbs-sc.shtml",
     "/pages/sicilian/sicilian-verbs.shtml"),
    ("next-steps", "/pages/sicilian/next-steps-sc.shtml", "/pages/sicilian/next-steps.shtml"),
    ("bibliography", "/pages/sicilian/bibliography-sc.shtml",
     "/pages/sicilian/bibliography.shtml"),
    ("come-to-napizia", "/pages/about/index.shtml", "/pages/about/index-en.shtml"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/napizia_site"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    combined: dict[str, list[str]] = {"scn": [], "en": []}
    for name, scn_url, en_url in PAGES:
        paras = {}
        for lang, url in (("scn", scn_url), ("en", en_url)):
            paras[lang] = parse_article(fetch(f"{BASE}{url}", args.delay))[0]
        pairs, aligned = pair_pages(paras["scn"], paras["en"])
        rec = {"page": name, "counts": {l: len(p) for l, p in paras.items()},
               "aligned": aligned, "pairs": len(pairs)}
        manifest.append(rec)
        if pairs:
            (args.out / f"{name}.scn").write_text(
                "\n".join(s for s, _ in pairs) + "\n", encoding="utf-8")
            (args.out / f"{name}.en").write_text(
                "\n".join(e for _, e in pairs) + "\n", encoding="utf-8")
            combined["scn"].extend(s for s, _ in pairs)
            combined["en"].extend(e for _, e in pairs)
        flag = "" if aligned else "  [!] paragraph counts differ"
        print(f"  {name}: scn={rec['counts']['scn']} en={rec['counts']['en']} "
              f"-> {len(pairs)} pairs{flag}")

    for lang, lines in combined.items():
        (args.out / f"napizia.{lang}").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(combined['scn'])} scn/en pairs from {len(PAGES)} pages -> {args.out}")


if __name__ == "__main__":
    main()
