#!/usr/bin/env python3
"""Run the modern extraction+alignment pipeline over ALL Arba Sicula issues.

Loads LaBSE once, processes every as-issues/*.pdf (skipping volumes that error),
dedups exact pairs across issues, and writes one combined parallel corpus plus a
per-issue summary.

    python experiments/extraction/build_all.py \
        --issues extract-text/as-issues --out data/processed/arbasicula
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from build_issue import process_issue
from extract_pages import load_scn_stopwords

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issues", type=Path, default=REPO / "extract-text/as-issues")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/arbasicula")
    ap.add_argument("--min-page-sim", type=float, default=0.50)
    ap.add_argument("--min-sent-sim", type=float, default=0.40)
    args = ap.parse_args()

    pdfs = sorted(args.issues.glob("*.pdf"))
    print(f"{len(pdfs)} issues found in {args.issues}", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE")
    scn_stop = load_scn_stopwords()

    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str]] = []  # (issue, scn, en)
    summary: list[tuple[str, int, int, int, int]] = []
    for pdf in pdfs:
        try:
            scn, en, n_cand, conf = process_issue(
                pdf, model, scn_stop, args.min_page_sim, args.min_sent_sim)
        except Exception as exc:  # noqa: BLE001 - keep batch going
            print(f"  {pdf.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            summary.append((pdf.stem, -1, -1, 0, 0))
            continue
        kept = 0
        for s, e in zip(scn, en):
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            rows.append((pdf.stem, s, e))
            kept += 1
        summary.append((pdf.stem, n_cand, conf, len(scn), kept))
        print(f"  {pdf.name}: cand {n_cand} -> conf {conf} -> {len(scn)} pairs "
              f"({kept} new after dedup)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "corpus.tsv", "w", encoding="utf-8") as f:
        f.write("issue\tsicilian\tenglish\n")
        for issue, s, e in rows:
            f.write(f"{issue}\t{s}\t{e}\n")
    (args.out / "corpus.scn").write_text(
        "\n".join(s for _, s, _ in rows) + "\n", encoding="utf-8")
    (args.out / "corpus.en").write_text(
        "\n".join(e for _, _, e in rows) + "\n", encoding="utf-8")

    total_raw = sum(s[3] for s in summary if s[3] > 0)
    print("\n==== SUMMARY ====")
    print(f"{'issue':10} {'cand':>5} {'conf':>5} {'pairs':>6} {'new':>6}")
    for issue, nc, conf, pr, kept in summary:
        tag = "ERR" if nc < 0 else ""
        print(f"{issue:10} {nc:>5} {conf:>5} {pr:>6} {kept:>6} {tag}")
    print(f"\nTOTAL raw pairs: {total_raw:,} | after cross-issue dedup: {len(rows):,}")
    print(f"wrote corpus.tsv / corpus.scn / corpus.en to {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
