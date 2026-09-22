#!/usr/bin/env python3
"""Run the modern extraction+alignment pipeline over ALL Arba Sicula issues.

Loads LaBSE once, processes every as-issues/*.pdf (skipping volumes that error),
cleans the pairs (clean_pairs.py, skipped with --no-clean),
dedups exact pairs across issues, and writes one combined parallel corpus plus a
per-issue summary. Each finished issue is checkpointed under <out>/issues/, so a run
killed halfway (e.g. out of memory) resumes from the missing issues.

    python experiments/extraction/build_all.py \
        --issues extract-text/as-issues --out data/processed/arbasicula
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer

from build_issue import process_issue
from clean_pairs import clean_pairs
from extract_pages import load_scn_stopwords

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issues", type=Path, default=REPO / "extract-text/as-issues")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/arbasicula")
    ap.add_argument("--min-page-sim", type=float, default=0.50)
    ap.add_argument("--min-sent-sim", type=float, default=0.40)
    ap.add_argument("--no-clean", action="store_true",
                    help="keep the raw aligner output (skip clean_pairs.py)")
    args = ap.parse_args()

    pdfs = sorted(args.issues.glob("*.pdf"))
    print(f"{len(pdfs)} issues found in {args.issues}", flush=True)
    ckpt = args.out / "issues"
    ckpt.mkdir(parents=True, exist_ok=True)
    model = None
    scn_stop = load_scn_stopwords()

    seen: set[tuple[str, str]] = set()
    rows: list[tuple] = []  # (issue, scn_page, en_page, sim, scn, en)
    summary: list[tuple[str, int, int, int, int, int]] = []
    cleaning: Counter = Counter()
    for pdf in pdfs:
        done = ckpt / f"{pdf.stem}.tsv"
        if done.exists():
            lines = done.read_text(encoding="utf-8").splitlines()
            n_cand, conf = map(int, lines[0].split("\t"))
            rows_i = [ln.split("\t") for ln in lines[1:]]
            if rows_i and len(rows_i[0]) == 2:      # checkpoint without page provenance
                prov = [("?", "?", 0.0) for _ in rows_i]
                scn, en = [r[0] for r in rows_i], [r[1] for r in rows_i]
            else:
                prov = [(r[0], r[1], float(r[2])) for r in rows_i]
                scn, en = [r[3] for r in rows_i], [r[4] for r in rows_i]
        else:
            if model is None:
                model = SentenceTransformer("sentence-transformers/LaBSE")
            prov = []
            try:
                scn, en, n_cand, conf = process_issue(
                    pdf, model, scn_stop, args.min_page_sim, args.min_sent_sim,
                    provenance=prov)
            except Exception as exc:  # noqa: BLE001 - keep batch going
                print(f"  {pdf.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                summary.append((pdf.stem, -1, -1, 0, 0, 0))
                continue
            done.write_text(f"{n_cand}\t{conf}\n" +
                            "".join(f"{sp}\t{ep}\t{si:.3f}\t{s}\t{e}\n"
                                    for (sp, ep, si), s, e in zip(prov, scn, en)),
                            encoding="utf-8")
        pairs = [(*p, s, e) for p, s, e in zip(prov, scn, en)]
        if not args.no_clean:
            pairs, why = clean_pairs(pairs)
            cleaning += why
        kept = 0
        for sp, ep, si, s, e in pairs:
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            rows.append((pdf.stem, sp, ep, si, s, e))
            kept += 1
        summary.append((pdf.stem, n_cand, conf, len(scn), len(pairs), kept))
        print(f"  {pdf.name}: cand {n_cand} -> conf {conf} -> {len(scn)} pairs "
              f"-> {len(pairs)} clean ({kept} new after dedup)", flush=True)

    with open(args.out / "corpus.tsv", "w", encoding="utf-8") as f:
        f.write("issue\tscn_page\ten_page\tsimilarity\tsicilian\tenglish\n")
        for issue, sp, ep, si, s, e in rows:
            f.write(f"{issue}\t{sp}\t{ep}\t{si:.3f}\t{s}\t{e}\n")
    (args.out / "corpus.scn").write_text(
        "\n".join(r[4] for r in rows) + "\n", encoding="utf-8")
    (args.out / "corpus.en").write_text(
        "\n".join(r[5] for r in rows) + "\n", encoding="utf-8")

    total_raw = sum(s[3] for s in summary if s[3] > 0)
    print("\n==== SUMMARY ====")
    print(f"{'issue':10} {'cand':>5} {'conf':>5} {'pairs':>6} {'clean':>6} {'new':>6}")
    for issue, nc, conf, pr, cl, kept in summary:
        tag = "ERR" if nc < 0 else ""
        print(f"{issue:10} {nc:>5} {conf:>5} {pr:>6} {cl:>6} {kept:>6} {tag}")
    total_clean = sum(s[4] for s in summary if s[4] > 0)
    print(f"\nTOTAL raw pairs: {total_raw:,} | clean: {total_clean:,} "
          f"| after cross-issue dedup: {len(rows):,}")
    if cleaning:
        print("cleaning:", dict(sorted(cleaning.items())))
    print(f"wrote corpus.tsv / corpus.scn / corpus.en to {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
