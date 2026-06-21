#!/usr/bin/env python3
"""Evaluation harness for Sicilian MT — BLEU + chrF via sacrebleu.

One scorer reused by every model so numbers are comparable. chrF is reported
because it is tokenization-independent (well suited to Sicilian, which has no
standard tokenizer); BLEU is reported for continuity with the paper.

    # score a hypotheses file against a reference
    python experiments/eval/evaluate.py --hyp out.en --ref data/dataset/test.en

    # sanity floor: copy the source as the "translation"
    python experiments/eval/evaluate.py --hyp data/dataset/test.scn --ref data/dataset/test.en --tag copy-baseline
"""
from __future__ import annotations
import argparse
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF


def score(hyp: list[str], ref: list[str]) -> dict:
    assert len(hyp) == len(ref), f"hyp/ref length mismatch: {len(hyp)} vs {len(ref)}"
    bleu_m, chrf_m, chrfpp_m = BLEU(), CHRF(), CHRF(word_order=2)  # CHRF++ = word_order 2
    bleu = bleu_m.corpus_score(hyp, [ref])
    chrf = chrf_m.corpus_score(hyp, [ref])
    chrfpp = chrfpp_m.corpus_score(hyp, [ref])
    return {
        "n": len(hyp),
        "bleu": bleu.score, "bleu_sig": bleu_m.get_signature().format(short=True),
        "chrf": chrf.score, "chrf_sig": chrf_m.get_signature().format(short=True),
        "chrf++": chrfpp.score,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    hyp = args.hyp.read_text(encoding="utf-8").splitlines()
    ref = args.ref.read_text(encoding="utf-8").splitlines()
    r = score(hyp, ref)
    tag = f"{args.tag}  " if args.tag else ""
    print(f"{tag}n={r['n']}  BLEU={r['bleu']:.2f}  chrF={r['chrf']:.2f}  chrF++={r['chrf++']:.2f}")
    print(f"  BLEU sig: {r['bleu_sig']}")
    print(f"  chrF sig: {r['chrf_sig']}")


if __name__ == "__main__":
    main()
