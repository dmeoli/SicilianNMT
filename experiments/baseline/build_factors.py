#!/usr/bin/env python3
"""Build a Sockeye source-factor stream (lemma per subword) — Sennrich's linguistic
input features (W16-2209). For each Sicilian word we look up its lemma
(vocab/lemma_dict_scn.json) and replicate it across the word's BPE subwords, so the
factor file aligns 1:1 with the BPE'd source. The factor tells the model the canonical
form, which helps it share statistics across inflections in a low-resource setting.

    python experiments/baseline/build_factors.py train.tok.scn train.bpe.scn train.fac.scn
"""
from __future__ import annotations
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEMMA = json.loads((REPO / "vocab/lemma_dict_scn.json").read_text(encoding="utf-8"))


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def lemma_of(word: str) -> str:
    """Lemma for a (tokenized) Sicilian word; '<unk>' when unknown (keeps the factor
    vocabulary small and the feature meaningful, rather than a sparse second embedding)."""
    return LEMMA.get(word) or LEMMA.get(_ascii(word)) or "<unk>"


def main() -> None:
    tok_f, bpe_f, out_f = (Path(a) for a in sys.argv[1:4])
    tok_lines = tok_f.read_text(encoding="utf-8").splitlines()
    bpe_lines = bpe_f.read_text(encoding="utf-8").splitlines()
    assert len(tok_lines) == len(bpe_lines), "tok/bpe line count mismatch"

    out, hits, total = [], 0, 0
    for tline, bline in zip(tok_lines, bpe_lines):
        words = tline.split()
        subs = bline.split()
        facs, wi = [], 0
        for s in subs:
            w = words[wi] if wi < len(words) else ""
            facs.append(lemma_of(w))
            total += 1
            hits += (w in LEMMA or _ascii(w) in LEMMA)
            if not s.endswith("@@"):   # last subword of the current word
                wi += 1
        out.append(" ".join(facs))
    out_f.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {out_f}  ({len(out)} lines, {hits}/{total} subwords lemmatized "
          f"= {hits/max(total,1):.0%})")


if __name__ == "__main__":
    main()
