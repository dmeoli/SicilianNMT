#!/usr/bin/env python3
"""Normalize Sicilian text toward Standard Sicilian (for the preprocessing ablation).

Two levels:
  std  -- standardization spelling replacements only (docs/sicilian-standardization.md);
          case, accents and contractions are KEPT (safest for NLLB's own tokenizer).
  full -- std + uncontraction (o^ -> a lu, du^ -> di lu, ha^ -> havi a, mu^ -> mi lu) +
          ASCII-fold (drop accents), preserving case. Mirrors Wdowiak's tokenizer
          preprocessing, minus the lowercasing (NLLB is cased).

    python experiments/dataset/normalize_scn.py --level full in.scn out.scn
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

# --- standardization: whole-word replacements (case-insensitive, capitalization kept) ---
WORD = {
    "cchiù": "chiù", "cci": "ci", "io": "jo", "iu": "jo", "eu": "jo", "iju": "jìu",
    "non": "nun", "pri": "pi", "pir": "pi", "picchì": "pirchì", "oi": "oggi",
    "oj": "oggi", "peju": "peggiu", "soccu": "zoccu",
}
# --- standardization: apostrophe/accent forms (literal, longest first) ---
APOS = [
    ("cc'è", "c'è"), ("cc'e'", "c'è"), ("c'e'", "c'è"), ("'unn", "nun"),
    ("'un", "nun"), ("cu'", "cui"), ("ch'", "chi"), ("me'", "mè"), ("po'", "pò"),
    ("si'", "sì"), ("so'", "sò"), ("su'", "sunnu"), ("to'", "tò"), ("e'", "è"),
]
# --- standardization: word-initial prefixes (bell- -> bedd-, etc.) ---
PREFIX = {"bell": "bedd", "crij": "crid", "nsign": "nzign", "pens": "penz", "pins": "pinz"}

# --- uncontraction (full level): preposition+article, "aviri", clitics ---
UNCONTRACT = {
    "ô": "a lu", "â": "a la", "ê": "a li", "ôn": "a un",
    "cû": "cu lu", "cô": "cu lu", "câ": "cu la", "chî": "cu li", "chê": "cu li",
    "dû": "di lu", "dô": "di lu", "dâ": "di la", "dî": "di li", "dê": "di li",
    "pû": "pi lu", "pô": "pi lu", "pâ": "pi la", "pî": "pi li", "pê": "pi li",
    "nnô": "nni lu", "nnû": "nni lu", "nnâ": "nni la", "nnê": "nni li", "nnî": "nni li",
    "ntô": "nta lu", "ntû": "nta lu", "ntâ": "nta la", "ntê": "nta li", "ntî": "nta li",
    "ntrô": "ntra lu", "ntrâ": "ntra la", "ntrê": "ntra li",
    "mû": "mi lu", "mâ": "mi la", "mî": "mi li",
    "tû": "ti lu", "tâ": "ti la", "tî": "ti li",
    "sû": "si lu", "sâ": "si la", "sî": "si li",
    "nû": "ni lu", "nâ": "ni la", "nî": "ni li",
    "vû": "vi lu", "vâ": "vi la", "vî": "vi li",
    "ciû": "ci lu", "ciâ": "ci la", "cî": "ci li",
    "hê": "haiu a", "hâ": "havi a", "amâ": "avemu a", "atâ": "aviti a", "hannâ": "hannu a",
}
ASCII = str.maketrans("àáâãèéêìíîòóôõùúûÀÁÂÃÈÉÊÌÍÎÒÓÔÕÙÚÛ",
                      "aaaaeeeiiioooouuuAAAAEEEIIIOOOOUUU")


def _cap(repl: str, matched: str) -> str:
    return repl[:1].upper() + repl[1:] if matched[:1].isupper() else repl


def _apply_word(text: str, mapping: dict) -> str:
    if not mapping:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in
                     sorted(mapping, key=len, reverse=True)) + r")\b", re.IGNORECASE)
    return pat.sub(lambda m: _cap(mapping[m.group(0).lower()], m.group(0)), text)


def _apply_apos(text: str, pairs: list) -> str:
    for src, dst in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(src) + r"(?!\w)", lambda m, d=dst: _cap(d, m.group(0)),
                      text, flags=re.IGNORECASE)
    return text


def _apply_prefix(text: str, mapping: dict) -> str:
    for src, dst in mapping.items():
        text = re.sub(r"\b" + re.escape(src), lambda m, d=dst: _cap(d, m.group(0)),
                      text, flags=re.IGNORECASE)
    return text


def normalize(text: str, level: str = "full") -> str:
    text = _apply_apos(text, APOS)
    text = _apply_word(text, WORD)
    text = _apply_prefix(text, PREFIX)
    if level == "full":
        text = _apply_word(text, UNCONTRACT)   # expand contractions...
        text = text.translate(ASCII)           # ...then drop any remaining accents
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", type=Path)
    ap.add_argument("outfile", type=Path)
    ap.add_argument("--level", choices=["std", "full"], default="full")
    args = ap.parse_args()
    lines = args.infile.read_text(encoding="utf-8").splitlines()
    out = [normalize(ln, args.level) for ln in lines]
    args.outfile.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"normalized {len(lines)} lines ({args.level}) -> {args.outfile}")


if __name__ == "__main__":
    main()
