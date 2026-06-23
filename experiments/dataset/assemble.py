#!/usr/bin/env python3
"""Assemble the unified Sicilian-English dataset from all sources.

Sources -> common cleaning -> cross-source dedup -> deterministic train/valid/test
split (test+valid held out from Arba Sicula = literary standard, not FLORES).
Italian-Sicilian (WikiMatrix) is cleaned and saved separately for the multilingual
direction.

    python experiments/dataset/assemble.py --out data/dataset

Writes train/valid/test .scn/.en (+ provenance tsv), itsc.{it,scn}, manifest.json.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AS_TSV = REPO / "data/processed/arbasicula/corpus.tsv"          # issue, sicilian, english
NLLB_TSV = REPO / "data/processed/nllb_clean/pairs.tsv"          # en, scn, score
WIKI_IT = REPO / "extract-text/WikiMatrix.it-scn.txt.it"
WIKI_SCN = REPO / "extract-text/WikiMatrix.it-scn.txt.scn"

EN_DROPCAP = re.compile(r"^([B-HJ-Z]) ([a-z])")   # drop-cap "T he"->"The"; skip A/I (real words)
GLOSS_NOISE = re.compile(r"(?:\b\w ){4,}")          # "C O S T I" glossary/OCR runs
KEY = re.compile(r"[^a-zà-ÿ0-9]+")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def clean(scn: str, en: str) -> tuple[str, str]:
    return norm(scn), EN_DROPCAP.sub(r"\1\2", norm(en))


def ok(scn: str, en: str) -> bool:
    sw, ew = scn.split(), en.split()
    if not (3 <= len(sw) <= 80 and 3 <= len(ew) <= 80):
        return False
    if not (1 / 3 <= len(sw) / len(ew) <= 3):
        return False
    if scn.lower() == en.lower():
        return False
    return not (GLOSS_NOISE.search(scn) or GLOSS_NOISE.search(en))


def key(s: str) -> str:
    return KEY.sub("", s.lower())


def load_pairs() -> list[dict]:
    """Return cleaned, filtered scn-en records with provenance (AS first = preferred)."""
    recs: list[dict] = []
    stats = {}
    asdf = pd.read_csv(AS_TSV, sep="\t", engine="python",
                       quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    n = 0
    for _, r in asdf.iterrows():
        s, e = clean(r["sicilian"], r["english"])
        if ok(s, e):
            # only the original LaBSE-extracted pairs are eligible for test/valid, so
            # adding Perl-merged (or any future) pairs never changes the held-out sets
            heldout = "_perl" not in str(r.get("issue", ""))
            recs.append({"scn": s, "en": e, "src": "arbasicula", "heldout": heldout}); n += 1
    stats["arbasicula"] = (len(asdf), n)

    nl = pd.read_csv(NLLB_TSV, sep="\t")
    n = 0
    for _, r in nl.iterrows():
        s, e = clean(r["scn"], r["en"])
        if ok(s, e):
            recs.append({"scn": s, "en": e, "src": "nllb", "heldout": False}); n += 1
    stats["nllb"] = (len(nl), n)
    return recs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data/dataset")
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--valid", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    recs, stats = load_pairs()
    print("cleaning/filter (raw -> kept):")
    for src, (raw, kept) in stats.items():
        print(f"  {src:12}: {raw:>7,} -> {kept:>7,}")

    # cross-source dedup on normalized Sicilian (AS processed first => preferred)
    seen, dedup = set(), []
    for r in recs:
        k = key(r["scn"])
        if k and k not in seen:
            seen.add(k); dedup.append(r)
    print(f"combined {len(recs):,} -> after scn dedup {len(dedup):,}")

    # split: if a frozen test/valid exists, reuse it VERBATIM so numbers stay comparable as
    # the dataset grows; otherwise draw test/valid from heldout-eligible Arba Sicula pairs.
    rng = random.Random(args.seed)
    FROZEN = REPO / "data/dataset_frozen_test"
    if (FROZEN / "test.scn").exists():
        def _load(name):
            sc = (FROZEN / f"{name}.scn").read_text(encoding="utf-8").splitlines()
            en = (FROZEN / f"{name}.en").read_text(encoding="utf-8").splitlines()
            return [{"scn": s, "en": e, "src": "frozen"} for s, e in zip(sc, en)]
        test, valid = _load("test"), _load("valid")
        holdkeys = {key(r["scn"]) for r in test} | {key(r["scn"]) for r in valid}
        train = [r for r in dedup if key(r["scn"]) not in holdkeys]
        print(f"using FROZEN test/valid ({len(test)}/{len(valid)}); train excludes their keys")
    else:
        cand = [r for r in dedup if r["src"] == "arbasicula" and r.get("heldout")]
        rng.shuffle(cand)
        test, valid = cand[:args.test], cand[args.test:args.test + args.valid]
        hold = {id(r) for r in test} | {id(r) for r in valid}
        train = [r for r in dedup if id(r) not in hold]
    rng.shuffle(train)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        (args.out / f"{name}.scn").write_text("\n".join(r["scn"] for r in split) + "\n", encoding="utf-8")
        (args.out / f"{name}.en").write_text("\n".join(r["en"] for r in split) + "\n", encoding="utf-8")
        with open(args.out / f"{name}.tsv", "w", encoding="utf-8") as f:
            f.write("src\tscn\ten\n")
            for r in split:
                f.write(f"{r['src']}\t{r['scn']}\t{r['en']}\n")

    # it-scn (auxiliary, multilingual direction): WikiMatrix + Napizia's hand-edited
    # "Good Sicilian from WikiMatrix" (514 curated pairs); clean + dedup, no split.
    it = WIKI_IT.read_text(encoding="utf-8").splitlines()
    sc = WIKI_SCN.read_text(encoding="utf-8").splitlines()
    GFWM = REPO / "data/external/Good-Sicilian-from-WikiMatrix/Napizia-edited-WikiMatrix.it-scn"
    if (GFWM / "Napizia-edited-WikiMatrix.it-scn.it").exists():
        it += (GFWM / "Napizia-edited-WikiMatrix.it-scn.it").read_text(encoding="utf-8").splitlines()
        sc += (GFWM / "Napizia-edited-WikiMatrix.it-scn.scn").read_text(encoding="utf-8").splitlines()
    seen_it, kept_it, kept_sc = set(), [], []
    for i, s in zip(it, sc):
        i, s = norm(i), norm(s)
        if len(i.split()) < 3 or len(s.split()) < 3:
            continue
        k = key(s)
        if k in seen_it:
            continue
        seen_it.add(k); kept_it.append(i); kept_sc.append(s)
    (args.out / "itsc.it").write_text("\n".join(kept_it) + "\n", encoding="utf-8")
    (args.out / "itsc.scn").write_text("\n".join(kept_sc) + "\n", encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "scn_en": {"train": len(train), "valid": len(valid), "test": len(test),
                   "by_source": {s: sum(r["src"] == s for r in dedup) for s in ("arbasicula", "nllb")}},
        "it_scn_wikimatrix": len(kept_it),
        "filters": "len 3-80 words, ratio 1/3-3, EN drop-cap fix, glossary-noise drop, scn-dedup",
        "test_valid_source": "arbasicula only",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nscn-en  train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    print(f"it-scn  (wikimatrix, aux): {len(kept_it):,}")
    print(f"wrote splits + itsc + manifest.json to {args.out}/")


if __name__ == "__main__":
    main()
