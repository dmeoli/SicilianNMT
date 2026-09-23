#!/usr/bin/env python3
"""Assemble the combined fine-tuning dataset (Eryk's curated data + our extraction).

Merges Eryk Wdowiak's private spreadsheet (Arba Sicula / Dieli / Napizia / textbooks) with
the parallel text our pipeline collected, into deduplicated per-direction files for the
fine-tuning stage. The validation set is kept frozen and held out of train (leakage guard).

PRIVACY: Eryk's spreadsheet is private ("do not share"). Its path is passed at runtime and
never committed; all outputs go under data/ which is gitignored. Do NOT git-add data/finetune.

Sources
  - Eryk ODS sheets: scn-eng [scn,en], scn-ita [scn,it], scn-ita-eng [scn,it,en],
    monolingual [scn], validation [scn,it,en].
  - Ours: data/processed/napizia_magazine (scn/en, and scn/it where present).
  - Ours: data/processed/napizia_manifesto (scn/en, Young Sicilian Manifesto).
  - Ours: data/processed/napizia_site (scn/en, the main Napizia pages Eryk listed).
  - Ours: data/processed/arbasicula*/corpus.{scn,en}, our Arba Sicula PDF extraction for
    the issues Eryk lacks (AS01-18, AS21, AS43-46) via build_all.py.

    python experiments/dataset/assemble_finetune.py \
        --ods ~/Downloads/eryk/ArbaSicula-Dieli_2024-10-20_Translation-Dataset.ods
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAG = REPO / "data/processed/napizia_magazine"
sys.path.insert(0, str(REPO / "experiments/dataset"))  # for normalize_scn

_NRM = None  # set from --normalize; applied to the Sicilian side only


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def nscn(s: str) -> str:
    """Whitespace-clean, then std/full-normalise the SICILIAN side if --normalize is set."""
    s = norm(s)
    return _NRM(s) if (_NRM and s) else s


SCANNED_VOLS = set(range(1, 19)) | {21}   # PDFs that are scans of the paper copies


def read_as_tsv(path: Path, min_vol: int, include_scans: bool) -> dict[str, list]:
    """(scn, en) pairs of our Arba Sicula extraction, by issue, volumes filtered."""
    out: dict[str, list] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            vol = int(row["issue"][2:4])
            if vol in SCANNED_VOLS and not include_scans:
                continue
            if vol < min_vol and vol not in SCANNED_VOLS:
                continue
            out.setdefault(row["issue"], []).append((row["sicilian"], row["english"]))
    return out


def read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def add_pairs(bucket: list[tuple[str, str, str]], src, tgt, provenance: str) -> None:
    """Append cleaned (scn, tgt, provenance) pairs; the scn side (src) is normalised."""
    for a, b in zip(src, tgt):
        a, b = nscn(a), norm(b)
        if a and b and a != b:
            bucket.append((a, b, provenance))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ods", type=Path, required=True, help="Eryk's private spreadsheet (.ods)")
    ap.add_argument("--out", type=Path, default=REPO / "data/finetune")
    ap.add_argument("--as-tsv", type=Path, default=REPO / "data/processed/as_full_gift/corpus.tsv",
                    help="our full Arba Sicula extraction (issue, pages, similarity, scn, en)")
    ap.add_argument("--as-min-vol", type=int, default=19,
                    help="first Arba Sicula volume to take from --as-tsv")
    ap.add_argument("--as-include-scans", action="store_true",
                    help="also take the scanned issues (AS01-18, AS21), whose OCR has no accents")
    ap.add_argument("--test-scn", type=Path,
                    help="frozen test set (Sicilian side); its lines are kept out of train. "
                         "It lives on Drive, e.g. SicilianNMT-colab/data/test.scn")
    ap.add_argument("--normalize", choices=["none", "std", "full"], default="std",
                    help="orthographic normalisation applied to the Sicilian side")
    args = ap.parse_args()
    if not args.ods.exists():
        raise SystemExit(f"spreadsheet not found: {args.ods}")

    global _NRM
    if args.normalize != "none":
        from normalize_scn import normalize
        _NRM = lambda s: normalize(s, args.normalize)  # noqa: E731

    import pandas as pd

    def sheet(name):
        return pd.read_excel(args.ods, sheet_name=name, engine="odf", header=None).fillna("")

    scn_en: list[tuple[str, str, str]] = []
    scn_it: list[tuple[str, str, str]] = []

    # --- Eryk's spreadsheet ---
    e_se = sheet("scn-eng")
    add_pairs(scn_en, e_se[0], e_se[1], "eryk:scn-eng")
    e_si = sheet("scn-ita")
    add_pairs(scn_it, e_si[0], e_si[1], "eryk:scn-ita")
    e_tri = sheet("scn-ita-eng")
    add_pairs(scn_en, e_tri[0], e_tri[2], "eryk:scn-ita-eng")
    add_pairs(scn_it, e_tri[0], e_tri[1], "eryk:scn-ita-eng")
    mono = [nscn(x) for x in sheet("monolingual")[0] if norm(x)]
    val = sheet("validation")
    valid = [(nscn(a), norm(b), norm(c)) for a, b, c in zip(val[0], val[1], val[2]) if norm(a)]

    # --- our Napizia magazine ---
    add_pairs(scn_en, read_lines(MAG / "magazine.scn"), read_lines(MAG / "magazine.en"),
              "ours:napizia-magazine")
    for it_file in sorted(MAG.glob("*.it")):
        base = it_file.with_suffix("")
        add_pairs(scn_it, read_lines(base.with_suffix(".scn")), read_lines(it_file),
                  "ours:napizia-magazine")

    # --- our Young Sicilian Manifesto scrape ---
    man = REPO / "data/processed/napizia_manifesto"
    add_pairs(scn_en, read_lines(man / "manifesto.scn"), read_lines(man / "manifesto.en"),
              "ours:napizia-manifesto")

    # --- our main-Napizia-site scrape ---
    site = REPO / "data/processed/napizia_site"
    add_pairs(scn_en, read_lines(site / "napizia.scn"), read_lines(site / "napizia.en"),
              "ours:napizia-site")

    # --- our Arba Sicula PDF extraction (scn-en) ---
    # Eryk's hand-aligned sheets and our alignment of the same issue are complements,
    # not alternatives (his cover 10-20% of the pairs of an issue), so every issue goes
    # in and the exact repeats are deduped below. AS01-18 and AS21 are left out by
    # default: those PDFs are scans of the paper copies, whose OCR carries no accent at
    # all over the vowels, while AS19 onward come from the original computer files.
    as_rows = read_as_tsv(args.as_tsv, args.as_min_vol, args.as_include_scans)
    for issue, pairs in sorted(as_rows.items()):
        add_pairs(scn_en, [a for a, _ in pairs], [b for _, b in pairs], f"ours:{issue}")

    # --- leakage guard: never let a train scn appear in valid, nor in the frozen test ---
    valid_scn = {v[0] for v in valid}
    test_scn: set[str] = set()
    test_inside: tuple[str, ...] = ()
    if args.test_scn:
        raw = read_lines(args.test_scn)
        test_scn = {x for x in raw if x} | {nscn(x) for x in raw if x}
        # a test sentence can also sit inside a longer training pair (a different
        # sentence split of the same page), which an exact match would not catch
        test_inside = tuple(sorted({x for x in test_scn if len(x.split()) >= 6}))
        print(f"test lines held out : {len(test_scn)} forms from {args.test_scn}")

    def dedup(pairs):
        seen, out, prov = set(), [], Counter()
        for a, b, p in pairs:
            if a in valid_scn:
                prov["dropped:in-valid"] += 1
                continue
            if a in test_scn:
                prov["dropped:in-test"] += 1
                continue
            if any(t in a for t in test_inside):
                prov["dropped:contains-test"] += 1
                continue
            k = (a, b)
            if k in seen:
                prov["dropped:dup"] += 1
                continue
            seen.add(k)
            out.append((a, b, p))
            prov[p] += 1
        return out, prov

    scn_en, prov_en = dedup(scn_en)
    scn_it, prov_it = dedup(scn_it)

    args.out.mkdir(parents=True, exist_ok=True)

    def write_pairs(pairs, a_ext, b_ext):
        (args.out / f"train.{a_ext}").write_text("\n".join(p[0] for p in pairs) + "\n", encoding="utf-8")
        (args.out / f"train.{b_ext}").write_text("\n".join(p[1] for p in pairs) + "\n", encoding="utf-8")

    write_pairs(scn_en, "scn-en.scn", "scn-en.en")
    write_pairs(scn_it, "scn-it.scn", "scn-it.it")
    (args.out / "mono.scn").write_text("\n".join(mono) + "\n", encoding="utf-8")
    (args.out / "valid.scn").write_text("\n".join(v[0] for v in valid) + "\n", encoding="utf-8")
    (args.out / "valid.it").write_text("\n".join(v[1] for v in valid) + "\n", encoding="utf-8")
    (args.out / "valid.en").write_text("\n".join(v[2] for v in valid) + "\n", encoding="utf-8")

    manifest = {
        "normalize": args.normalize,
        "scn_en_pairs": len(scn_en), "scn_it_pairs": len(scn_it),
        "mono_scn_lines": len(mono), "valid_lines": len(valid),
        "scn_en_provenance": dict(prov_en), "scn_it_provenance": dict(prov_it),
        "note": "PRIVATE (contains Eryk's data) -- gitignored, do not share.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print(f"scn-en train : {len(scn_en):>6}  {dict(prov_en)}")
    print(f"scn-it train : {len(scn_it):>6}  {dict(prov_it)}")
    print(f"mono scn     : {len(mono):>6}")
    print(f"valid (frozen): {len(valid):>6}  (held out of train)")
    print(f"\nwrote {args.out}  (PRIVATE, gitignored)")


if __name__ == "__main__":
    main()
