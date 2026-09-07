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
  - TODO (not yet local): our Arba Sicula PDF extraction for the issues Eryk lacks
    (AS43-46, AS01-18, AS21) via build_all.py -> add with --extra-scn-en once produced.

    python experiments/dataset/assemble_finetune.py \
        --ods ~/Downloads/ArbaSicula-Dieli_2024-10-20_Translation-Dataset.ods
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAG = REPO / "data/processed/napizia_magazine"


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def add_pairs(bucket: list[tuple[str, str, str]], src, tgt, provenance: str) -> None:
    """Append cleaned (src, tgt, provenance) pairs, skipping empties and identical sides."""
    for a, b in zip(src, tgt):
        a, b = norm(a), norm(b)
        if a and b and a != b:
            bucket.append((a, b, provenance))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ods", type=Path, required=True, help="Eryk's private spreadsheet (.ods)")
    ap.add_argument("--out", type=Path, default=REPO / "data/finetune")
    args = ap.parse_args()
    if not args.ods.exists():
        raise SystemExit(f"spreadsheet not found: {args.ods}")

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
    mono = [norm(x) for x in sheet("monolingual")[0] if norm(x)]
    val = sheet("validation")
    valid = [(norm(a), norm(b), norm(c)) for a, b, c in zip(val[0], val[1], val[2]) if norm(a)]

    # --- our Napizia magazine ---
    add_pairs(scn_en, read_lines(MAG / "magazine.scn"), read_lines(MAG / "magazine.en"),
              "ours:napizia-magazine")
    for it_file in sorted(MAG.glob("*.it")):
        base = it_file.with_suffix("")
        add_pairs(scn_it, read_lines(base.with_suffix(".scn")), read_lines(it_file),
                  "ours:napizia-magazine")

    # --- validation leakage guard: never let a train scn appear in valid ---
    valid_scn = {v[0] for v in valid}

    def dedup(pairs):
        seen, out, prov = set(), [], Counter()
        for a, b, p in pairs:
            if a in valid_scn:
                prov["dropped:in-valid"] += 1
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
