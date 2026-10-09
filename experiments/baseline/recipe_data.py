#!/usr/bin/env python3
"""Training sets for reproducing the 2022 baseline of E. Wdowiak with our pipeline (§7.12).

Three scn-en training sets, trained with his Sockeye configuration and scored on his own
hand-selected test (the 121 AS38-39 lines of the `validation` sheet of his data), so that the
only thing that changes between them is where the parallel text comes from:

  A  his hand-aligned sheets as they are (scn-eng, plus the scn/en columns of scn-ita-eng):
     checks that our re-implementation of his training reproduces his published score;
  B  our automatic alignment of the born-digital Arba Sicula issues (AS19 onward), plus the
     textbook exercises of scn-ita-eng; it lacks Dieli's translations of Pitre's tales, which
     his sheet mixes with the journal without a provenance column, so B has less data than A;
  C  A and B together (hand-aligned and automatic text as complements, deduplicated).

Every set is guarded against his test and against the validation lines used to stop the
training (the first VALID lines of our frozen validation set): a pair is dropped if either
side equals one of them (raw or std-normalized) or its Sicilian side contains a test
sentence of six or more words.

PRIVACY: the sheets are private; outputs go to data/recipe/ (gitignored) and to the private
Colab folder on Drive, never to the repository.

    python experiments/baseline/recipe_data.py \\
        --ods data/external/eryk/attachments/2026-09-07/ArbaSicula-Dieli_2024-10-20_Translation-Dataset.ods
"""
from __future__ import annotations
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments' / 'dataset'))
from normalize_scn import normalize  # noqa: E402

VALID = 500


def read(p):
    return Path(p).read_text(encoding='utf-8').splitlines()


def clean(s):
    return re.sub(r'\s+', ' ', str(s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ods', default=str(ROOT / 'data' / 'external' / 'eryk' / 'attachments' / '2026-09-07'
                                          / 'ArbaSicula-Dieli_2024-10-20_Translation-Dataset.ods'))
    ap.add_argument('--colab', default=str(Path.home() / 'GDrive' / 'SicilianNMT-colab'))
    ap.add_argument('--gift', default=str(ROOT / 'data' / 'processed' / 'as_full_gift' / 'corpus.tsv'))
    ap.add_argument('--min-vol', type=int, default=19)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'recipe'))
    a = ap.parse_args()
    import pandas as pd
    sheets = pd.read_excel(a.ods, engine='odf', sheet_name=['scn-eng', 'scn-ita-eng', 'validation'],
                           header=None)
    pairs = lambda df, i, j: [(clean(s), clean(t)) for s, t in zip(df[i], df[j])
                              if clean(s) and clean(t) and str(s) != 'nan' and str(t) != 'nan']
    hand = pairs(sheets['scn-eng'], 0, 1)
    textbook = pairs(sheets['scn-ita-eng'], 0, 2)
    test = pairs(sheets['validation'], 0, 2)
    valid = list(zip(read(f'{a.colab}/data/valid.scn'), read(f'{a.colab}/data/valid.en')))[:VALID]
    ours = []
    with open(a.gift, encoding='utf-8', newline='') as f:
        rows = csv.reader(f, delimiter='\t'); hdr = [h.strip() for h in next(rows)]
        bad = 0
        for r in rows:
            if len(r) != len(hdr):
                bad += 1; continue
            d = dict(zip(hdr, r))
            vol = int(re.match(r'as(\d+)', d['issue']).group(1))
            if vol >= a.min_vol:
                ours.append((clean(d['sicilian']), clean(d['english'])))
        print(f'our extraction: {len(ours)} pairs from AS{a.min_vol:02d} onward, {bad} malformed rows skipped')

    held = set()
    for s, t in test + valid:
        held |= {s, normalize(s, 'std'), t}
    long_test = [normalize(s, 'std') for s, _ in test if len(s.split()) >= 6]

    def guard(ps, name):
        kept, seen = [], set()
        for s, t in ps:
            n = normalize(s, 'std')
            if s in held or n in held or t in held or any(x in n for x in long_test):
                continue
            if (n, t) not in seen:
                seen.add((n, t)); kept.append((s, t))
        print(f'{name}: {len(kept)} pairs (from {len(ps)})')
        return kept

    sets = {'A': guard(hand + textbook, 'A'),
            'B': guard(ours + textbook, 'B'),
            'C': guard(hand + textbook + ours, 'C')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, ps in sets.items():
        (out / f'{name}.scn').write_text('\n'.join(s for s, _ in ps) + '\n', encoding='utf-8')
        (out / f'{name}.en').write_text('\n'.join(t for _, t in ps) + '\n', encoding='utf-8')
    for name, ps in (('valid', valid), ('test_eryk', test)):
        (out / f'{name}.scn').write_text('\n'.join(s for s, _ in ps) + '\n', encoding='utf-8')
        (out / f'{name}.en').write_text('\n'.join(t for _, t in ps) + '\n', encoding='utf-8')
    dst = Path(a.colab) / 'recipe'
    shutil.copytree(out, dst, dirs_exist_ok=True)
    print(f'written to {out} and {dst}')


if __name__ == '__main__':
    main()
