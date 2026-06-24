#!/usr/bin/env python3
"""Freeze an it-scn split (train/valid/test) from the WikiMatrix it-scn pairs,
so Italian directions can be evaluated comparably (like the frozen scn-en test).

    python experiments/dataset/split_itsc.py

Reads data/dataset/itsc.{it,scn}; writes itsc_{train,valid,test}.{it,scn} alongside.
Deterministic (seeded shuffle), deduped. 1000 test + 1000 valid, rest train.
"""
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data" / "dataset"


def read(p):
    return (D / p).read_text(encoding="utf-8").splitlines()


def write(name, rows, lang_idx):
    (D / name).write_text("\n".join(r[lang_idx] for r in rows) + "\n", encoding="utf-8")


def main():
    it, scn = read("itsc.it"), read("itsc.scn")
    assert len(it) == len(scn), f"length mismatch {len(it)} vs {len(scn)}"
    seen, pairs = set(), []
    for i, s in zip(it, scn):
        i, s = i.strip(), s.strip()
        if not i or not s or (i, s) in seen:
            continue
        seen.add((i, s))
        pairs.append((i, s))
    random.Random(13).shuffle(pairs)
    test, valid, train = pairs[:1000], pairs[1000:2000], pairs[2000:]
    for name, rows in [("test", test), ("valid", valid), ("train", train)]:
        write(f"itsc_{name}.it", rows, 0)
        write(f"itsc_{name}.scn", rows, 1)
    print(f"it-scn: {len(pairs)} unique pairs -> "
          f"train {len(train)} / valid {len(valid)} / test {len(test)}")
    print(f"wrote itsc_{{train,valid,test}}.{{it,scn}} to {D}/")


if __name__ == "__main__":
    main()
