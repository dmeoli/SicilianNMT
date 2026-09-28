#!/usr/bin/env python3
"""Would our LaBSE aligner work on the other language varieties of Italy?

The aligner works on Sicilian although LaBSE does not support it, because LaBSE places
Sicilian next to Italian (labse_why_sicilian.py). This script asks the same question for
the varieties of Italy in FLORES-200 (Friulian, Ligurian, Lombard, Sardinian, Venetian,
with Sicilian as the reference case), and for supported Romance languages as a control, on
the multi-parallel devtest (1012 sentences aligned by line). For each language X it reports

  1. X->en and en->X retrieval precision@1 (the nearest neighbour is the true translation);
  2. the share of true X-en pairs whose cosine clears the sentence threshold of our aligner
     (tau_sent = 0.40), i.e. the pairs the aligner could keep;
  3. the mean cosine of the true X-en pairs;
  4. excluding English, the language whose true translation is most often the nearest one
     to a sentence of X (the "anchor" through which LaBSE reads X), with its share.

    OMP_NUM_THREADS=2 python experiments/extraction/labse_varieties.py
"""
from __future__ import annotations
import collections
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
FLORES = REPO / "data/external/flores/flores200_dataset/devtest"
OUT = REPO / "data/processed/analysis/labse_varieties.tsv"
VARIETIES = {"scn": "scn_Latn", "fur": "fur_Latn", "lij": "lij_Latn", "lmo": "lmo_Latn",
             "srd": "srd_Latn", "vec": "vec_Latn"}
CONTROLS = {"it": "ita_Latn", "es": "spa_Latn", "fr": "fra_Latn", "ca": "cat_Latn",
            "pt": "por_Latn", "ro": "ron_Latn", "oc": "oci_Latn", "gl": "glg_Latn"}
TAU_SENT = 0.40


def load(code):
    return (FLORES / f"{code}.devtest").read_text(encoding="utf-8").splitlines()


def main():
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    langs = {**VARIETIES, **CONTROLS, "en": "eng_Latn"}
    emb = {k: model.encode(load(v), batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False) for k, v in langs.items()}
    n = len(emb["en"])
    rows = []
    for x in list(VARIETIES) + list(CONTROLS):
        sim = emb[x] @ emb["en"].T
        p_xe = float(np.mean(sim.argmax(1) == np.arange(n)))
        p_ex = float(np.mean(sim.argmax(0) == np.arange(n)))
        true = np.diag(sim)
        others = [k for k in langs if k not in (x, "en")]
        nearest = collections.Counter(
            others[int(np.argmax([emb[x][i] @ emb[k][i] for k in others]))] for i in range(n))
        anchor, cnt = nearest.most_common(1)[0]
        rows.append((x, p_xe, p_ex, float(np.mean(true >= TAU_SENT)), float(true.mean()),
                     anchor, cnt / n))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("lang\tP@1_x2en\tP@1_en2x\tshare_cos>=0.40\tmean_cos\tanchor\tanchor_share\n")
        for r in rows:
            f.write("\t".join([r[0]] + [f"{v:.3f}" for v in r[1:5]] + [r[5], f"{r[6]:.3f}"]) + "\n")
    print(f"{'lang':5s} {'x->en':>6s} {'en->x':>6s} {'>=0.40':>7s} {'cos':>6s}  anchor")
    for r in rows:
        print(f"{r[0]:5s} {r[1]:6.1%} {r[2]:6.1%} {r[3]:7.1%} {r[4]:6.3f}  {r[5]} ({r[6]:.0%})")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
