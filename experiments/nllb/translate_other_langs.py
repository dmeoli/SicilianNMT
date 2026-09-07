#!/usr/bin/env python3
"""How well does our trilingual adapter translate OTHER languages to/from Sicilian?

Eryk's curiosity (2026-09): the model was fine-tuned only on scn<->en and it<->scn, but
Sicilian sits in NLLB's Romance region, so zero-shot Spanish/French/Portuguese <-> Sicilian
might already be decent. We test on FLORES-200 devtest (multi-parallel, includes scn_Latn):
translate es/fr/pt <-> scn with the adapted model and score BLEU/chrF against the references.

GPU-oriented; on CPU use a subset (--n) and greedy decoding (--beams 1). For a full run,
use Colab (add this to sicilian_nmt.ipynb like the FLORES it<->en cell).

    python experiments/nllb/translate_other_langs.py --n 200 --beams 1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import util  # noqa: F401  (ensures torch stack is importable)

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/nllb"))
from nllb_pipeline import load_base, score  # reuse loader + sacreBLEU scorer

FLORES = REPO / "data/external/flores/flores200_dataset/devtest"
ADAPTER = Path("/home/david/Insync/donato.meoli.95@gmail.com/Google Drive/"
               "SicilianNMT-colab/nllb-lora-multi-1.3B")
CODE = {"scn": "scn_Latn", "es": "spa_Latn", "fr": "fra_Latn", "pt": "por_Latn",
        "en": "eng_Latn", "it": "ita_Latn"}
PAIRS = [("es", "scn"), ("fr", "scn"), ("pt", "scn"),
         ("scn", "es"), ("scn", "fr"), ("scn", "pt")]


def load(code: str, n: int) -> list[str]:
    return (FLORES / f"{code}.devtest").read_text(encoding="utf-8").splitlines()[:n]


@torch.no_grad()
def translate(model, tok, texts, src, tgt, bs, max_len, beams):
    tok.src_lang = CODE[src]
    tgt_id = tok.convert_tokens_to_ids(CODE[tgt])
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(dev)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_len, num_beams=beams)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="sentences per direction (CPU: keep small)")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/analysis")
    args = ap.parse_args()

    if not ADAPTER.exists():
        raise SystemExit(f"adapter not found at {ADAPTER}")
    print(f"loading NLLB-1.3B + adapter (device: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}) ...", flush=True)
    model, tok = load_base("facebook/nllb-200-1.3B", adapter=str(ADAPTER))

    results = {}
    for s, t in PAIRS:
        hyp = translate(model, tok, load(CODE[s], args.n), s, t, args.bs, 128, args.beams)
        results[f"{s}->{t}"] = score(hyp, load(CODE[t], args.n))
        print(f"  {s}->{t}:  BLEU={results[f'{s}->{t}'][0]}  chrF={results[f'{s}->{t}'][1]}",
              flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"n": args.n, "beams": args.beams, "adapter": ADAPTER.name, "results": results}
    (args.out / "other_langs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'other_langs.json'}  (n={args.n}, beams={args.beams})")


if __name__ == "__main__":
    main()
