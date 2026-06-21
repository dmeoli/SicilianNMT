#!/usr/bin/env python3
"""Zero-shot (and optionally fine-tuned) NLLB-200 evaluation on our Sicilian test set.

NLLB-200 already supports Sicilian as `scn_Latn`. This translates a source file and
scores it against a reference with BLEU + chrF (same metrics as the local harness),
so the number is comparable to the Sockeye baseline.

Designed for Colab GPU but runs anywhere with CUDA (or CPU, slowly).

    python nllb_eval.py --src test.scn --ref test.en --src-lang scn_Latn --tgt-lang eng_Latn
    python nllb_eval.py --src test.en  --ref test.scn --src-lang eng_Latn --tgt-lang scn_Latn \
        --model facebook/nllb-200-1.3B
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF


def translate(texts, tok, model, src_lang, tgt_lang, device, batch_size=32, max_len=160):
    tok.src_lang = src_lang
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                 max_length=max_len, num_beams=5)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  translated {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--src-lang", required=True, help="e.g. scn_Latn / eng_Latn / ita_Latn")
    ap.add_argument("--tgt-lang", required=True)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--out", type=Path, help="write hypotheses here")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    src = args.src.read_text(encoding="utf-8").splitlines()
    ref = args.ref.read_text(encoding="utf-8").splitlines()
    hyp = translate(src, tok, model, args.src_lang, args.tgt_lang, device, args.batch_size)

    if args.out:
        args.out.write_text("\n".join(hyp) + "\n", encoding="utf-8")

    bleu = BLEU().corpus_score(hyp, [ref])
    chrf = CHRF().corpus_score(hyp, [ref])
    print(f"\n{args.src_lang}->{args.tgt_lang}  n={len(hyp)}  "
          f"BLEU={bleu.score:.2f}  chrF={chrf.score:.2f}")


if __name__ == "__main__":
    main()
