# NLLB-200 on Sicilian (Colab GPU)

Meta's **NLLB-200** already covers Sicilian (`scn_Latn`), so we can get a strong
modern reference number on our test set with **zero training**, then optionally
fine-tune. Same BLEU/chrF metrics as the local Sockeye baseline → directly comparable.

Sicilian = `scn_Latn`, English = `eng_Latn`, Italian = `ita_Latn`.

## Quick path (Colab)

Open **`sicilian_nllb_colab.ipynb`** (all-in-one: zero-shot + LoRA fine-tune) — use the
"Open in Colab" badge in the top-level README, or open it manually. Set Runtime → GPU,
run top to bottom, and upload the six `data/dataset/{train,valid,test}.{scn,en}` files
when prompted. It prints zero-shot and fine-tuned BLEU/chrF for both directions.

`nllb_eval.py` / `nllb_finetune.py` are the standalone script equivalents.

Models: `facebook/nllb-200-distilled-600M` (fast, fits free T4) → `...-1.3B` →
`...-3.3B` (more VRAM). Bigger = better, slower.

## Script path (any CUDA box)

```bash
pip install transformers sentencepiece sacrebleu torch
python nllb_eval.py --src data/dataset/test.scn --ref data/dataset/test.en \
    --src-lang scn_Latn --tgt-lang eng_Latn --out hyp.en
python nllb_eval.py --src data/dataset/test.en  --ref data/dataset/test.scn \
    --src-lang eng_Latn --tgt-lang scn_Latn
```

## Caveat (important for interpretation)

NLLB's own Sicilian training/eval used a post-2017 orthography that differs from the
Arba Sicula literary standard our test set uses (see the Napizia "Good-Sicilian-in-NLLB"
notes). So zero-shot NLLB may be penalised on chrF/BLEU for orthographic mismatch even
when meaning is right — fine-tuning on our `train` data should help a lot.
