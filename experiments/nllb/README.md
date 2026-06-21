# NLLB-200 on Sicilian (Colab GPU)

Meta's **NLLB-200** already covers Sicilian (`scn_Latn`), so we can get a strong
modern reference number on our test set with **zero training**, then optionally
fine-tune. Same BLEU/chrF metrics as the local Sockeye baseline → directly comparable.

Sicilian = `scn_Latn`, English = `eng_Latn`, Italian = `ita_Latn`.

## Quick path (Colab)

1. Open `nllb_colab.ipynb` in Colab; set Runtime → GPU (T4 is enough for 600M).
2. Run cells; upload `data/dataset/test.scn` and `data/dataset/test.en` when prompted.
3. It prints BLEU/chrF for scn→en and en→scn.

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
