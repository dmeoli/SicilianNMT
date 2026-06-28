# NLLB-200 on Sicilian

Meta's **NLLB-200** already covers Sicilian (`scn_Latn`), so we adapt it with LoRA rather
than training from scratch. Sicilian = `scn_Latn`, English = `eng_Latn`, Italian = `ita_Latn`.

## Code

- **`nllb_pipeline.py`** — the reusable engine: `load_base`, `attach_lora`, `build_dataset`,
  `finetune`, `translate`, `score`. Everything else (the notebook, any script) calls these.

## How to run it

The whole pipeline — data → preprocessing → fine-tune → evaluate, plus the optional levers
(Italian bridge, back-translation, normalization ablation) — is one narrated notebook at the
repo root: **[`sicilian_nmt.ipynb`](../../sicilian_nmt.ipynb)** (Colab badge in the top-level README).
It imports `nllb_pipeline` and reads the prepared data from Drive.

Minimal programmatic use:

```python
from nllb_pipeline import load_base, attach_lora, build_dataset, finetune, translate, score
model, tok = load_base('facebook/nllb-200-1.3B')
ft = attach_lora(model)
ds = build_dataset(tok, [(train_scn, train_en, 'scn', 'en'), (train_en, train_scn, 'en', 'scn')])
finetune(ft, tok, ds, out_dir='nllb-lora-bidir', epochs=2)
print(score(translate(ft, tok, test_scn, 'scn', 'en'), test_en))
```

Models: `facebook/nllb-200-distilled-600M` (fast, free T4) → `...-1.3B` (our default) →
`...-3.3B` (more VRAM).

## Caveat (important for interpretation)

NLLB's own Sicilian training used a post-2017 orthography that differs from the Arba Sicula
literary standard our test set uses. Zero-shot NLLB may be penalised on BLEU/chrF for
orthographic mismatch even when meaning is right — fine-tuning on our `train` data helps a lot.
We also adopt light Standard-Sicilian normalization (`../dataset/normalize_scn.py`).
