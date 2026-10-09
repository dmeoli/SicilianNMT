# Sicilian NMT

[![Reproduce the pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/SicilianNMT/blob/main/sicilian_nmt.ipynb)

**[`sicilian_nmt.ipynb`](sicilian_nmt.ipynb)** runs the whole model pipeline end-to-end (data →
Standard-Sicilian preprocessing → NLLB-200 + LoRA → fine-tune → evaluate), narrated step by
step, calling the implementations in `experiments/*.py`.

A fork of Eryk Wdowiak's [_Tradutturi Sicilianu_](https://translate.napizia.com/),
the first neural machine translator for the Sicilian language
([paper](https://arxiv.org/abs/2110.01938), Springer
[DOI](https://doi.org/10.1007/978-3-031-10464-0_50)). This fork rebuilds the data
pipeline with maintained, CPU-only tooling and adds a reproducible dataset,
an evaluation harness, and baselines.

Upstream: [`ewdowiak/Sicilian_Translator`](https://github.com/ewdowiak/Sicilian_Translator).

## What's here

```
experiments/
  extraction/    PDF -> text -> page-language classify -> LaBSE confirm -> LaBSE sentence-align
                 (PyMuPDF + sentence-transformers; replaces the old pdflatex+hunalign Perl)
  tokenization/  pure-Python Sicilian/English tokenizer (port of the Napizia Perl module)
  dataset/       NLLB quality inspection, clean-subset builder, unified dataset assembly
  eval/          BLEU + chrF harness (sacrebleu)
  baseline/      Sockeye-3 (PyTorch, CPU) small Transformer + the paper's "lever B"
                 (tokenization + desinence-biased subwords)
  nllb/          NLLB-200 + LoRA engine (nllb_pipeline.py); driven by sicilian_nmt.ipynb
  serving/       FastAPI /translate + Telegram bot over the NLLB adapter
data/            gitignored: external/ (raw: arbasicula/pdf, arbasicula/perl_2023, wikimatrix,
                 flores, eryk), processed/, finetune/, recipe/
vocab/           Sicilian stopwords, Dieli/Chiù-dâ-Palora inflections (desinence bias), lemmas
docs/            Standard-Sicilian standardization & contraction references
papers/ presentation/
```

Two environments (kept isolated): **`.venv`** (CPython 3.12) for the data pipeline,
tokenization, evaluation and NLLB; **`.venv-sockeye`** (CPython 3.10) for Sockeye-3
training. See `experiments/baseline/README.md`.

## Data pipeline (all CPU)

1. **Extract** parallel text from Arba Sicula PDFs, with `experiments/extraction/build_all.py`
   (PyMuPDF + LaBSE). Recovers **~14k** sentence pairs from 44 issues, fully automated;
   thresholds tuned against Eryk Wdowiak's hand-aligned AS41-42 gold (`tune_vs_gold.py`).
2. **Add** the author's two public Napizia datasets, both included:
   [`Good-Sicilian-in-NLLB`](https://huggingface.co/datasets/Napizia/Good-Sicilian-in-NLLB)
   (the scored NLLB en–scn subset, further filtered here for quality and Corsican
   contamination) and
   [`Good-Sicilian-from-WikiMatrix`](https://huggingface.co/datasets/Napizia/Good-Sicilian-from-WikiMatrix)
   (curated it–scn), plus WikiMatrix it–scn.
3. **Assemble** a unified, deduped, split dataset, with `experiments/dataset/assemble.py`:
   **~29k scn–en** (train 27.4k, + a frozen 1k valid / 1k test held out from Arba Sicula =
   literary standard, not FLORES) + 11.4k it–scn.

## Results

All the numbers below are printed by the notebooks (`sicilian_nmt.ipynb` for NLLB,
`experiments/baseline/recipe_colab.ipynb` for Sockeye) from the results files of the saved
models, and are the ones of the paper (`docs/paper/`).

**Our test set** (1,000 literary pairs of *Arba Sicula*, BLEU / chrF). The Sockeye rows and the
floor are scored in the space of the Napizia tokenizer, the NLLB rows on raw text.

| model | scn→en | en→scn |
|---|---|---|
| floor (copy source) | 5.18 / 25.95 | – |
| Sockeye-3 baseline (web set) | 8.51 / 32.73 | – |
| + lever B (tokenization + desinence-biased BPE) | 10.94 / 34.66 | – |
| + lever D (lemma source factor) | 10.18 / 34.15 | – |
| NLLB-200 1.3B, out of the box | 29.00 / 55.23 | 9.90 / 41.09 |
| LoRA, bidirectional (web set) | 31.27 / 56.92 | 19.19 / 50.02 |
| + back-translation (7,498 pairs) | 31.60 / 57.21 | – |
| staged: Wdowiak's back-translations, then the curated set | 32.40 / 58.19 | 21.72 / 52.79 |
| curriculum (coarse stage first) | 33.01 / 58.62 | 21.78 / 52.94 |
| **final**: 3x back-translations, rank-64 adapter, 256 tokens, WikiMatrix it–scn, 4 epochs | **36.01 / 60.70** | **26.34 / 55.75** |

The final model is chosen on the 1,000 validation pairs among the variants of the ablation
(notebook §7.4).

**Wdowiak's test set** (the 121 trilingual lines of AS38-39), in his tokenized space (BLEU),
against his published systems:

| direction | his 2022 baseline | his reverse training | our final model |
|---|---|---|---|
| en→scn | 25.1 | 45.1 | **49.25** |
| scn→en | 29.1 | 48.6 | **59.34** |
| it→scn | – | **61.4** | 61.19 |
| scn→it | – | **62.9** | 60.30 |
| it→en | – | 48.2 | **58.42** |
| en→it | – | 46.7 | **48.77** |

He chose his checkpoints on this test set, while our model never saw its lines and was
chosen on our own validation set. His 2022 baseline, retrained with his configuration on his
sheets, our alignment and both (Table 6 of the paper), does not reach its published scores.

## References

We build on (PDFs and links in [`papers/`](papers/)): Wdowiak's *Recipe for Low-Resource NMT*;
the Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)); the low-resource
recipe ([Sennrich & Zhang 2019](https://arxiv.org/abs/1905.11901)) and subword splitting
([Sennrich et al. 2016](https://arxiv.org/abs/1508.07909)); linguistic input features
([Sennrich & Haddow 2016](https://arxiv.org/abs/1606.02892)); NLLB-200
([NLLB Team 2022](https://arxiv.org/abs/2207.04672)) and LASER3
([Heffernan et al. 2022](https://arxiv.org/abs/2205.12654)); LaBSE
([Feng et al. 2022](https://arxiv.org/abs/2007.01852)); LoRA
([Hu et al. 2021](https://arxiv.org/abs/2106.09685)); WikiMatrix
([Schwenk et al. 2021](https://arxiv.org/abs/1907.05791)); Sockeye
([Hieber et al. 2017](https://arxiv.org/abs/1712.05690)); plus BPE-dropout, back-translation,
multilingual zero-shot and beyond-English-centric MT. Full list in `papers/README.md`.

## License

Apache-2.0 (see `LICENSE`). Data sources retain their own terms (Arba Sicula; NLLB ODC-BY).
