# Sicilian NMT — modernized pipeline

[![Open NLLB notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/SicilianNMT/blob/main/experiments/nllb/sicilian_nllb_colab.ipynb)

A reproduction and modernization of Eryk Wdowiak's [_Tradutturi Sicilianu_](https://translate.napizia.com/),
the first neural machine translator for the Sicilian language
([paper](https://arxiv.org/abs/2110.01938), Springer
[DOI](https://doi.org/10.1007/978-3-031-10464-0_50)). This fork rebuilds the data
pipeline with modern, maintained, CPU-friendly tooling and a reproducible dataset,
evaluation harness, and baselines.

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
  nllb/          NLLB-200 zero-shot / fine-tune on Colab (Sicilian = scn_Latn)
training/ translations/   upstream's Reverse-Training (Sockeye-3) scripts + recorded scores
fastapi/         upstream's FastAPI translation server
extract-text/    Arba Sicula PDFs (gitignored) + WikiMatrix it-scn + aligned gold CSVs
vocab/           Sicilian stopwords, Dieli/Chiù-dâ-Palora inflections (desinence bias), lemmas
docs/            Standard-Sicilian standardization & contraction references
papers/ presentation/
```

Two environments (kept isolated): **`.venv`** (CPython 3.12) for the data pipeline,
tokenization, evaluation and NLLB; **`.venv-sockeye`** (CPython 3.10) for Sockeye-3
training. See `experiments/baseline/README.md`.

## Data pipeline (all CPU)

1. **Extract** parallel text from Arba Sicula PDFs — `experiments/extraction/build_all.py`
   (PyMuPDF + LaBSE). Recovers ~9.4k sentence pairs from 44 issues, fully automated,
   surfacing parallel text the original hand-curated process missed.
2. **Add** the author's curated `Napizia/Good-Sicilian-in-NLLB` (filtered for quality and
   Corsican contamination) and WikiMatrix it-scn.
3. **Assemble** a unified, deduped, split dataset — `experiments/dataset/assemble.py`:
   **13.3k scn–en** (train/valid/test, test+valid held out from Arba Sicula = literary
   standard, not FLORES) + 11k it–scn.

## Results so far (our held-out test set, scn→en, BLEU/chrF)

| model | BLEU | chrF |
|---|---|---|
| floor (copy source) | 5.27 | 25.40 |
| Sockeye-3 baseline (raw) | 5.54 | 28.28 |
| Sockeye-3 + lever B (tokenization + desinences) | **7.24** | **29.52** |

Lever B confirms the paper's recipe (+1.7 BLEU). Absolute scores are low — small,
noisy data and a hard literary test set; the modern model (NLLB) and more data are the
next levers. Numbers are **not** comparable to the paper's (different, harder test set).

## License

Apache-2.0 (see `LICENSE`). Data sources retain their own terms (Arba Sicula; NLLB ODC-BY).
