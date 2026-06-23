# Sicilian NMT

[![Open NLLB notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/SicilianNMT/blob/main/experiments/nllb/sicilian_nllb_colab.ipynb)

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
  nllb/          NLLB-200 zero-shot / fine-tune on Colab (Sicilian = scn_Latn)
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
   (PyMuPDF + LaBSE). Recovers **~14k** sentence pairs from 44 issues, fully automated;
   thresholds tuned against Eryk Wdowiak's hand-aligned AS41-42 gold (`tune_vs_gold.py`).
2. **Add** the author's curated `Napizia/Good-Sicilian-in-NLLB` (filtered for quality and
   Corsican contamination), `Good-Sicilian-from-WikiMatrix`, and WikiMatrix it-scn.
3. **Assemble** a unified, deduped, split dataset — `experiments/dataset/assemble.py`:
   **~29k scn–en** (train 27.4k, + a frozen 1k valid / 1k test held out from Arba Sicula =
   literary standard, not FLORES) + 11.4k it–scn.

## Results

**Our models, on our held-out test set** (1000 Arba-Sicula literary pairs), scn→en:

| model | BLEU | chrF |
|---|---|---|
| floor (copy source) | 5.27 | 25.40 |
| Sockeye-3 baseline | 5.54 | 28.28 |
| Sockeye-3 + lever B (tokenization + desinences) | 7.24 | 29.52 |
| \;+ more data (22k, NLLB threshold 3.0) | **9.79** | **33.82** |
| NLLB-200 distilled-600M, zero-shot † | 25.63 | 52.53 |
| NLLB-200 distilled-600M, LoRA fine-tuned † | **28.93** | **55.12** |

Each Sockeye lever stacks (tokenization + desinences +1.7, more data +2.55: 5.54→9.79).
The modern pretrained model wins decisively: **NLLB-200 zero-shot scn→en 25.63 BLEU**
(en→scn 9.64), and **LoRA fine-tuning lifts it to 28.93** (+3.3) — essentially matching the
paper's published *baseline* (Sc→En 29.1), but on our harder held-out literary test set.

† NLLB rows are evaluated on raw text; the Sockeye rows are tokenized space (raw floor is
5.27 BLEU). Even allowing for that, the 600M pretrained model far exceeds the 6.6M baseline.

**Wdowiak's published numbers, on his own test set** (BLEU):

| | En→Sc | Sc→En |
|---|---|---|
| paper baseline | 25.1 | 29.1 |
| + backtranslation + multilingual | 35.0 | 36.8 |
| Reverse-Training (his best) | 45.1 | 48.6 |

⚠️ **Not a head-to-head.** His numbers are on a different, in-domain, hand-selected test
set with his full recipe (backtranslation, multilingual, bigger model, curated data); ours
is a small baseline on a harder held-out literary test set. Different rulers — a fair
comparison requires running one model on the other's test set, or reproducing his full
recipe on our data. Closing that gap on *our* test set is the point of the next experiments.

## License

Apache-2.0 (see `LICENSE`). Data sources retain their own terms (Arba Sicula; NLLB ODC-BY).
