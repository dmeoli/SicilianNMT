# Sicilian NMT

[![Reproduce the pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/SicilianNMT/blob/main/reproduce.ipynb)

**[`reproduce.ipynb`](reproduce.ipynb)** runs the whole model pipeline end-to-end — data →
Standard-Sicilian preprocessing → NLLB-200 + LoRA → fine-tune → evaluate — narrated step by
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
  nllb/          NLLB-200 + LoRA engine (nllb_pipeline.py); driven by reproduce.ipynb
  serving/       FastAPI /translate + Telegram bot over the NLLB adapter
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
2. **Add** the author's two public Napizia datasets — both included:
   [`Good-Sicilian-in-NLLB`](https://huggingface.co/datasets/Napizia/Good-Sicilian-in-NLLB)
   (the scored NLLB en–scn subset, further filtered here for quality and Corsican
   contamination) and
   [`Good-Sicilian-from-WikiMatrix`](https://huggingface.co/datasets/Napizia/Good-Sicilian-from-WikiMatrix)
   (curated it–scn), plus WikiMatrix it–scn.
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
| \;+ more data (22k, NLLB threshold 3.0) | 9.79 | 33.82 |
| \;+ lever D (27k + lemma source factors) | **10.85** | **35.22** |
| NLLB-200 distilled-600M, zero-shot † | 25.63 | 52.53 |
| NLLB-200 distilled-600M, LoRA fine-tuned † | 28.93 | 55.12 |
| NLLB-200 1.3B, zero-shot † | 29.02 | 55.23 |
| NLLB-200 1.3B, LoRA fine-tuned, scn→en only (27k) † | 31.16 | 56.79 |
| NLLB-200 1.3B, LoRA **bidirectional** (27k) † | 31.43 | 56.94 |
| \;+ back-translation (7.5k synthetic) † | **31.60** | **57.21** |

Each Sockeye lever stacks (tokenization + desinences +1.7, more data +2.55: 5.54→9.79).
The modern pretrained model wins decisively: NLLB-200 zero-shot scn→en 25.63 (600M) / 29.02
(1.3B), and LoRA fine-tuning on our ~27k train lifts the 1.3B to **31.43 BLEU** — above
Wdowiak's published *baseline* (Sc→En 29.1) on our harder held-out literary test set, **but
well below his reverse-training state of the art** (see below).

**en→scn** (the reverse direction) on the same test set: NLLB-1.3B zero-shot 9.89 →
**bidirectional LoRA 18.73 BLEU / 49.96 chrF** — the bidirectional fine-tune nearly doubles
the weak direction at no cost to scn→en, and yields a usable two-way model. Still below the
paper's en→scn baseline (25.1); back-translation is the next lever.

**Italian (trilingual model).** One multilingual LoRA adapter fine-tuned on four directions
(scn↔en, it↔scn). On the frozen WikiMatrix it–scn test, Italian is the easiest pair:

| direction | zero-shot | multilingual FT |
|---|---|---|
| it→scn | 17.61 / 44.77 | **26.97 / 51.69** |
| scn→it | 42.20 / 59.04 | **43.47 / 59.79** |

It costs a little on scn↔en (30.75 / 17.49 vs the dedicated 31.43 / 18.73), so we can ship the
trilingual all-rounder or the specialised pair. The site and Telegram bot serve all three
languages (scn/en/it).

† NLLB rows are evaluated on raw text; the Sockeye rows are tokenized space (raw floor is
5.27 BLEU). Even allowing for that, the 600M pretrained model far exceeds the 6.6M baseline.

### We do not beat the state of the art (yet)

The SOTA is Wdowiak's **reverse-training** system, on his own in-domain test set (BLEU):

| | En→Sc | Sc→En | It→Sc | Sc→It |
|---|---|---|---|---|
| paper baseline | 25.1 | 29.1 | — | — |
| + backtranslation + multilingual | 35.0 | 36.8 | 36.5 | 30.9 |
| **reverse-training (his best)** | **45.1** | **48.6** | **61.4** | **62.9** |
| **ours** (NLLB+LoRA + back-transl.) | 18.7 | 31.6 | 27.0 | 43.5 |

⚠️ **Not a head-to-head** (different test sets), but the gap is too large to be that alone:
we only clear his *baseline* on Sc→En, and trail on every other direction. His reverse-training
builds a **custom pretrained model from tens of millions of pairs** (forward- and
back-translation, three stages) before fine-tuning on hand-curated *Arba Sicula* — infeasible
to reproduce on one GPU. But **NLLB-200 already provides that pretraining**, so our run is the
analogue of his *stage-3 fine-tune*. The plan: apply the feasible parts of his recipe on top of
NLLB (back-translation, multilingual) to isolate **method vs data** — if, with the method
matched, our automatic corpus still trails his curated one, the residual is the **data**, which
argues for combining his corpus with our pipeline rather than competing.

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
