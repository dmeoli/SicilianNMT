# Low-resource NMT methods, survey & experiment plan

Methods to push Sicilian↔English BLEU beyond the current baselines, ranked by
expected gain vs effort/compute. All evaluated on our fixed held-out test set with
`experiments/eval/evaluate.py` (BLEU + chrF), so numbers stay comparable.

Current state (scn→en, tokenized eval): floor 5.27 · raw baseline 5.54 · **lever-B 7.24**.

## What the original paper already used (and we reproduced)

Sennrich & Zhang (2019) low-resource recipe: small high-dropout Transformer, reduced
BPE vocab, + the paper's own tricks (ASCII-fold/uncontract tokenization, desinence-biased
subwords). This is our **lever B** (+1.7 BLEU). Plus backtranslation + multilingual
(Italian bridge) + Reverse-Training in later work.

## Ranked experiment menu

### 1. NLLB-200 fine-tuning (LoRA), STRONGEST, Colab GPU
Pretrained on 200 langs incl. `scn_Latn`. Literature: +33% BLEU typical on tiny corpora
(e.g. 19.8→26.4), preserves multilingual knowledge, zero inference latency once merged.
- Effort: low (script ready, `experiments/nllb/nllb_finetune.py`). Compute: Colab T4/A100.
- Start from zero-shot eval (already set up) → LoRA fine-tune on our `train` → re-eval.
- Risk: NLLB's seed orthography differs from our literary standard (chrF penalty), 
  fine-tuning on our data is exactly the fix.

### 2. BPE-dropout (subword regularization), CHEAP, CPU
Provilkov et al. (2020): stochastically drop BPE merges so the model sees varied
segmentations → robustness, esp. low/medium resource. Apply with `subword-nmt --dropout`.
- Effort: low (`experiments/baseline/lever_c_bpe_dropout.sh`). Compute: same CPU train.
- Queue right after the 22k run; expected modest but reliable gain.

### 3. Back-translation (tagged), MEDIUM, Colab + CPU
Use NLLB to translate monolingual text into synthetic parallel data, tag synthetic
sources, mix with real. We have monolingual Sicilian (unused AS pages, the 1M NLLB scn
side) and English. Most effective low-resource augmentation per the surveys.
- Effort: medium (`experiments/nllb/backtranslate.py` scaffold). Manage noise ratio.

### 4. More / cleaner data, IN PROGRESS
Already running: enlarge NLLB threshold 2.0→3.0 (train 11k→22k). Also possible: lower the
AS alignment min-sim to recover more in-domain literary pairs; the early-issue glossary
extractor for dictionary-style pairs.

### 5. Transfer learning from high-resource Romance, MEDIUM/HIGH
Pre-train scn↔en jointly with it↔en / es↔en (linguistically close), then fine-tune on
scn. Or the paper's multilingual m2m with the Italian bridge (we have 11k it-scn).
- Effort: medium-high. Uses our it-scn WikiMatrix data.

### 6. LLM few-shot / fine-tune, EXPLORATORY
Prompt a strong multilingual LLM with dictionary + few examples, or fine-tune a small LLM.
Useful as an alternative baseline; dictionary-constrained decoding can help rare words.

## Recommended order

1. (running) enlarge data → measure.
2. **NLLB LoRA fine-tune** (Colab), likely the biggest single jump.
3. **BPE-dropout** on the best Sockeye config (CPU, queueable now).
4. **Back-translation** with the fine-tuned NLLB.
5. Multilingual/transfer (Italian bridge) if we push further.

Each result goes into the comparison table (README + the LaTeX paper) vs the floor and the
author's published numbers (noting test-set differences).

## Sources
- Sennrich & Zhang (2019), *Revisiting Low-Resource NMT*, arXiv 1905.11901
- Provilkov et al. (2020), *BPE-Dropout*, ACL
- Haddow et al. (2022), *Survey of Low-Resource MT*, Computational Linguistics 48(3)
- NLLB Team (2022) + LoRA fine-tuning case studies (2024–2025)
