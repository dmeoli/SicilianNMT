# CPU baseline (Sockeye 3, PyTorch)

Faithful small high-dropout Transformer baseline on the unified dataset, trained
on CPU (the original paper also trained on CPU). Direction: **scn → en**.

## Two environments (the ML stack must stay isolated)

- **Main env** `.venv` (uv, CPython 3.12): data pipeline, BPE, evaluation.
- **Sockeye env** `.venv-sockeye` (uv, CPython 3.10): Sockeye 3.1.34 + torch 1.13 (CPU).
  Installed separately because Sockeye pins old torch; `uv pip install sockeye` alone
  resolves to the dead **MXNet** Sockeye 2.x, so we force `'sockeye>=3,<4'`.

## Pipeline

```bash
# 1. subword (main env)
source .venv/bin/activate
bash experiments/baseline/01_subword.sh 4000

# 2. prepare + 3. train (sockeye env)
source .venv-sockeye/bin/activate
bash experiments/baseline/02_prepare.sh
bash experiments/baseline/train.sh        # writes data/baseline/model-scn2en

# 4. translate test + evaluate (see 04_translate_eval.sh once training is done)
```

## Model (paper "our models" config)

3 layers, embed/model 256, 4 heads, FF 1024, shared embeddings (weight tying),
dropout 0.4 (embed) / 0.2 (attention/act/prepost), label smoothing 0.1, Adam lr 2e-4,
joint BPE 4000, ~6.6M params. Early stopping on validation perplexity.

Reference floor (copy source as translation): scn→en BLEU 5.27 / chrF 25.40.
All data under `data/baseline/` is gitignored.
