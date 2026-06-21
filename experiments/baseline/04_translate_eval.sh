#!/bin/bash
# Translate the test set with the trained model, de-BPE, then evaluate.
# Translate in the SOCKEYE env; evaluate in the MAIN env.
set -e
O=data/baseline

# --- in .venv-sockeye ---
sockeye-translate --input "$O/test.bpe.scn" --output "$O/test.hyp.bpe.en" \
  --models "$O/model-scn2en" --use-cpu --beam-size 5 --batch-size 16
sed -r 's/(@@ )|(@@ ?$)//g' "$O/test.hyp.bpe.en" > "$O/test.hyp.en"

echo "Wrote $O/test.hyp.en. Now evaluate in the MAIN env (.venv):"
echo "  python experiments/eval/evaluate.py --hyp $O/test.hyp.en --ref data/dataset/test.en --tag sockeye-scn2en"
