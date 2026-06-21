#!/bin/bash
# Sockeye-3 data preparation (scn->en). Shared vocab because BPE was learned jointly.
# Run in the SOCKEYE env (.venv-sockeye):  bash experiments/baseline/02_prepare.sh
set -e
O=data/baseline
sockeye-prepare-data --source "$O/train.bpe.scn" --target "$O/train.bpe.en" \
  --output "$O/prep" --max-seq-len 100 --shared-vocab
