#!/bin/bash
# Joint BPE subword splitting for the scn<->en baseline.
# Run in the MAIN env (.venv, has subword-nmt):  bash experiments/baseline/01_subword.sh [merges]
set -e
D=data/dataset
O=data/baseline
mkdir -p "$O"
MERGES=${1:-4000}

subword-nmt learn-joint-bpe-and-vocab --input "$D/train.scn" "$D/train.en" -s "$MERGES" \
  -o "$O/codes.bpe" --write-vocabulary "$O/vocab.scn" "$O/vocab.en"

for split in train valid test; do
  for lang in scn en; do
    subword-nmt apply-bpe -c "$O/codes.bpe" --vocabulary "$O/vocab.$lang" \
      --vocabulary-threshold 10 < "$D/$split.$lang" > "$O/$split.bpe.$lang"
  done
done
echo "BPE done -> $O/{train,valid,test}.bpe.{scn,en}"
