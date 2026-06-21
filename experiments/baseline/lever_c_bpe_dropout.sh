#!/bin/bash
# Lever C: BPE-dropout (subword regularization, Provilkov et al. 2020) on top of the
# lever-B tokenized + desinence-biased data. Builds K dropout-sampled copies of the
# training data (varied segmentations), clean BPE for valid/test. Run in MAIN env.
#
#   bash experiments/baseline/lever_c_bpe_dropout.sh data/baseline-tok-l data/baseline-bpedrop 4 0.1
#   # then, in .venv-sockeye:
#   bash experiments/baseline/02_prepare.sh data/baseline-bpedrop
#   bash experiments/baseline/train.sh      data/baseline-bpedrop
set -e
BASE=${1:-data/baseline-tok-l}        # must contain *.tok.{scn,en}, codes.bpe, vocab.{scn,en}
O=${2:-data/baseline-bpedrop}
K=${3:-4}                              # number of dropout-sampled training copies
P=${4:-0.1}                            # BPE-dropout probability
mkdir -p "$O"
cp "$BASE/codes.bpe" "$BASE/vocab.scn" "$BASE/vocab.en" "$O/"

# valid/test: deterministic (no dropout)
for split in valid test; do
  for lang in scn en; do
    subword-nmt apply-bpe -c "$BASE/codes.bpe" --vocabulary "$BASE/vocab.$lang" \
      --vocabulary-threshold 10 < "$BASE/$split.tok.$lang" > "$O/$split.bpe.$lang"
  done
done

# train: K copies, each with an independent dropout sampling (lines stay aligned per copy)
: > "$O/train.bpe.scn"; : > "$O/train.bpe.en"
for k in $(seq 1 "$K"); do
  for lang in scn en; do
    subword-nmt apply-bpe -c "$BASE/codes.bpe" --vocabulary "$BASE/vocab.$lang" \
      --vocabulary-threshold 10 --dropout "$P" < "$BASE/train.tok.$lang" >> "$O/train.bpe.$lang"
  done
done
echo "BPE-dropout (K=$K, p=$P) -> $O/{train,valid,test}.bpe.{scn,en}"
