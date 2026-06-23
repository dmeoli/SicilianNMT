#!/bin/bash
# Lever D: linguistic input features (lemma source factor, Sennrich W16-2209).
# Same small Transformer as train.sh, plus a lemma factor summed into the source
# embedding. Run in the SOCKEYE env after build_factors.py + prepare-data:
#   bash experiments/baseline/train_factors.sh data/baseline-tok-d
set -e
O=${1:-data/baseline-tok-d}
export OMP_NUM_THREADS=8

sockeye-train \
  --prepared-data "$O/prep-fac" \
  --validation-source "$O/valid.bpe.scn" --validation-source-factors "$O/valid.fac.scn" \
  --validation-target "$O/valid.bpe.en" \
  --output "$O/model-fac-scn2en" \
  --use-cpu --shared-vocab --seed 13 --max-seq-len 100 \
  --batch-size 1024 --batch-type word \
  --max-num-epochs 30 --checkpoint-interval 500 --max-num-checkpoint-not-improved 8 \
  --encoder transformer --decoder transformer \
  --num-layers 3 --num-embed 256 \
  --transformer-model-size 256 --transformer-attention-heads 4 \
  --transformer-feed-forward-num-hidden 1024 \
  --source-factors-num-embed 256 --source-factors-combine sum \
  --embed-dropout 0.4 --transformer-dropout-attention 0.2 \
  --transformer-dropout-act 0.2 --transformer-dropout-prepost 0.2 \
  --label-smoothing 0.1 --optimizer adam --initial-learning-rate 0.0002 \
  --decode-and-evaluate 300
