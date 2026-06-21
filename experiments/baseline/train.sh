#!/bin/bash
# CPU baseline: small high-dropout Transformer (paper "our models" config), Sockeye 3.
# Run inside .venv-sockeye:  bash experiments/baseline/train.sh
set -e
O=${1:-data/baseline}
export OMP_NUM_THREADS=8

sockeye-train \
  --prepared-data $O/prep \
  --validation-source $O/valid.bpe.scn --validation-target $O/valid.bpe.en \
  --output $O/model-scn2en \
  --use-cpu \
  --shared-vocab \
  --seed 13 \
  --max-seq-len 100 \
  --batch-size 1024 --batch-type word \
  --max-num-epochs 30 \
  --checkpoint-interval 500 \
  --max-num-checkpoint-not-improved 8 \
  --encoder transformer --decoder transformer \
  --num-layers 3 \
  --num-embed 256 \
  --transformer-model-size 256 \
  --transformer-attention-heads 4 \
  --transformer-feed-forward-num-hidden 1024 \
  --embed-dropout 0.4 \
  --transformer-dropout-attention 0.2 \
  --transformer-dropout-act 0.2 \
  --transformer-dropout-prepost 0.2 \
  --label-smoothing 0.1 \
  --optimizer adam \
  --initial-learning-rate 0.0002 \
  --decode-and-evaluate 300
