#!/bin/bash
# Lever B prep: paper-style Sicilian tokenization + desinence-biased subwords.
# Run in the MAIN env (.venv: subword-nmt + perl):  bash experiments/baseline/lever_b_prep.sh
set -e
D=data/dataset
O=data/baseline-tok
mkdir -p "$O"
TOK="perl experiments/baseline/tokenize.pl"
export PERL5LIB=perl-module

# 1. tokenize splits (sc with sc_tokenizer, en with en_tokenizer)
for split in train valid test; do
  $TOK sc < "$D/$split.scn" > "$O/$split.tok.scn"
  $TOK en < "$D/$split.en"  > "$O/$split.tok.en"
done

# 2. tokenize the desinence word list (Dieli + Chiù dâ Palora inflections), dedup
$TOK sc < vocab/dieli-cchiu-vocab.txt | sort -u > "$O/desinences.tok.scn"

# 3. learn joint BPE with the desinences appended to the Sicilian side (the paper's bias:
#    each word once -> no effect on whole-word counts, big effect on subword merges)
cat "$O/train.tok.scn" "$O/desinences.tok.scn" > "$O/bpe_input.scn"
subword-nmt learn-joint-bpe-and-vocab --input "$O/bpe_input.scn" "$O/train.tok.en" -s 4000 \
  -o "$O/codes.bpe" --write-vocabulary "$O/vocab.scn" "$O/vocab.en"

# 4. apply BPE to the real split data
for split in train valid test; do
  subword-nmt apply-bpe -c "$O/codes.bpe" --vocabulary "$O/vocab.scn" --vocabulary-threshold 10 \
    < "$O/$split.tok.scn" > "$O/$split.bpe.scn"
  subword-nmt apply-bpe -c "$O/codes.bpe" --vocabulary "$O/vocab.en" --vocabulary-threshold 10 \
    < "$O/$split.tok.en" > "$O/$split.bpe.en"
done
echo "lever-B prep done -> $O/{train,valid,test}.bpe.{scn,en}"
