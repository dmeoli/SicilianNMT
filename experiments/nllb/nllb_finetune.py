#!/usr/bin/env python3
"""LoRA fine-tuning of NLLB-200 on our Sicilian-English data (Colab GPU).

Parameter-efficient fine-tuning (LoRA via peft) adapts NLLB to the Arba-Sicula
literary standard while keeping the base weights frozen. Literature reports ~+33%
BLEU on tiny corpora. Evaluate with the same harness for a comparable number.

Colab:
    !pip install transformers datasets peft sentencepiece sacrebleu accelerate
    # upload train.scn/train.en/valid.scn/valid.en/test.scn/test.en (data/dataset/)
    python nllb_finetune.py --direction scn2en --epochs 3

After training it prints test BLEU/chrF and saves the LoRA adapter.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

LANG = {"scn": "scn_Latn", "en": "eng_Latn"}


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=["scn2en", "en2scn"], default="scn2en")
    ap.add_argument("--data", type=Path, default=Path("."))
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()

    src, tgt = args.direction.split("2")
    src_lang, tgt_lang = LANG[src], LANG[tgt]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=src_lang, tgt_lang=tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "v_proj"], task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def build(split):
        s = read(args.data / f"{split}.{src}")
        t = read(args.data / f"{split}.{tgt}")
        return Dataset.from_dict({"src": s, "tgt": t})

    def tokenize(batch):
        enc = tok(batch["src"], text_target=batch["tgt"],
                  max_length=args.max_len, truncation=True)
        return enc

    train_ds = build("train").map(tokenize, batched=True, remove_columns=["src", "tgt"])
    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=f"nllb-lora-{args.direction}", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        fp16=(device == "cuda"), logging_steps=50, save_strategy="no", report_to=[])
    Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                   data_collator=collator).train()
    model.save_pretrained(f"nllb-lora-{args.direction}")

    # evaluate on test
    model.eval().to(device)
    tok.src_lang = src_lang
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    test_src, test_ref = read(args.data / f"test.{src}"), read(args.data / f"test.{tgt}")
    hyp = []
    for i in range(0, len(test_src), args.batch_size):
        enc = tok(test_src[i:i + args.batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                 max_length=args.max_len, num_beams=5)
        hyp += tok.batch_decode(gen, skip_special_tokens=True)
    print(f"\n{args.direction} (LoRA fine-tuned)  "
          f"BLEU={BLEU().corpus_score(hyp, [test_ref]).score:.2f}  "
          f"chrF={CHRF().corpus_score(hyp, [test_ref]).score:.2f}")


if __name__ == "__main__":
    main()
