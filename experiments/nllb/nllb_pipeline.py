"""Reusable NLLB-200 fine-tuning / translation pipeline.

The scattered Colab notebooks duplicated this logic; it lives here once, so the master
`reproduce.ipynb` (and any script) just calls these functions. GPU-oriented but CPU-safe.

Typical use:
    from nllb_pipeline import load_base, attach_lora, build_dataset, finetune, translate, score
    model, tok = load_base()
    ft = attach_lora(model)
    ds = build_dataset(tok, [(train_scn, train_en, 'scn', 'en'),
                             (train_en, train_scn, 'en', 'scn')])   # bidirectional
    finetune(ft, tok, ds, out_dir='adapter-bidir', epochs=2)
    hyp = translate(ft, tok, test_scn, 'scn', 'en')
    bleu, chrf = score(hyp, test_en)
"""
from __future__ import annotations
import gc
import os

import torch

LANG = {"scn": "scn_Latn", "en": "eng_Latn", "it": "ita_Latn"}
DEFAULT_MODEL = "facebook/nllb-200-1.3B"


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base(model_id: str = DEFAULT_MODEL, adapter: str | None = None):
    """Load NLLB (fp16 on GPU, OOM-safe) and optionally merge a trained LoRA adapter."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _free()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval(), tok


def attach_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    """Wrap with a LoRA adapter on the attention projections; keep adapter params in fp32."""
    from peft import LoraConfig, get_peft_model
    ft = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], task_type="SEQ_2_SEQ_LM"))
    for p in ft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ft.config.use_cache = False
    ft.enable_input_require_grads()
    ft.print_trainable_parameters()
    return ft


def build_dataset(tok, directions, max_len: int = 128, seed: int = 13):
    """directions: list of (src_lines, tgt_lines, src_code, tgt_code). Concatenated+shuffled."""
    from datasets import Dataset, concatenate_datasets
    parts = []
    for src, tgt, sc, tc in directions:
        tok.src_lang, tok.tgt_lang = LANG[sc], LANG[tc]
        parts.append(Dataset.from_dict({"src": src, "tgt": tgt}).map(
            lambda b: tok(b["src"], text_target=b["tgt"], max_length=max_len, truncation=True),
            batched=True, remove_columns=["src", "tgt"]))
    return concatenate_datasets(parts).shuffle(seed=seed)


def finetune(ft, tok, dataset, out_dir: str, epochs: int = 2, lr: float = 2e-4,
             batch_size: int = 4, grad_accum: int = 4):
    """LoRA fine-tune (fp16, gradient checkpointing). Saves the adapter to out_dir."""
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
    _free()
    args = Seq2SeqTrainingArguments(
        output_dir=f"{out_dir}-trainer", num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr, fp16=torch.cuda.is_available(), logging_steps=100,
        save_strategy="no", report_to=[])
    Seq2SeqTrainer(model=ft, args=args, train_dataset=dataset,
                   data_collator=DataCollatorForSeq2Seq(tok, model=ft)).train()
    ft.save_pretrained(out_dir)
    ft.eval()
    ft.config.use_cache = True
    return ft


@torch.no_grad()
def translate(model, tok, texts, src: str, tgt: str, bs: int = 8, max_len: int = 160, beams: int = 5):
    tok.src_lang = LANG[src]
    tgt_id = tok.convert_tokens_to_ids(LANG[tgt])
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_len, num_beams=beams)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    _free()
    return out


def score(hyp, ref):
    """Return (BLEU, chrF) with sacreBLEU (tok:13a)."""
    from sacrebleu.metrics import BLEU, CHRF
    return (round(BLEU().corpus_score(hyp, [ref]).score, 2),
            round(CHRF().corpus_score(hyp, [ref]).score, 2))
