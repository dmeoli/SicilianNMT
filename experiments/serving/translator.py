"""Shared Sicilian<->English translation engine.

Wraps NLLB-200 (base) plus our LoRA adapter, if present. Both the FastAPI service
and the Telegram bot import `Translator` from here, so the model is loaded and
defined in exactly one place.
"""
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = os.environ.get("NLLB_BASE", "facebook/nllb-200-1.3B")
# directory of a trained LoRA adapter (e.g. the one the Colab notebook saves to Drive);
# leave unset to serve the plain zero-shot base model.
ADAPTER_DIR = os.environ.get("NLLB_ADAPTER", "").strip() or None

LANG = {"scn": "scn_Latn", "en": "eng_Latn"}


class Translator:
    def __init__(self, base=BASE_MODEL, adapter=ADAPTER_DIR, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()   # fold LoRA in for faster inference
        self.model = model.to(self.device).eval()
        self.adapter = adapter

    @torch.no_grad()
    def translate(self, text, src, tgt, num_beams=5, max_len=160):
        if src not in LANG or tgt not in LANG:
            raise ValueError(f"languages must be one of {list(LANG)}")
        self.tok.src_lang = LANG[src]
        tgt_id = self.tok.convert_tokens_to_ids(LANG[tgt])
        enc = self.tok(text, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len).to(self.device)
        gen = self.model.generate(**enc, forced_bos_token_id=tgt_id,
                                  max_length=max_len, num_beams=num_beams)
        return self.tok.batch_decode(gen, skip_special_tokens=True)
