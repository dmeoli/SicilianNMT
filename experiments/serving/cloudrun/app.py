"""The translator of SicilianNMT as a service on Cloud Run: the CTranslate2 model (NLLB-200
1.3B with the LoRA adapter merged, int8 weights), baked into the image in MODEL_DIR, answers
POST /translate, the API of experiments/serving/api.py."""
import os

import ctranslate2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
LANG = {"scn": "scn_Latn", "en": "eng_Latn", "it": "ita_Latn"}
BEAM = int(os.environ.get("BEAM", "5"))

translator = ctranslate2.Translator(MODEL_DIR, device="cpu", compute_type="int8",
                                    intra_threads=os.cpu_count() or 2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

app = FastAPI(title="Tradutturi Sicilianu")
app.add_middleware(CORSMiddleware, allow_methods=["POST", "GET"], allow_headers=["*"],
                   allow_origins=["https://dmeoli.github.io", "http://localhost:8771"])


class Req(BaseModel):
    text: str
    src: str = "scn"
    tgt: str = "it"


@app.get("/")
def health():
    return {"ok": True}


@app.post("/translate")
def translate(r: Req):
    if r.src not in LANG or r.tgt not in LANG or r.src == r.tgt:
        raise HTTPException(400, f"src and tgt must be two of {list(LANG)}")
    text = r.text.strip()[:1000]
    if not text:
        return {"translation": ""}
    tokenizer.src_lang = LANG[r.src]
    toks = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    res = translator.translate_batch([toks], target_prefix=[[LANG[r.tgt]]], beam_size=BEAM,
                                     max_decoding_length=256)
    out = res[0].hypotheses[0][1:]
    return {"translation": tokenizer.decode(tokenizer.convert_tokens_to_ids(out),
                                            skip_special_tokens=True)}
