"""FastAPI translation service. Run:  uvicorn api:app --host 0.0.0.0 --port 8000

POST /translate  {"text": "...", "src": "scn", "tgt": "en"}  ->  {"translation": "..."}
The Telegram bot (and any other front-end) talk to this one endpoint.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from translator import Translator

app = FastAPI(title="Sicilian NMT")
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = Translator()
    return _engine


class Req(BaseModel):
    text: str
    src: str = "scn"
    tgt: str = "en"


@app.on_event("startup")
def _warm():
    engine()   # load the model once at boot, not on the first request


@app.get("/health")
def health():
    return {"ok": True, "adapter": engine().adapter or "none (zero-shot base)"}


@app.post("/translate")
def translate(req: Req):
    out = engine().translate([req.text], req.src, req.tgt)[0]
    return {"translation": out}
