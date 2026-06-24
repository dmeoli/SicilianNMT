"""FastAPI app: the translation API + the local web front-end.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000   (from experiments/serving/)

- POST /translate  {"text": "...", "src": "scn", "tgt": "en"}  ->  {"translation": "..."}
- GET  /health
- GET  /            -> the web UI (static/index.html), which calls /translate

The same /translate endpoint backs the Telegram bot (telegram_bot.py with SICILIAN_API).
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from translator import Translator

app = FastAPI(title="Tradutturi Sicilianu")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine = None
HERE = os.path.dirname(os.path.abspath(__file__))


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
    if os.environ.get("WARM", "1") == "1":
        engine()   # load the model once at boot, not on the first request


@app.get("/health")
def health():
    return {"ok": True, "adapter": engine().adapter or "none (zero-shot base)"}


@app.post("/translate")
def translate(req: Req):
    text = req.text.strip()
    if not text:
        return {"translation": ""}
    out = engine().translate([text], req.src, req.tgt)[0]
    return {"translation": out}


# serve the web UI last, so it does not shadow the API routes above
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
