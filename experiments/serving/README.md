# Serving

Ship the translator as a chat bot. One model engine (`translator.py`), two front-ends:
a FastAPI service (`api.py`) and a Telegram bot (`telegram_bot.py`). The bot can either
load the model in-process or forward to the API.

```
translator.py    NLLB-200 base + optional LoRA adapter -> .translate(text, src, tgt)
api.py           FastAPI: POST /translate  +  serves the web UI at /
static/          the local web front-end (Sicilian blue+gold), calls /translate
telegram_bot.py  Telegram bot (BotFather token); in-process or via the API
```

`src`/`tgt` are `scn` or `en`.

## Run

```bash
pip install -r requirements.txt
export NLLB_ADAPTER=/path/to/nllb-lora-bidir-1.3B   # optional; omit to serve zero-shot base
```

API + web UI (run from `experiments/serving/`):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/  for the translator web page, or call the API directly:
curl -s localhost:8000/translate -H 'content-type: application/json' \
     -d '{"text":"Bongiornu, comu stai?","src":"scn","tgt":"en"}'
```

The page at `/` is the local site (a Python-served, Perl-free front-end inspired by the
original *Tradutturi*); it just POSTs to `/translate`. Set `WARM=0` to skip loading the model
at startup (handy for serving the static UI without a GPU).

Telegram bot (in-process model):

```bash
export TELEGRAM_TOKEN=...        # from @BotFather
python telegram_bot.py
```

Or point the bot at a running API instead of loading the model itself:

```bash
export SICILIAN_API=http://localhost:8000
python telegram_bot.py
```

## Notes

- NLLB-1.3B runs on CPU (a few seconds per sentence — fine for light traffic) or a small GPU.
- The adapter is what the Colab notebook saves to Drive (`nllb-lora-bidir-1.3B`);
  `translator.py` merges it into the base weights at load for faster inference.
- **WhatsApp** would need Meta's official WhatsApp Business Cloud API (verified Business
  account + dedicated number; 24h reply window). Unofficial libraries violate the ToS.
  Add it later as a second front-end against the same `/translate` endpoint.
