"""Telegram bot front-end for the trilingual translator (Sicilian / English / Italian).

Setup: talk to @BotFather, /newbot, copy the token, then:
    export TOKEN=123:abc            # or TELEGRAM_TOKEN
    python telegram_bot.py

By default it loads the model in-process (via translator.Translator). Set
SICILIAN_API=http://host:8000 to instead forward to a running FastAPI service.

In chat: /start to pick a direction with buttons, then send text. Or use a command:
    /sc2en /en2sc /sc2it /it2sc /en2it /it2en
"""
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes, filters)

TOKEN = os.environ.get("TELEGRAM_TOKEN") or os.environ["TOKEN"]
API = os.environ.get("SICILIAN_API", "").strip() or None

LANGS = {"scn": "Sicilianu", "en": "English", "it": "Italiano"}
# the six directed pairs we offer
PAIRS = [("scn", "en"), ("en", "scn"), ("scn", "it"),
         ("it", "scn"), ("en", "it"), ("it", "en")]

if API:
    import requests

    def translate(text, src, tgt):
        r = requests.post(f"{API}/translate",
                          json={"text": text, "src": src, "tgt": tgt}, timeout=120)
        r.raise_for_status()
        return r.json()["translation"]
else:
    from translator import Translator
    _engine = Translator()

    def translate(text, src, tgt):
        return _engine.translate([text], src, tgt)[0]


def direction(context):
    return context.user_data.get("dir", ("scn", "en"))


def keyboard():
    rows, row = [], []
    for src, tgt in PAIRS:
        row.append(InlineKeyboardButton(f"{LANGS[src][:3]}→{LANGS[tgt][:3]}",
                                        callback_data=f"{src}:{tgt}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ciau! Sugnu u tradutturi. Scegli a direzioni, poi mandami u testu:",
        reply_markup=keyboard())


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    src, tgt = q.data.split(":")
    context.user_data["dir"] = (src, tgt)
    await q.answer()
    await q.edit_message_text(f"Direzioni: {LANGS[src]} → {LANGS[tgt]}. Mandami u testu.")


def make_cmd(src, tgt):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data["dir"] = (src, tgt)
        text = " ".join(context.args)
        if not text:
            await update.message.reply_text(f"Usa: manda u testu doppu u cumannu ({src}→{tgt}).")
            return
        await update.message.reply_text(translate(text, src, tgt))
    return handler


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    src, tgt = direction(context)
    await update.message.reply_text(translate(update.message.text, src, tgt))


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    for src, tgt in PAIRS:
        app.add_handler(CommandHandler(f"{src[:2] if src != 'scn' else 'sc'}2{tgt[:2] if tgt != 'scn' else 'sc'}",
                                       make_cmd(src, tgt)))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()
