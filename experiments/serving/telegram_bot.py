"""Telegram bot front-end for the Sicilian translator.

Setup: talk to @BotFather, /newbot, copy the token, then:
    export TELEGRAM_TOKEN=123:abc
    python telegram_bot.py

By default it loads the model in-process (via translator.Translator). Set
SICILIAN_API=http://host:8000 to instead forward to a running FastAPI service.

Usage in chat:
    /sc2en <text>   translate Sicilian -> English
    /en2sc <text>   translate English -> Sicilian
    plain text      uses the last direction you picked (default scn->en)
"""
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes, filters)

TOKEN = os.environ["TELEGRAM_TOKEN"]
API = os.environ.get("SICILIAN_API", "").strip() or None

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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Sicilianu → English", callback_data="scn:en"),
        InlineKeyboardButton("English → Sicilianu", callback_data="en:scn"),
    ]])
    await update.message.reply_text(
        "Ciau! Sugnu u tradutturi sicilianu. Scrivimi na frasi o scegli a direzioni:",
        reply_markup=kb)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    src, tgt = q.data.split(":")
    context.user_data["dir"] = (src, tgt)
    await q.answer()
    await q.edit_message_text(f"Direzioni: {src} → {tgt}. Mandami u testu.")


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
    app.add_handler(CommandHandler("sc2en", make_cmd("scn", "en")))
    app.add_handler(CommandHandler("en2sc", make_cmd("en", "scn")))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()
