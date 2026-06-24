# Light image for the Telegram bot in API mode (forwards to SICILIAN_API).
# No torch/transformers -> fits the 1 GB shared host. The translation model
# runs separately (uvicorn api:app) on a capable host; the bot just calls it.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-bot.txt .
RUN pip install --no-cache-dir -r requirements-bot.txt

COPY telegram_bot.py .

# SICILIAN_API must be set (so the bot does NOT import the local model);
# TOKEN / TELEGRAM_TOKEN carries the BotFather token.
ENV SICILIAN_API=""
CMD ["python", "telegram_bot.py"]
