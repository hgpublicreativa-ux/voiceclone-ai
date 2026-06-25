FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV TTS_HOME=/app/tts_models
ENV COQUI_TOS_AGREED=1
RUN python -c "\
import os; os.makedirs('/app/tts_models', exist_ok=True); \
from TTS.api import TTS; \
TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY backend/ .
COPY embed/ ./static/embed/

RUN mkdir -p voices outputs

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
