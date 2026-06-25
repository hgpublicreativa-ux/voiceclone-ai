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

ENV TTS_HOME=/app/voices/tts_models
ENV COQUI_TOS_AGREED=1

COPY backend/ .
COPY embed/ ./static/embed/
COPY frontend/ ./static/app/

RUN mkdir -p voices outputs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
