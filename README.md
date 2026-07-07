# VoiceClone AI

Sistema de clonación de voz con XTTS v2 — genera audio con cualquier voz en tiempo real.

## Características

✅ **Clonación de voz natural** — soporta hasta 3 clips de referencia para mejor precisión
✅ **Idioma colombiano** — voces con acento latino (es-co)
✅ **Velocidad exacta** — 2.5 palabras por segundo garantizado
✅ **Altamente emocional** — tono entusiasta y expresivo
✅ **Sin artefactos** — sentence-splitting inteligente + silence trimming
✅ **Preprocesamiento agresivo** — FFT denoising + neural denoise para audio limpio

## Cómo usar

### 1. Subir voz de referencia
- Usa 1, 2 o 3 clips de audio (WAV, MP3, OGG, M4A)
- Duración: 15-20 segundos cada clip
- Audio limpio, sin ruido de fondo

### 2. Guardar en biblioteca
- Dale nombre a la voz
- Se guarda permanentemente en el servidor
- Puedes usarla sin volver a subirla

### 3. Generar audio
- Escribe el texto en español
- Selecciona la voz
- El sistema genera a 2.5 palabras/segundo

## Mejoras recientes (2024)

### Audio Quality
- **FFT Denoising agresivo** (afftdn nf=-18)
- **Neural speech denoise** (anlmdn) para ruido de fondo
- **High-pass 100Hz** para eliminar rumble
- **Loudness normalization** a -13 LUFS

### Voice Cloning
- **Multi-clip support** — soporta hasta 3 clips de referencia
- **Sentence-level splitting** — evita cortes en palabras
- **Silence trimming** — elimina padding de XTTS
- **120ms crossfade** entre chunks para transiciones suaves

### Prosody & Emotion
- **Temperature 0.95** — máxima expresividad
- **Repetition penalty 1.8** — permite variación de entonación
- **Entusiasta por defecto** — ideal para contenido dinámico

### Timing
- **Velocidad exacta** — 2.5 palabras/segundo mediante ajuste de tempo
- **Atempo filter** — ajusta la velocidad preservando pitch

### Idioma
- **Español Colombiano (es-co)** — acento latino nativo
- **Multilenguaje disponible** — en, fr, de, pt, it, zh

## Arquitectura

```
Frontend (HTML/JS)
    ↓
FastAPI Backend
    ↓
XTTS v2 (Coqui TTS)
    ↓
FFmpeg (procesamiento de audio)
    ↓
Voice Library (Google Drive)
```

## API

### POST `/voice/save`
Guardar voz en biblioteca (hasta 3 clips)

```bash
curl -X POST https://kind-contentment-production-4275.up.railway.app/voice/save \
  -F "audio=@voice1.wav" \
  -F "audio=@voice2.wav" \
  -F "name=Mi Voz"
```

### POST `/voice/generate`
Generar audio desde voz guardada

```bash
curl -X POST https://kind-contentment-production-4275.up.railway.app/voice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, soy tu asistente.",
    "voice_id": "abc12345",
    "language": "es-co"
  }' \
  --output output.wav
```

## Deploy

Hospedado en Railway:
- **URL:** https://kind-contentment-production-4275.up.railway.app
- **Runtime:** Node.js 20 + Python 3 + FFmpeg
- **Volume:** `/app/voices` (persistente)

## Parámetros XTTS v2

```python
temperature = 0.95        # expresividad (0.0-1.0)
repetition_penalty = 1.8  # variación (1.0-10.0)
top_p = 0.92              # diversidad
top_k = 60                # diversidad
speed = 1.0               # velocidad base (ajustada por /voice/generate)
enable_text_splitting = False  # splitting manual
```

## Limitaciones

- Máximo 3 clips de referencia
- Máximo ~1000 caracteres por generación
- Tempo adjustment solo soporta 0.5x-2.0x
- Requiere acceso a micrófono/archivos de audio

## Futuro

- [ ] GPU support en Railway
- [ ] Caché de resultados (misma voz + texto)
- [ ] Fine-tuning de parámetros por voice
- [ ] Export a múltiples formatos (MP3, OGG)
