# VoiceClone AI

Sistema de clonación de voz con XTTS v2 — genera audio con cualquier voz en tiempo real.

## Características

✅ **Clonación de voz natural** — soporta hasta 3 clips de referencia para mejor precisión  
✅ **Puntuación limpia** — elimina símbolos que XTTS lee en voz alta (elipsis, comillas, paréntesis, etc.)  
✅ **Sin alucinaciones** — cada chunk termina en punto/exclamación/interrogación para evitar palabras inventadas  
✅ **Estilos expresivos** — calmado, natural, expresivo, enérgico, o **reportera de espectáculos**  
✅ **Ritmo adaptable** — velocidad variable por estilo (2.3–3.7 palabras/segundo)  
✅ **Sin artefactos** — sentence-splitting inteligente + comma pauses + silence trimming

## Cómo usar

### 1. Subir voz de referencia
- Usa 1, 2 o 3 clips de audio (WAV, MP3, OGG, M4A)
- Duración: 15–20 segundos cada clip
- Audio limpio, sin ruido de fondo

### 2. Guardar en biblioteca
- Dale nombre a la voz
- Se guarda permanentemente en el servidor
- Puedes usarla sin volver a subirla

### 3. Generar audio
- Escribe el texto en español (o en otro idioma soportado)
- Selecciona la voz y el estilo
- El sistema genera automáticamente con puntuación normalizada

## Estilos disponibles

| Estilo | Temperature | Repetition Penalty | Speed | Uso |
|--------|-------------|-------------------|-------|-----|
| **😌 Calmado** | 0.55 | 3.5 | 0.97x | Audiolibros, meditación |
| **🗣 Natural** | 0.68 | 2.8 | 1.0x | Narración neutral |
| **✨ Expresivo** | 0.78 | 2.4 | 1.0x | Contenido general con emoción |
| **🔥 Enérgico** | 0.85 | 2.1 | 1.05x | Anuncios, motivacional |
| **🎤 Reportera** ⭐ | 0.85 | 2.1 | 1.07x | **Espectáculos, primicia, entusiasmo máximo** |

**Default:** Reportera (máxima emoción, declarativas leídas como `¡exclamación!`)

## Mejoras recientes (2026)

### Limpieza de puntuación (v2026-07-01)
- **Sanitización agresiva** — convierte `…` en `.`, `;:—–` en `,`, elimina símbolos
- **No más lectura de signos** — XTTS nunca lee `***`, `«»`, paréntesis, etc.
- **Chunks cerrados** — cada fragmento termina en terminal mark (`.!?`) para evitar alucinaciones
- **Comas como pausa** — clauses intermedias terminan en punto, pero con pausa corta (160ms)

### Estilo Reportera (v2026-07-01)
- **Entonación exclamativa** — declarativas se sintetizan como `¡...!` para máxima energía
- **Ritmo rápido** — 1.07x speed + banda de naturalidad extendida a 3.7 wps
- **Sampling seguro** — temp 0.85 / rep 2.1 (tope XTTS antes de alucinaciones)
- **Ideal para** — contenido de entretenimiento, noticias, primicia

### Audio Quality
- **Gentle preprocessing** — highpass 80Hz + FFT denoise (nf=-30) + loudnorm -16 LUFS
- **Silence trimming** — elimina padding XTTS en inicio/final de cada chunk
- **Voice conditioning** — usa hasta 30s de referencia (6s windows) para mejor timbre

### Voice Cloning
- **Multi-clip support** — 1–3 clips de referencia para más variación de voz
- **Auto-merge** — concatena y normaliza múltiples referencias
- **Reference cleanup** — preprocesa audios para que XTTS capture timbre real

### Timing & Prosody
- **Sentence + comma splitting** — frases largas se parten en cláusulas (comma=160ms pause, period=320ms pause)
- **Natural speed band** — reportera permite 2.3–3.7 wps; otros estilos 2.3–3.3 wps
- **Pause preservation** — las pausas por coma se honran incluso cuando se cierran chunks

### Idioma
- **Español Colombiano (es-co)** — default, acento latino nativo
- **Multilenguaje disponible** — en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi
- **Spanish marks auto** — agrega `¿` y `¡` de apertura automáticamente en preguntas/exclamaciones

## Arquitectura

```
Frontend (HTML/JS)
    ↓
FastAPI Backend (voice cloning endpoints)
    ↓
XTTS v2 (Coqui TTS — multilingual)
    ↓
FFmpeg (audio preprocessing + concatenation)
    ↓
Railway Volume (/app/voices — persistent storage)
```

## API

### POST `/voice/save`
Guardar voz en biblioteca (1–3 clips)

```bash
curl -X POST https://kind-contentment-production-4275.up.railway.app/voice/save \
  -F "audio=@clip1.wav" \
  -F "audio=@clip2.wav" \
  -F "name=Mi Voz"
```

**Respuesta:**
```json
{
  "status": "saved",
  "voice_id": "ea24d6e0",
  "name": "Mi Voz",
  "clips": 2
}
```

### POST `/voice/generate`
Generar audio desde voz guardada

```bash
curl -X POST https://kind-contentment-production-4275.up.railway.app/voice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "¡Última hora! La estrella acaba de llegar a la alfombra roja...",
    "voice_id": "ea24d6e0",
    "language": "es",
    "style": "reportera"
  }' \
  --output speech.wav
```

**Parámetros:**
- `text` (required): texto a sintetizar (sin límite de longitud)
- `voice_id` (optional): ID de la voz guardada; sin él usa la primera
- `language` (default: "es-co"): código de idioma
- `style` (default: "reportera"): calmado | natural | expresivo | energico | reportera

### GET `/voice/list`
Listar voces guardadas

```bash
curl https://kind-contentment-production-4275.up.railway.app/voice/list
```

### POST `/clone`
One-shot voice clone (sin guardar)

```bash
curl -X POST https://kind-contentment-production-4275.up.railway.app/clone \
  -F "audio=@reference.wav" \
  -F "text=Hola, esto es una prueba." \
  -F "language=es" \
  -F "style=reportera" \
  --output output.wav
```

### GET `/health`
Verificar estado y versión del deploy

```bash
curl https://kind-contentment-production-4275.up.railway.app/health
# {"status": "ok", "build": "vq-2026-07-01-reportera"}
```

## Deploy

Hospedado en Railway:
- **URL:** https://kind-contentment-production-4275.up.railway.app
- **Environment:** Python 3.10 + FastAPI + XTTS v2 (vía Coqui)
- **Storage:** `/app/voices` volume (persistent, 4.9 GB)
- **Build version:** `vq-2026-07-01-reportera`

## Parámetros XTTS v2 por estilo

```python
STYLE_PRESETS = {
    "calmado":   {"temperature": 0.55, "repetition_penalty": 3.5, "top_k": 45, "top_p": 0.82, "speed": 0.97},
    "natural":   {"temperature": 0.68, "repetition_penalty": 2.8, "top_k": 50, "top_p": 0.85, "speed": 1.0},
    "expresivo": {"temperature": 0.78, "repetition_penalty": 2.4, "top_k": 55, "top_p": 0.88, "speed": 1.0},
    "energico":  {"temperature": 0.85, "repetition_penalty": 2.1, "top_k": 60, "top_p": 0.90, "speed": 1.05},
    "reportera": {"temperature": 0.85, "repetition_penalty": 2.1, "top_k": 60, "top_p": 0.90, "speed": 1.07},
}
```

- `temperature`: diversidad prosódica (0.0–1.0); capped at 0.85 para evitar alucinaciones
- `repetition_penalty`: cuánto penaliza palabras repetidas; más alto = menos repetición
- `top_k` / `top_p`: límites de muestreo para variación
- `speed`: velocidad de síntesis (ajustada luego por banda de naturalidad)

## Limitaciones

- Máximo 3 clips de referencia por voz
- Máximo ~2000 caracteres por generación (se dividen automáticamente en chunks)
- Tempo adjustment solo soporta ±12% (0.9x–1.12x) para mantener naturalidad
- Requiere acceso a micrófono/archivos de audio (para referencia)

## Futuro

- [ ] GPU support en Railway (CUDA)
- [ ] Caché de síntesis (misma voz + texto = respuesta cached)
- [ ] Fine-tuning de parámetros por voice
- [ ] Export a múltiples formatos (MP3, OGG, FLAC)
- [ ] Streaming de audio en tiempo real
