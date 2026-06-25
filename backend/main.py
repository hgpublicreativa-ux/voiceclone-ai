import os
import uuid
import shutil
import json
import secrets
import subprocess
from pathlib import Path
from datetime import datetime

# Patch torch.load before any TTS import — PyTorch 2.6 changed weights_only default
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="VoiceClone AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────
VOICES_DIR = Path("voices")
OUTPUTS_DIR = Path("outputs")
STATIC_DIR = Path("static")
KEYS_FILE = VOICES_DIR / "api_keys.json"
VOICES_META_FILE = VOICES_DIR / "voices_meta.json"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "admin-change-me")

# ── TTS singleton ─────────────────────────────────────────────────────────────
_tts_instance = None

def get_tts():
    global _tts_instance
    if _tts_instance is None:
        from TTS.api import TTS
        _tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts_instance


def preprocess_reference(src_path: str, dst_path: str) -> bool:
    """
    Clean the reference audio so XTTS produces natural output:
    - convert to 22050 Hz mono WAV (XTTS native rate)
    - loudness-normalize (consistent level)
    - high-pass 70 Hz to cut rumble, light noise gate
    Returns True if ffmpeg succeeded, else False (caller falls back to raw file).
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=22050",
                "-ac", "1", "-ar", "22050",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return Path(dst_path).exists() and Path(dst_path).stat().st_size > 1000
    except Exception:
        return False


def synthesize(text: str, speaker_wav: str, language: str, output_path: str):
    """
    Run XTTS synthesis with parameters tuned for natural, non-robotic output.
    - temperature 0.75: more expressive prosody variation
    - repetition_penalty 5.0: avoids monotone/robotic repetition
    - top_k/top_p: balanced sampling for natural intonation
    - enable_text_splitting: better sentence-level prosody on long text
    """
    tts = get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=5.0,
        top_k=50,
        top_p=0.85,
        speed=1.0,
        enable_text_splitting=True,
    )

# ── Storage helpers ───────────────────────────────────────────────────────────
def load_keys() -> dict:
    if KEYS_FILE.exists():
        return json.loads(KEYS_FILE.read_text())
    return {}

def save_keys(keys: dict):
    KEYS_FILE.write_text(json.dumps(keys, indent=2))

def load_voices_meta() -> dict:
    if VOICES_META_FILE.exists():
        return json.loads(VOICES_META_FILE.read_text())
    return {}

def save_voices_meta(meta: dict):
    VOICES_META_FILE.write_text(json.dumps(meta, indent=2))

# ── Auth ──────────────────────────────────────────────────────────────────────
def require_admin(x_admin_secret: str = Header(...)):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Admin secret inválido.")

def require_api_key(x_api_key: str = Header(...)):
    keys = load_keys()
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="API key inválida o revocada.")
    return keys[x_api_key]

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    index = Path("static/app/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"status": "VoiceClone AI API", "docs": "/docs"}

@app.get("/config.js", response_class=None)
def config_js(request_origin: str = ""):
    from fastapi.responses import Response
    js = "const API_BASE = window.location.origin;"
    return Response(content=js, media_type="application/javascript")

# ── API Key management ────────────────────────────────────────────────────────
@app.post("/api-keys")
def create_api_key(
    name: str = Form(...),
    _: None = Depends(require_admin),
):
    """Create a new API key. Requires X-Admin-Secret header."""
    key = f"vc-{secrets.token_urlsafe(32)}"
    keys = load_keys()
    keys[key] = {
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "active": True,
    }
    save_keys(keys)
    return {"api_key": key, "name": name}

@app.get("/api-keys")
def list_api_keys(_: None = Depends(require_admin)):
    """List all API keys."""
    keys = load_keys()
    return [{"key": k[:12] + "...", "name": v["name"], "created_at": v["created_at"]} for k, v in keys.items()]

@app.delete("/api-keys/{key}")
def revoke_api_key(key: str, _: None = Depends(require_admin)):
    """Revoke an API key."""
    keys = load_keys()
    if key not in keys:
        raise HTTPException(status_code=404, detail="Key no encontrada.")
    del keys[key]
    save_keys(keys)
    return {"status": "revoked"}

# ── Voice management ──────────────────────────────────────────────────────────
@app.post("/voices")
async def upload_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    _: None = Depends(require_admin),
):
    """Upload and save a voice. Returns voice_id."""
    voice_id = str(uuid.uuid4())[:8]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)
    audio_path = voice_dir / f"reference{Path(audio.filename).suffix}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    if audio_path.stat().st_size < 1000:
        shutil.rmtree(voice_dir)
        raise HTTPException(status_code=400, detail="Archivo de audio demasiado pequeño.")

    # Clean the reference for more natural cloning
    clean_path = voice_dir / "reference_clean.wav"
    final_path = str(clean_path) if preprocess_reference(str(audio_path), str(clean_path)) else str(audio_path)

    meta = load_voices_meta()
    meta[voice_id] = {
        "name": name,
        "audio_file": final_path,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_voices_meta(meta)
    return {"voice_id": voice_id, "name": name}

@app.get("/voices")
def list_voices(_: None = Depends(require_admin)):
    """List all saved voices."""
    meta = load_voices_meta()
    return [{"voice_id": vid, "name": v["name"], "created_at": v["created_at"]} for vid, v in meta.items()]

@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str, _: None = Depends(require_admin)):
    """Delete a voice."""
    meta = load_voices_meta()
    if voice_id not in meta:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    del meta[voice_id]
    save_voices_meta(meta)
    return {"status": "deleted"}

# ── Speech synthesis (public with API key) ────────────────────────────────────
class SpeakRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    language: str = "es"

@app.post("/speak")
def speak(
    req: SpeakRequest,
    key_data: dict = Depends(require_api_key),
):
    """
    Generate speech using a saved voice.
    Requires X-API-Key header.
    If voice_id is omitted, uses the first available voice.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    meta = load_voices_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="No hay voces guardadas. Sube una voz primero.")

    voice_id = req.voice_id
    if voice_id is None:
        voice_id = next(iter(meta))
    if voice_id not in meta:
        raise HTTPException(status_code=404, detail=f"Voz '{voice_id}' no encontrada.")

    speaker_wav = meta[voice_id]["audio_file"]
    if not Path(speaker_wav).exists():
        raise HTTPException(status_code=404, detail="Archivo de voz no encontrado en disco.")

    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.wav"
    try:
        synthesize(req.text, speaker_wav, req.language, str(output_path))
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")

    return FileResponse(path=str(output_path), media_type="audio/wav", filename="speech.wav")

# ── One-shot clone (no API key needed, for frontend testing) ─────────────────
@app.post("/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(default="es"),
):
    """Clone voice on the fly without saving. For frontend testing."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    tmp_path = VOICES_DIR / f"tmp_{uuid.uuid4()}{Path(audio.filename).suffix}"
    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.wav"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        synthesize(text, str(tmp_path), language, str(output_path))
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return FileResponse(path=str(output_path), media_type="audio/wav", filename="cloned_voice.wav")


# ── Public voice library (used by the web UI) ─────────────────────────────────
@app.post("/voice/save")
async def save_voice_library(
    audio: UploadFile = File(...),
    name: str = Form(default="Mi Voz"),
):
    """Save a voice to the library with a unique id. Each voice persists in the volume."""
    voice_id = str(uuid.uuid4())[:8]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)
    audio_path = voice_dir / f"reference{Path(audio.filename).suffix}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    if audio_path.stat().st_size < 1000:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Archivo demasiado pequeño.")

    # Clean the reference for more natural cloning
    clean_path = voice_dir / "reference_clean.wav"
    final_path = str(clean_path) if preprocess_reference(str(audio_path), str(clean_path)) else str(audio_path)

    meta = load_voices_meta()
    meta[voice_id] = {
        "name": name,
        "audio_file": final_path,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_voices_meta(meta)
    return {"status": "saved", "name": name, "voice_id": voice_id}


@app.get("/voice/list")
def voice_library_list():
    """List all saved voices (public — for the web UI)."""
    meta = load_voices_meta()
    return [
        {"voice_id": vid, "name": v["name"], "created_at": v["created_at"]}
        for vid, v in sorted(meta.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True)
    ]


@app.delete("/voice/item/{voice_id}")
def voice_library_delete(voice_id: str):
    """Delete a voice from the library (public — for the web UI)."""
    meta = load_voices_meta()
    if voice_id not in meta:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir, ignore_errors=True)
    del meta[voice_id]
    save_voices_meta(meta)
    return {"status": "deleted", "voice_id": voice_id}


@app.post("/voice/generate")
def voice_library_generate(req: SpeakRequest):
    """Generate speech from a saved voice by id (public — for the web UI test)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")
    meta = load_voices_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="No hay voces guardadas.")
    voice_id = req.voice_id or next(iter(meta))
    if voice_id not in meta:
        raise HTTPException(status_code=404, detail=f"Voz '{voice_id}' no encontrada.")
    speaker_wav = meta[voice_id]["audio_file"]
    if not Path(speaker_wav).exists():
        raise HTTPException(status_code=404, detail="Archivo de voz no encontrado.")

    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.wav"
    try:
        synthesize(req.text, speaker_wav, req.language, str(output_path))
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")
    return FileResponse(path=str(output_path), media_type="audio/wav", filename="speech.wav")
