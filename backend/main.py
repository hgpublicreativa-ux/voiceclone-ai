import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="VoiceClone AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
VOICE_DIR = Path("voices")
STATIC_DIR = Path("static")
OUTPUT_DIR.mkdir(exist_ok=True)
VOICE_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_VOICE_PATH = VOICE_DIR / "default_voice.wav"
DEFAULT_VOICE_META = VOICE_DIR / "default_voice.txt"

_tts_instance = None

def get_tts():
    global _tts_instance
    if _tts_instance is None:
        from TTS.api import TTS
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts_instance


# — Health —

@app.get("/health")
def health():
    return {"status": "ok"}


# — Voice management —

@app.post("/voice/save")
async def save_voice(
    audio: UploadFile = File(...),
    name: str = Form(default="Mi Voz"),
):
    """Save reference audio as the default voice."""
    tmp_path = VOICE_DIR / f"tmp_{uuid.uuid4()}{Path(audio.filename).suffix}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        # Convert to WAV via XTTS preprocessing (validates audio)
        tts = get_tts()
        # Just verify it loads without error by doing a short synthesis
        test_out = OUTPUT_DIR / f"test_{uuid.uuid4()}.wav"
        tts.tts_to_file(
            text="Prueba de voz.",
            speaker_wav=str(tmp_path),
            language="es",
            file_path=str(test_out),
        )
        test_out.unlink(missing_ok=True)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Audio inválido: {str(e)}")

    shutil.move(str(tmp_path), str(DEFAULT_VOICE_PATH))
    DEFAULT_VOICE_META.write_text(name, encoding="utf-8")

    return {"status": "saved", "name": name}


@app.get("/voice/status")
def voice_status():
    """Check if a default voice is saved."""
    if DEFAULT_VOICE_PATH.exists():
        name = DEFAULT_VOICE_META.read_text(encoding="utf-8") if DEFAULT_VOICE_META.exists() else "Sin nombre"
        return {"has_voice": True, "name": name}
    return {"has_voice": False, "name": None}


@app.delete("/voice")
def delete_voice():
    """Remove the saved default voice."""
    DEFAULT_VOICE_PATH.unlink(missing_ok=True)
    DEFAULT_VOICE_META.unlink(missing_ok=True)
    return {"status": "deleted"}


# — Synthesis —

class SpeakRequest(BaseModel):
    text: str
    language: str = "es"

@app.post("/speak")
def speak(req: SpeakRequest):
    """Generate speech using the saved default voice. For use by external apps."""
    if not DEFAULT_VOICE_PATH.exists():
        raise HTTPException(status_code=404, detail="No hay voz predeterminada guardada. Usa /voice/save primero.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    output_path = OUTPUT_DIR / f"{uuid.uuid4()}_output.wav"

    try:
        tts = get_tts()
        tts.tts_to_file(
            text=req.text,
            speaker_wav=str(DEFAULT_VOICE_PATH),
            language=req.language,
            file_path=str(output_path),
        )
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")

    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename="speech.wav",
    )


# — One-shot clone (sin guardar) —

@app.post("/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(default="es"),
):
    """Clone a voice on the fly without saving it."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

    job_id = str(uuid.uuid4())
    audio_path = VOICE_DIR / f"tmp_{job_id}{Path(audio.filename).suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_output.wav"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        tts = get_tts()
        tts.tts_to_file(
            text=text,
            speaker_wav=str(audio_path),
            language=language,
            file_path=str(output_path),
        )
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)

    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename="cloned_voice.wav",
    )
