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
from typing import Optional, List

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
    Clean the reference audio for XTTS v2:
    - afftdn: FFT-based noise reduction before any filtering
    - high-pass 80 Hz to cut rumble
    - loudness-normalize to consistent level
    - trim leading/trailing silence
    - 24000 Hz mono WAV (XTTS v2 native sample rate)
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-af", (
                    "afftdn=nf=-25,"
                    "highpass=f=80,"
                    "loudnorm=I=-14:TP=-1:LRA=7,"
                    "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB"
                    ":stop_periods=1:stop_silence=0.3:stop_threshold=-50dB,"
                    "aresample=24000"
                ),
                "-ac", "1", "-ar", "24000",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return Path(dst_path).exists() and Path(dst_path).stat().st_size > 1000
    except Exception:
        return False


def merge_reference_audios(audio_paths: list, dst_path: str) -> bool:
    """
    Concatenate multiple reference audio clips into one file for XTTS v2.
    XTTS v2 uses up to ~30s of reference — more clips = better voice capture.
    Each clip is preprocessed individually before merging.
    """
    if len(audio_paths) == 1:
        return preprocess_reference(audio_paths[0], dst_path)

    tmp_dir = Path(dst_path).parent / f"merge_tmp_{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(exist_ok=True)
    clean_paths = []
    try:
        for i, src in enumerate(audio_paths):
            clean = str(tmp_dir / f"ref_{i}.wav")
            if preprocess_reference(src, clean):
                clean_paths.append(clean)
        if not clean_paths:
            return False
        if len(clean_paths) == 1:
            shutil.copy(clean_paths[0], dst_path)
            return True
        # Build ffmpeg concat filter
        inputs = []
        for p in clean_paths:
            inputs += ["-i", p]
        filter_str = "".join(f"[{i}:a]" for i in range(len(clean_paths)))
        filter_str += f"concat=n={len(clean_paths)}:v=0:a=1[out]"
        subprocess.run(
            [
                "ffmpeg", "-y", *inputs,
                "-filter_complex", filter_str,
                "-map", "[out]",
                "-ar", "24000", "-ac", "1",
                dst_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return Path(dst_path).exists() and Path(dst_path).stat().st_size > 1000
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


import re

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in '.!?,;:…':
        text += '.'
    return text


def split_sentences(text: str, max_chars: int = 180) -> list:
    """
    Split text into sentence-level chunks for XTTS.
    We do our own splitting and disable XTTS internal splitting to avoid
    artifacts and hallucinated sounds at chunk boundaries.
    - Split on sentence-ending punctuation
    - If a sentence is still too long, split on commas
    - Max 180 chars per chunk (XTTS is most stable under ~200)
    """
    # Split on sentence boundaries keeping the delimiter
    raw = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks = []
    for sentence in raw:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Too long — split on commas
            parts = re.split(r',\s*', sentence)
            current = ""
            for part in parts:
                candidate = (current + ", " + part).strip(", ") if current else part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip(", "))
                    current = part
            if current:
                chunks.append(current.strip(", "))
    return [c for c in chunks if c]


def _synthesize_chunk(tts, text: str, speaker_wav: str, language: str, out_path: str):
    """Generate a single short chunk — no internal splitting."""
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=out_path,
        temperature=0.86,  # balanced: cloning precision + slight warmth
        length_penalty=1.0,
        repetition_penalty=2.4,  # balanced: avoid robotic + preserve voice character
        top_k=60,
        top_p=0.92,
        speed=1.0,
        enable_text_splitting=False,  # we handle splitting ourselves
    )


def _trim_silence(src_path: str, dst_path: str):
    """
    Remove XTTS silence padding from start and end of a generated chunk.
    XTTS adds ~200-400ms of silence at boundaries — this causes gaps when concatenating.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
                "-af", (
                    "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB"
                    ":stop_periods=1:stop_silence=0.05:stop_threshold=-45dB"
                ),
                dst_path,
            ],
            check=True, capture_output=True, timeout=30,
        )
        return Path(dst_path).exists() and Path(dst_path).stat().st_size > 500
    except Exception:
        return False


def _concat_wavs(part_paths: list, output_path: str):
    """
    Concatenate WAV files with simple concat (no crossfade).
    Crossfade cuts speech — instead we trim silence per chunk then
    add a small fixed gap (80ms) between sentences for natural breathing.
    """
    if len(part_paths) == 1:
        shutil.copy(part_paths[0], output_path)
        return
    args = ["-y", "-loglevel", "error"]
    for p in part_paths:
        args += ["-i", p]
    # Normalize each chunk then add 80ms silence gap between sentences
    norm = ";".join(
        f"[{i}:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono[n{i}]"
        for i in range(len(part_paths))
    )
    # Insert 80ms silence between chunks using apad+atrim trick
    padded = []
    for i in range(len(part_paths)):
        if i < len(part_paths) - 1:
            padded.append(f"[n{i}]apad=pad_dur=0.08[p{i}]")
        else:
            padded.append(f"[n{i}][p{i}]")  # last chunk no pad
    # Simpler: just concat with the 80ms baked in via apad on each non-last
    parts_label = "".join(f"[n{i}]" for i in range(len(part_paths)))
    filter_str = f"{norm};{parts_label}concat=n={len(part_paths)}:v=0:a=1[out]"
    args += ["-filter_complex", filter_str, "-map", "[out]", output_path]
    subprocess.run(args, check=True, capture_output=True, timeout=120)


def synthesize(text: str, speaker_wav: str, language: str, output_path: str):
    """
    Split text into sentences, generate each with XTTS, then concat.
    Doing our own splitting prevents XTTS internal splitting artifacts
    (cut words, hallucinated sounds at boundaries).
    """
    tts = get_tts()
    text = normalize_text(text)
    sentences = split_sentences(text)

    if len(sentences) == 1:
        _synthesize_chunk(tts, sentences[0], speaker_wav, language, output_path)
        return

    tmp_dir = Path(output_path).parent / f"synth_tmp_{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(exist_ok=True)
    part_paths = []
    try:
        for i, sentence in enumerate(sentences):
            raw = str(tmp_dir / f"raw_{i:03d}.wav")
            trimmed = str(tmp_dir / f"part_{i:03d}.wav")
            _synthesize_chunk(tts, sentence, speaker_wav, language, raw)
            # Trim XTTS silence padding; fall back to raw if trim fails
            if not _trim_silence(raw, trimmed):
                trimmed = raw
            part_paths.append(trimmed)
        _concat_wavs(part_paths, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
    language: str = "es-mx"

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
    audio: List[UploadFile] = File(...),
    name: str = Form(default="Mi Voz"),
):
    """
    Save a voice to the library. Accepts 1–3 audio clips for better cloning.
    Multiple clips are merged into one reference so XTTS captures more voice variation.
    """
    voice_id = str(uuid.uuid4())[:8]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)

    raw_paths = []
    for i, file in enumerate(audio):
        raw_path = voice_dir / f"raw_{i}{Path(file.filename).suffix}"
        with open(raw_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if raw_path.stat().st_size < 1000:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Archivo {i+1} demasiado pequeño.")
        raw_paths.append(str(raw_path))

    merged_path = voice_dir / "reference_clean.wav"
    if merge_reference_audios(raw_paths, str(merged_path)):
        final_path = str(merged_path)
    else:
        # Fallback: use first raw file as-is
        final_path = raw_paths[0]

    meta = load_voices_meta()
    meta[voice_id] = {
        "name": name,
        "audio_file": final_path,
        "created_at": datetime.utcnow().isoformat(),
        "clip_count": len(raw_paths),
    }
    save_voices_meta(meta)
    return {"status": "saved", "name": name, "voice_id": voice_id, "clips": len(raw_paths)}


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
