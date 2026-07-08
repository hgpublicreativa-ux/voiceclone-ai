import os
import uuid
import shutil
import json
import secrets
import subprocess
from pathlib import Path
from datetime import datetime

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

def cleanup_old_outputs(max_age_hours: int = 24):
    """Remove output files older than max_age_hours to save disk space."""
    import time
    now = time.time()
    for f in OUTPUTS_DIR.glob("*.wav"):
        if now - f.stat().st_mtime > max_age_hours * 3600:
            f.unlink(missing_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "admin-change-me")

# ── TTS worker process ───────────────────────────────────────────────────────
# XTTS (~5 GB) runs in a CHILD process, never in the API process. An
# in-process "unload" (del + gc.collect) does NOT give the memory back to the
# OS — CPython keeps the freed pages mapped — and Railway bills resident
# memory per minute, which showed up as a flat ~8 GB for days (~$80/month of
# RAM alone). Killing the worker after an idle period is the only reliable
# way to return the RAM; the API process stays at a few hundred MB (torch is
# never imported here). Consecutive jobs reuse the live worker, so a batch of
# videos pays the model load (~1-2 min) only once.
import sys
import select
import threading
import time

WORKER_SCRIPT = os.environ.get(
    "TTS_WORKER_SCRIPT", str(Path(__file__).resolve().parent / "tts_worker.py")
)
TTS_IDLE_UNLOAD_SECONDS = int(os.environ.get("TTS_IDLE_UNLOAD_SECONDS", "600"))
TTS_JOB_TIMEOUT_SECONDS = int(os.environ.get("TTS_JOB_TIMEOUT_SECONDS", "1800"))

_worker = None            # subprocess.Popen of tts_worker.py, or None
_tts_lock = threading.RLock()
_tts_busy = 0             # jobs in flight; the reaper never kills while > 0
_tts_last_used = 0.0

# One synthesis at a time. Parallel XTTS runs on a shared CPU just thrash each
# other (client retries after a timeout pile zombie syntheses onto the
# threadpool and every chunk slows to a crawl); serializing keeps each chunk
# fast and bounds peak RAM.
_synth_gate = threading.Semaphore(1)

def _spawn_or_get_worker():
    global _worker
    with _tts_lock:
        if _worker is None or _worker.poll() is not None:
            _worker = subprocess.Popen(
                [sys.executable, "-u", WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,   # protocol replies only
                stderr=None,              # worker logs → container log
                text=True,
                bufsize=1,
            )
        return _worker

def _kill_worker():
    global _worker
    with _tts_lock:
        if _worker is not None:
            _worker.terminate()
            try:
                _worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _worker.kill()
                try:
                    _worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            _worker = None

def _worker_request(job: dict) -> dict:
    """Send one job to the worker and wait for its single-line JSON reply.
    On timeout or worker death the worker is killed/cleared so the next job
    gets a fresh process."""
    w = _spawn_or_get_worker()
    w.stdin.write(json.dumps(job) + "\n")
    w.stdin.flush()
    deadline = time.time() + TTS_JOB_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            _kill_worker()
            raise RuntimeError("síntesis excedió el tiempo límite (worker colgado)")
        ready, _, _ = select.select([w.stdout], [], [], min(remaining, 5.0))
        if ready:
            line = w.stdout.readline()
            if not line:  # EOF — worker died mid-job (likely OOM-killed)
                _kill_worker()
                raise RuntimeError("el worker TTS murió durante la síntesis (¿sin memoria?)")
            return json.loads(line)
        if w.poll() is not None:
            code = w.returncode
            _kill_worker()
            raise RuntimeError(f"el worker TTS terminó inesperadamente (código {code})")

def _busy_keepalive():
    """While a synthesis is in flight, ping our own public URL so Railway's
    serverless idle detector sees traffic. A long chunk can be silent for
    minutes (no bytes moving on the open request) and Railway would otherwise
    stop the container mid-job — observed as 502 'Application failed to
    respond' on the caller's side."""
    import urllib.request
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if not domain:
        return
    url = f"https://{domain}/health"
    while True:
        time.sleep(45)
        if _tts_busy > 0:
            try:
                urllib.request.urlopen(url, timeout=10)
            except Exception:
                pass

threading.Thread(target=_busy_keepalive, daemon=True).start()

def _tts_idle_reaper():
    """Kill the worker after TTS_IDLE_UNLOAD_SECONDS without jobs — process
    death is what actually returns the RAM to the OS (and stops Railway
    billing it). The next job pays the model load again (~1-2 min on CPU)."""
    while True:
        time.sleep(30)
        with _tts_lock:
            if (
                _worker is not None
                and _tts_busy == 0
                and time.time() - _tts_last_used > TTS_IDLE_UNLOAD_SECONDS
            ):
                _kill_worker()

threading.Thread(target=_tts_idle_reaper, daemon=True).start()


def preprocess_reference(src_path: str, dst_path: str) -> bool:
    """
    Gentle audio cleanup for XTTS v2 reference.
    The goal is to keep the speaker's real timbre intact — over-processing
    (double denoisers, heavy loudnorm) strips the vocal detail XTTS needs to
    sound like the original person and makes the clone sound generic.
    - high-pass 80Hz: remove rumble only, keep low-mid warmth
    - afftdn: single light FFT denoise (nf=-30, gentle)
    - loudnorm: standard -16 LUFS target with wide LRA to preserve dynamics
    - trim silence with relaxed thresholds (don't clip soft speech)
    - 24000 Hz mono WAV (XTTS v2 native)
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-af", (
                    "highpass=f=80,"
                    "afftdn=nf=-30:tr=1,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11,"
                    # Trim leading + trailing silence only (reverse trick) so we
                    # never truncate the reference at an internal pause — XTTS
                    # needs as much real speech as possible to capture the voice.
                    "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                    "areverse,"
                    "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                    "areverse,"
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


# All text processing and synthesis logic lives in tts_worker.py — the API
# process only needs the default style name for its request models.
DEFAULT_STYLE = "reportera"  # calmado | natural | expresivo | energico | reportera


def synthesize(text: str, speaker_wav: str, language: str, output_path: str, style: str = DEFAULT_STYLE):
    """Run one synthesis job on the worker process, holding the busy counter
    (so the idle reaper can't kill the worker mid-job and the busy keepalive
    knows to ping) and the synth gate (one job at a time — queued requests
    wait their turn instead of thrashing the CPU)."""
    global _tts_busy, _tts_last_used
    with _tts_lock:
        _tts_busy += 1
    try:
        with _synth_gate:
            result = _worker_request({
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "output_path": output_path,
                "style": style,
            })
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "error desconocido del worker TTS"))
    finally:
        with _tts_lock:
            _tts_busy -= 1
            _tts_last_used = time.time()


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
# Bump BUILD_VERSION on every deploy so we can confirm from outside that the
# new container has actually rolled out (Railway rebuilds take several minutes).
BUILD_VERSION = "vq-2026-07-07-tts-worker-process"

@app.get("/health")
def health():
    cleanup_old_outputs(max_age_hours=24)  # Clean 24h+ old files
    return {"status": "ok", "build": BUILD_VERSION}

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
    language: str = "es-co"
    style: str = DEFAULT_STYLE  # calmado | natural | expresivo | energico | reportera

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
        synthesize(req.text, speaker_wav, req.language, str(output_path), req.style)
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
    style: str = Form(default=DEFAULT_STYLE),
):
    """Clone voice on the fly without saving. For frontend testing."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    tmp_path = VOICES_DIR / f"tmp_{uuid.uuid4()}{Path(audio.filename).suffix}"
    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.wav"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        synthesize(text, str(tmp_path), language, str(output_path), style)
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
        synthesize(req.text, speaker_wav, req.language, str(output_path), req.style)
    except Exception as e:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error TTS: {str(e)}")
    return FileResponse(path=str(output_path), media_type="audio/wav", filename="speech.wav")
