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


import re

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in '.!?,;:…':
        text += '.'
    return text


# XTTS v2 only accepts these base language codes (no regional variants).
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}

def normalize_language(lang: str) -> str:
    """
    Map a possibly regional language code to one XTTS v2 understands.
    The UI may send variants like 'es-co' (Colombian), 'es-mx', 'pt-br' — XTTS
    only knows the base language ('es', 'pt', ...), so strip the region.
    Chinese is the one exception: its XTTS code is 'zh-cn'.
    """
    if not lang:
        return "es"
    lang = lang.strip().lower()
    if lang in XTTS_LANGS:
        return lang
    base = lang.split("-")[0]
    if base == "zh":
        return "zh-cn"
    if base in XTTS_LANGS:
        return base
    return "es"  # safe default


def _split_by_words(text: str, max_chars: int) -> list:
    """Last-resort split for a fragment with no commas: break on word boundaries
    so no chunk ever exceeds max_chars (XTTS throws 'index out of range in self'
    when a chunk is longer than the model's token limit)."""
    words = text.split()
    out, current = [], ""
    for w in words:
        candidate = (current + " " + w).strip() if current else w
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            # A single word longer than max_chars: hard-slice it.
            while len(w) > max_chars:
                out.append(w[:max_chars])
                w = w[max_chars:]
            current = w
    if current:
        out.append(current)
    return out


# Pause durations inserted between synthesized chunks (see _concat_wavs).
# A period/question/exclamation gets a real breath pause; a comma gets a
# shorter one; a hard word-limit split (not a natural pause point) gets
# just enough to stop words from mashing together.
SENTENCE_PAUSE = 0.32
COMMA_PAUSE = 0.16
HARD_SPLIT_PAUSE = 0.06

# Only split a sentence on its internal commas if it's long enough that the
# extra synthesis calls are worth it — fragmenting short sentences into tiny
# chunks adds latency and raises XTTS hallucination risk without gaining
# much, since a short sentence is already read at a natural pace in one go.
COMMA_SPLIT_MIN_LEN = 70


def split_into_chunks(text: str, max_chars: int = 200) -> list:
    """
    Split text into synthesis chunks for XTTS, each paired with the pause
    that should follow it. PRESERVES the punctuation that drives intonation
    (¿ ? ¡ ! …) — terminal marks are what make XTTS raise pitch on questions
    and add energy on exclamations, so we never strip them.

    Returns a list of (chunk_text, pause_after_seconds) tuples.

    - First split on sentence-ending punctuation (. ! ? …), keeping the mark.
    - Long sentences are further split on commas so the comma actually gets
      a pause instead of being read straight through.
    - Any fragment still over max_chars is hard-split on words (prevents the
      'index out of range' XTTS error on very long comma-less runs).
    """
    raw = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks = []
    for sentence in raw:
        sentence = sentence.strip()
        if not sentence:
            continue
        terminal = sentence[-1] if sentence[-1] in '.!?…' else ''

        if len(sentence) <= max_chars and (
            len(sentence) <= COMMA_SPLIT_MIN_LEN or ',' not in sentence
        ):
            chunks.append((sentence, SENTENCE_PAUSE))
            continue

        # Split into comma-clauses (also covers the over-max_chars case).
        parts = [p.strip() for p in re.split(r',\s*', sentence) if p.strip()]
        clause_items = []  # (text, pause_after)
        n = len(parts)
        for i, part in enumerate(parts):
            is_last = i == n - 1
            if len(part) > max_chars:
                words = _split_by_words(part, max_chars)
                for j, w in enumerate(words):
                    w_is_last = is_last and j == len(words) - 1
                    pause = SENTENCE_PAUSE if w_is_last else (
                        COMMA_PAUSE if is_last else HARD_SPLIT_PAUSE
                    )
                    clause_items.append([w, pause])
            else:
                # Keep the comma on non-terminal clauses so XTTS still hears
                # it and produces the natural "continuing" intonation before
                # we cut and insert our own timed pause.
                text_piece = part if is_last else f"{part},"
                pause = SENTENCE_PAUSE if is_last else COMMA_PAUSE
                clause_items.append([text_piece, pause])
        # Restore the sentence's terminal mark on the very last fragment.
        if clause_items and terminal and clause_items[-1][0] and clause_items[-1][0][-1] not in '.!?…':
            clause_items[-1][0] += terminal
        chunks.extend((t, p) for t, p in clause_items if t)
    return chunks


# Expressiveness presets. Higher temperature + lower repetition_penalty =
# more prosody variation, but also more risk of XTTS hallucinating extra
# words after the intended text ends — that failure mode gets worse fast
# above ~0.85 temperature / below ~2.0 repetition_penalty, so the top presets
# are capped there instead of chasing maximum expressiveness.
STYLE_PRESETS = {
    "calmado":   {"temperature": 0.55, "repetition_penalty": 3.5, "top_k": 45, "top_p": 0.82, "speed": 0.97},
    "natural":   {"temperature": 0.68, "repetition_penalty": 2.8, "top_k": 50, "top_p": 0.85, "speed": 1.0},
    "expresivo": {"temperature": 0.78, "repetition_penalty": 2.4, "top_k": 55, "top_p": 0.88, "speed": 1.0},
    "energico":  {"temperature": 0.85, "repetition_penalty": 2.1, "top_k": 60, "top_p": 0.90, "speed": 1.05},
}
DEFAULT_STYLE = "expresivo"

# Conservative fallback params used when a chunk fails to generate (e.g. XTTS
# 'index out of range in self' from runaway generation at high temperature).
_SAFE_PARAMS = {"temperature": 0.65, "repetition_penalty": 3.0, "top_k": 50, "top_p": 0.85, "speed": 1.0}


def _ensure_spanish_marks(text: str, language: str) -> str:
    """
    XTTS intones Spanish questions/exclamations far better when the OPENING
    mark (¿ / ¡) is present — the closing ? / ! alone barely changes the
    prosody. If the writer only used the closing mark, add the opening one so
    the model actually raises the pitch on questions and adds energy on
    exclamations.
    """
    if not language.startswith("es"):
        return text
    t = text.strip()
    if not t:
        return t
    if t.endswith("?") and "¿" not in t:
        t = "¿" + t
    elif t.endswith("!") and "¡" not in t:
        t = "¡" + t
    return t


def _synthesize_chunk(tts, text: str, speaker_wav: str, language: str, out_path: str, style: str = DEFAULT_STYLE):
    """Generate a single short chunk — no internal splitting.
    Retries once with conservative params if XTTS throws (high-temperature
    runaway can raise 'index out of range in self')."""
    text = _ensure_spanish_marks(text, language)
    # NOTE: we intentionally do NOT boost temperature/lower repetition_penalty
    # for ?/! chunks anymore — that combo made XTTS hallucinate extra words
    # after the intended text (more randomness = more runaway generation).
    # The ¿/¡ opening mark above already drives most of the intonation gain
    # without touching sampling params, so it doesn't add hallucination risk.
    p = dict(STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE]))
    for params in (p, _SAFE_PARAMS):
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=out_path,
                temperature=params["temperature"],
                length_penalty=1.0,
                repetition_penalty=params["repetition_penalty"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                speed=params["speed"],
                enable_text_splitting=False,  # we handle splitting ourselves
                # Voice-conditioning tuning: use up to 30s of the reference
                # (in 6s windows) so XTTS captures more of the real timbre
                # instead of just the first ~4s. sound_norm_refs re-normalizes
                # the reference internally for more consistent cloning.
                gpt_cond_len=30,
                gpt_cond_chunk_len=6,
                sound_norm_refs=True,
            )
            return
        except Exception:
            if params is _SAFE_PARAMS:
                raise  # both attempts failed — propagate


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
                    # Trim ONLY the leading and trailing silence. Using
                    # stop_periods=1 here would cut the chunk at its first
                    # internal pause (comma/breath); reverse+trim+reverse trims
                    # the tail without touching pauses inside the speech.
                    "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
                    "areverse,"
                    "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
                    "areverse"
                ),
                dst_path,
            ],
            check=True, capture_output=True, timeout=30,
        )
        return Path(dst_path).exists() and Path(dst_path).stat().st_size > 500
    except Exception:
        return False


def _concat_wavs(part_paths: list, gaps: list, output_path: str):
    """
    Concatenate WAV files with a natural pause between chunks. Crossfade
    would cut speech — instead we trim silence per chunk (via _trim_silence,
    so no doubled-up pause) then add an explicit silent gap after each chunk.
    `gaps` has one entry per part in part_paths; the gap after the LAST part
    is ignored (nothing follows it).
    """
    if len(part_paths) == 1:
        shutil.copy(part_paths[0], output_path)
        return
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in part_paths:
        args += ["-i", p]
    # Normalize each chunk; pad every chunk but the last with its own gap
    # duration so pauses reflect whether they followed a comma or a period.
    norm_parts = []
    for i in range(len(part_paths)):
        chain = f"[{i}:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono"
        if i < len(part_paths) - 1:
            chain += f",apad=pad_dur={gaps[i]}"
        chain += f"[n{i}]"
        norm_parts.append(chain)
    norm = ";".join(norm_parts)
    parts_label = "".join(f"[n{i}]" for i in range(len(part_paths)))
    filter_str = f"{norm};{parts_label}concat=n={len(part_paths)}:v=0:a=1[out]"
    args += ["-filter_complex", filter_str, "-map", "[out]", output_path]
    subprocess.run(args, check=True, capture_output=True, timeout=120)


def _adjust_tempo_to_natural(text: str, audio_path: str, output_path: str) -> bool:
    """
    Keep the reading speed in a natural band instead of forcing an exact rate.
    Forcing every clip to a fixed words-per-second with atempo time-stretches
    the whole audio, which is what makes XTTS output sound robotic. Natural
    Spanish narration sits around 2.3–3.3 wps, so we only nudge clips that fall
    clearly outside that band, and we cap the correction to ±12% so the stretch
    stays inaudible (atempo only sounds clean very close to 1.0x).
    """
    words = len(text.split())
    if words == 0:
        shutil.copy(audio_path, output_path)
        return True

    NATURAL_LOW, NATURAL_HIGH = 2.3, 3.3  # words per second
    TEMPO_MIN, TEMPO_MAX = 0.9, 1.12      # ±~12% — stays transparent

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", audio_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(probe.stdout)
        actual_duration = float(data.get("format", {}).get("duration", 0))
        if actual_duration <= 0:
            shutil.copy(audio_path, output_path)
            return True

        wps = words / actual_duration
        if NATURAL_LOW <= wps <= NATURAL_HIGH:
            # Already natural — leave the prosody untouched.
            shutil.copy(audio_path, output_path)
            return True

        target = NATURAL_LOW if wps < NATURAL_LOW else NATURAL_HIGH
        tempo = max(TEMPO_MIN, min(TEMPO_MAX, target / wps))
        if abs(tempo - 1.0) < 0.02:
            shutil.copy(audio_path, output_path)
            return True

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
             "-af", f"atempo={tempo}",
             "-ar", "24000", "-ac", "1",
             output_path],
            check=True, capture_output=True, timeout=30
        )
        return Path(output_path).exists() and Path(output_path).stat().st_size > 500
    except Exception:
        return False


def synthesize(text: str, speaker_wav: str, language: str, output_path: str, style: str = DEFAULT_STYLE):
    """
    Split text into chunks at sentence AND comma boundaries, generate each
    with XTTS, concat with a pause sized to the punctuation that produced the
    split (comma = short pause, period/!/? = longer pause), then keep the
    overall reading speed within a natural band.
    `style` selects an expressiveness preset (see STYLE_PRESETS).
    """
    tts = get_tts()
    language = normalize_language(language)
    text = normalize_text(text)
    items = split_into_chunks(text)  # list of (chunk_text, pause_after_seconds)

    if len(items) == 1:
        chunk_text, _ = items[0]
        raw = str(Path(output_path).parent / f"raw_{uuid.uuid4().hex[:6]}.wav")
        _synthesize_chunk(tts, chunk_text, speaker_wav, language, raw, style)
        # Keep tempo natural; fall back to the raw chunk before cleaning it up.
        if not _adjust_tempo_to_natural(chunk_text, raw, output_path):
            shutil.copy(raw, output_path)
        Path(raw).unlink(missing_ok=True)
        return

    tmp_dir = Path(output_path).parent / f"synth_tmp_{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(exist_ok=True)
    part_paths, gaps = [], []
    try:
        for i, (chunk_text, pause_after) in enumerate(items):
            raw = str(tmp_dir / f"raw_{i:03d}.wav")
            trimmed = str(tmp_dir / f"part_{i:03d}.wav")
            _synthesize_chunk(tts, chunk_text, speaker_wav, language, raw, style)
            # Trim XTTS silence padding; fall back to raw if trim fails
            if not _trim_silence(raw, trimmed):
                trimmed = raw
            part_paths.append(trimmed)
            gaps.append(pause_after)
        concat_tmp = str(tmp_dir / "concat.wav")
        _concat_wavs(part_paths, gaps, concat_tmp)
        # Keep the final reading speed natural (gentle, only if out of band).
        if not _adjust_tempo_to_natural(text, concat_tmp, output_path):
            shutil.copy(concat_tmp, output_path)
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
# Bump BUILD_VERSION on every deploy so we can confirm from outside that the
# new container has actually rolled out (Railway rebuilds take several minutes).
BUILD_VERSION = "vq-2026-06-30-commapauses"

@app.get("/health")
def health():
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
    style: str = DEFAULT_STYLE  # calmado | natural | expresivo | energico

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
