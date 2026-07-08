"""
XTTS synthesis worker — runs in its OWN process, spawned by main.py.

Why a separate process: CPython never returns freed heap pages to the OS, so
once XTTS (~5 GB) has been loaded, the process RSS stays high even after an
in-process "unload" (del + gc.collect) — and Railway bills resident memory
per minute (observed: a flat ~8 GB for days ≈ $80/month of RAM alone). The
API process kills this worker after an idle period; process death is the only
reliable way to hand the memory back. While alive, the worker keeps the model
loaded, so a batch of consecutive jobs pays the model load only once.

Protocol: one JSON job per line on stdin → one JSON reply per line on the
ORIGINAL stdout. Coqui TTS prints progress chatter to stdout, which would
corrupt the protocol, so at startup we duplicate the real stdout for replies
and point fd 1 (and sys.stdout) at stderr — library output lands in the
container log instead.

Job:   {"text", "speaker_wav", "language", "output_path", "style"}
Reply: {"ok": true} | {"ok": false, "error": "..."}
"""
import os
import sys
import json

# Reserve the real stdout for protocol replies; everything else → stderr.
_PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import re
import uuid
import shutil
import subprocess
from pathlib import Path

# Patch torch.load before any TTS import — PyTorch 2.6 changed weights_only default
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Inference-only worker: never build autograd graphs. Cuts peak RAM during
# synthesis substantially (no activation buffers kept for backward).
torch.set_grad_enabled(False)

# Cap torch threads to the Railway vCPU limit. Inside a container torch sees
# the HOST machine's cores and spawns that many threads; the cgroup then
# throttles them and synthesis crawls (observed real-time factor ~54 vs the
# ~3-5 expected on 8 vCPU).
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))

_tts = None

def get_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts


def sanitize_for_tts(text: str) -> str:
    """
    Reduce punctuation to the small set XTTS actually intones well: . , ! ? ¿ ¡
    Anything outside that set the model tends to READ ALOUD as a word
    ("punto punto punto", "asterisco", "guión"...) instead of pausing.
    - ellipsis / repeated dots → single period
    - ; : — – and spaced hyphens → comma (same pause role)
    - quotes, brackets, markdown symbols → removed
    - repeated terminal marks (!!! ?? ?!) → single mark
    """
    t = text
    t = t.replace('…', '.')
    t = re.sub(r'\.{2,}', '.', t)
    # Semicolon, colon and dashes act as a comma-length pause.
    t = re.sub(r'\s*[;:—–]\s*', ', ', t)
    # Hyphen used as a pause (spaced); keep in-word hyphens (bien-estar).
    t = re.sub(r'\s+-\s+', ', ', t)
    # Symbols XTTS speaks literally — drop them (keep apostrophes for
    # contractions like "don't").
    t = re.sub(r'[«»“”"`´\(\)\[\]\{\}\*_#~\^<>|\\/=+]', ' ', t)
    # Collapse repeated/mixed terminal marks: "!!!" → "!", "?!" → "?".
    t = re.sub(r'([!?])[!?]+', r'\1', t)
    # Clean up stray punctuation left by the removals.
    t = re.sub(r'\s*,(\s*,)+', ',', t)
    t = re.sub(r',\s*([.!?])', r'\1', t)
    t = re.sub(r'([.!?])\s*,', r'\1 ', t)
    t = re.sub(r'\s+([.,!?])', r'\1', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def normalize_text(text: str) -> str:
    text = sanitize_for_tts(text)
    if text and text[-1] not in '.!?,;:':
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
                    # Every chunk must end in a terminal mark: without a stop
                    # signal the XTTS decoder keeps generating past the text
                    # (hallucinated words). The last fragment gets the real
                    # terminal restored below.
                    if not w_is_last:
                        w += '.'
                    clause_items.append([w, pause])
            else:
                # End non-terminal clauses with a PERIOD, not a comma — a
                # trailing comma leaves the utterance "open" and XTTS keeps
                # generating (invented words after the pause). The comma's
                # pause is still honored as the timed gap in _concat_wavs.
                text_piece = part if is_last else f"{part}."
                pause = SENTENCE_PAUSE if is_last else COMMA_PAUSE
                clause_items.append([text_piece, pause])
        # Restore the sentence's terminal mark on the very last fragment.
        if clause_items and terminal and clause_items[-1][0] and clause_items[-1][0][-1] not in '.!?…':
            clause_items[-1][0] += terminal
        chunks.extend((t, p) for t, p in clause_items if t)
    # Never send a punctuation-only chunk to XTTS — with no real words the
    # model "reads" the marks aloud or hallucinates from silence.
    return [(t, p) for t, p in chunks if re.search(r'\w', t)]


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
    # Entertainment-reporter delivery: max SAFE sampling energy (still at the
    # 0.85 / 2.0 hallucination cap) + faster pace. Most of the "excited" feel
    # comes from the text transform in _synthesize_chunk (declaratives are
    # read as exclamations), which adds energy WITHOUT touching sampling.
    "reportera": {"temperature": 0.85, "repetition_penalty": 2.1, "top_k": 60, "top_p": 0.90, "speed": 1.07},
}
DEFAULT_STYLE = "reportera"

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
    if style == "reportera" and text.endswith('.'):
        # Showbiz-reporter energy: read declaratives with exclamation prosody.
        # This is a TEXT-level intonation boost (like the ¡/¿ marks) — it adds
        # excitement without raising sampling randomness, so no extra
        # hallucination risk. Questions keep their '?' untouched.
        text = text[:-1] + '!'
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


def _adjust_tempo_to_natural(text: str, audio_path: str, output_path: str, natural_high: float = 3.3) -> bool:
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

    NATURAL_LOW, NATURAL_HIGH = 2.3, natural_high  # words per second
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
    # Reporters speak faster — let the reportera style keep its pace instead
    # of getting slowed back down by the natural-band correction.
    natural_high = 3.7 if style == "reportera" else 3.3

    if len(items) == 1:
        chunk_text, _ = items[0]
        raw = str(Path(output_path).parent / f"raw_{uuid.uuid4().hex[:6]}.wav")
        _synthesize_chunk(tts, chunk_text, speaker_wav, language, raw, style)
        # Keep tempo natural; fall back to the raw chunk before cleaning it up.
        if not _adjust_tempo_to_natural(chunk_text, raw, output_path, natural_high):
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
        if not _adjust_tempo_to_natural(text, concat_tmp, output_path, natural_high):
            shutil.copy(concat_tmp, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Job loop: one JSON job per stdin line, one JSON reply per job."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            synthesize(
                job["text"],
                job["speaker_wav"],
                job["language"],
                job["output_path"],
                job.get("style", DEFAULT_STYLE),
            )
            reply = {"ok": True}
        except Exception as e:
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _PROTO.write(json.dumps(reply) + "\n")
        _PROTO.flush()


if __name__ == "__main__":
    main()
