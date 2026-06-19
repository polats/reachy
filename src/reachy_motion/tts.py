"""Text-to-speech for the robot via kokoro-onnx (onnxruntime, no torch).

Synthesizes speech and plays it through the robot speaker (the ALSA path in audio.py).
The model loads lazily and auto-downloads on first use (kept out of git, ~336 MB).
Exposed as a simple ``speak()`` so the conversation loop can swap echo -> LLM later
without touching this module.
"""

from __future__ import annotations

import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

import numpy as np

from . import audio

_DIR = Path(__file__).resolve().parents[2] / "assets" / "kokoro"
_MODEL = _DIR / "kokoro-v1.0.onnx"
_VOICES = _DIR / "voices-v1.0.bin"
_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_LANG = "en-us"
# English voices only (our lang is en-us; a*=US, b*=UK; m=male, f=female). Default = young US male.
VOICES = [
    "am_puck", "am_liam", "am_echo", "am_eric", "am_fenrir", "am_onyx", "am_michael", "am_adam", "am_santa",
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_aoede", "af_kore", "af_sky", "af_nova",
    "af_alloy", "af_jessica", "af_river",
    "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
]
_VOICE = "am_puck"
_voice = [_VOICE]    # current voice (mutable; set via set_voice from the UI dropdown)


def list_voices() -> list[str]:
    return list(VOICES)


def current_voice() -> str:
    return _voice[0]


def set_voice(name: str) -> None:
    if name in VOICES:
        _voice[0] = name

_kokoro = None
_load_lock = threading.Lock()
_speaking = threading.Event()


def is_speaking() -> bool:
    return _speaking.is_set()


def _ensure_model() -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    for path in (_MODEL, _VOICES):
        if not path.exists():
            print(f"[tts] downloading {path.name} …", flush=True)
            urllib.request.urlretrieve(f"{_BASE}/{path.name}", path)


def _load():
    global _kokoro
    if _kokoro is None:
        _ensure_model()
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(_MODEL), str(_VOICES))
    return _kokoro


def warmup() -> None:
    """Load the model ahead of time (off the first-reply latency path)."""
    try:
        _load()
    except Exception as e:  # noqa: BLE001
        print(f"[tts] warmup failed: {e}")


def speak(text: str, *, voice: str | None = None, blocking: bool = True) -> None:
    """Synthesize ``text`` and play it on the robot speaker (uses the current voice if unset)."""
    text = (text or "").strip()
    if not text:
        return
    with _load_lock:
        samples, sr = _load().create(text, voice=voice or _voice[0], speed=1.0, lang=_LANG)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    _speaking.set()
    try:
        audio.play_wav(path, blocking=blocking)
    finally:
        _speaking.clear()
        try:
            Path(path).unlink()
        except Exception:
            pass
