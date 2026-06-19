"""Streaming speech-to-text over the mic, displayed live in the viewer.

Reuses the mic + Silero VAD in :mod:`reachy_motion.audio_monitor`: while you speak, a
background worker re-decodes the in-progress utterance every ~0.6 s with faster-whisper
(GPU) and publishes it as **interim** text; when the VAD says speech ended, the completed
utterance is decoded once more and **committed** as a final line. This is the pseudo-streaming
Whisper approach (WhisperLive-style) — Whisper-quality text that appears as you talk.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from . import audio_monitor

_MODEL = "large-v3-turbo"   # near-large-v3 quality, ~4x faster -> good for repeated partial decodes
_DEVICE = "cuda"
_COMPUTE = "float16"
_LANG = "en"
_PARTIAL_EVERY = 0.6        # seconds between partial re-decodes
_MIN_SAMPLES = 8000         # ~0.5 s: ignore utterances/partials shorter than this
_HISTORY = 4                # committed lines kept for display


class StreamingTranscriber:
    def __init__(self) -> None:
        self._model = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._committed: deque[str] = deque(maxlen=_HISTORY)
        self._interim = ""
        self._ready = False
        self.on_final = None   # optional callback(text) when a user utterance finalizes (the responder seam)

    @property
    def running(self) -> bool:
        return self._running

    def _load(self) -> bool:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(_MODEL, device=_DEVICE, compute_type=_COMPUTE)
            except Exception as e:  # noqa: BLE001
                print(f"[transcribe] model load failed: {e}")
                return False
        return True

    def _decode(self, audio) -> str:
        try:
            segs, _ = self._model.transcribe(audio, beam_size=1, language=_LANG, vad_filter=False)
            return " ".join(s.text for s in segs).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[transcribe] decode error: {e}")
            return ""

    def start(self) -> None:
        """Spin up the worker (idempotent). The model loads lazily inside the worker."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if not self._load():
            self._running = False
            return
        with self._lock:
            self._ready = True
        last = 0.0
        while self._running:
            final = audio_monitor.monitor.pop_final()
            if final is not None:
                if len(final) >= _MIN_SAMPLES:
                    text = self._decode(final)
                    with self._lock:
                        if text:
                            self._committed.append(text)
                        self._interim = ""
                    if text and self.on_final:        # hand the utterance to the responder/TTS loop
                        try:
                            self.on_final(text)
                        except Exception as e:  # noqa: BLE001
                            print(f"[transcribe] on_final error: {e}")
                else:
                    with self._lock:
                        self._interim = ""
                continue
            part = audio_monitor.monitor.partial_audio()
            now = time.monotonic()
            if part is not None and len(part) >= _MIN_SAMPLES and (now - last) >= _PARTIAL_EVERY:
                text = self._decode(part)
                last = now
                with self._lock:
                    self._interim = text
            elif part is None:
                with self._lock:
                    self._interim = ""
                time.sleep(0.05)
            else:
                time.sleep(0.05)
        self._running = False

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._interim = ""
            self._committed.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {"committed": list(self._committed), "interim": self._interim,
                    "stt_ready": self._ready}


transcriber = StreamingTranscriber()
