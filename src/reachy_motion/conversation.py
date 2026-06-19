"""Voice conversation loop: user utterance -> responder -> spoken reply.

Ties together the mic/VAD (audio_monitor), STT (transcribe), a **responder**, and TTS (tts).
For now the responder just echoes ("repeat what we say"); swapping in an LLM later is a
one-line change (``conversation.set_responder(LLMResponder())``) — nothing else moves.

Half-duplex: while the robot speaks, the mic is muted so it doesn't transcribe itself.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque

from . import audio_monitor, transcribe, tts


class Responder:
    """text in -> reply text out. Subclass to plug in an LLM later."""

    def reply(self, text: str) -> str:
        raise NotImplementedError


class EchoResponder(Responder):
    """Repeat what was said (placeholder until the LLM lands)."""

    def reply(self, text: str) -> str:
        return text


class Conversation:
    def __init__(self) -> None:
        self._responder: Responder = EchoResponder()
        self._q: queue.Queue[str] = queue.Queue()
        self._dialogue: deque[dict] = deque(maxlen=8)   # {role: 'user'|'bot', text}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._speaking = False

    def set_responder(self, responder: Responder) -> None:
        self._responder = responder

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        transcribe.transcriber.on_final = self.handle_user   # wire the STT -> responder seam
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def handle_user(self, text: str) -> None:
        """Called (from the STT worker) when a user utterance finalizes."""
        text = (text or "").strip()
        if text:
            self._q.put(text)

    def _run(self) -> None:
        tts.warmup()   # preload the voice in the worker (off the timer/UI path)
        while self._running:
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                self._dialogue.append({"role": "user", "text": text})
            try:
                reply = (self._responder.reply(text) or "").strip()
            except Exception as e:  # noqa: BLE001
                print(f"[conversation] responder error: {e}")
                reply = ""
            if not reply:
                continue
            with self._lock:
                self._dialogue.append({"role": "bot", "text": reply})
                self._speaking = True
            audio_monitor.monitor.mute(True)         # don't hear ourselves
            try:
                tts.speak(reply)                      # blocks until playback finishes
            except Exception as e:  # noqa: BLE001
                print(f"[conversation] tts error: {e}")
            time.sleep(0.25)                          # let the speaker tail decay before listening
            audio_monitor.monitor.mute(False)
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._dialogue.clear()
            self._speaking = False
        transcribe.transcriber.on_final = None
        try:
            audio_monitor.monitor.mute(False)
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._lock:
            return {"dialogue": [dict(d) for d in self._dialogue], "speaking": self._speaking}


conversation = Conversation()
