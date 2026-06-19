"""Capture the robot's mic and expose a live level (waveform) + Silero VAD 'voice detected'.

We read the mic DIRECTLY via ALSA (``arecord``) rather than through the daemon's WebRTC
audio branch: the daemon runs video-only (its audio auto-detect is broken and the failing
source tore down the video pipeline — see docs/camera-webrtc.md), so the Reachy Mini Audio
mic is free for us to capture.

Voice detection uses **Silero VAD** (the vendored ONNX model, run via onnxruntime — no torch),
which is speech-specific; the RMS level is kept only for the visual waveform amplitude.
"""

from __future__ import annotations

import ctypes
import re
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _die_with_parent():
    """preexec: have the child (arecord) get SIGKILL if this process dies, so a hard kill /
    restart never leaves a zombie holding the mic (which would block the next capture)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass

_CARD_NAME = "Reachy Mini Audio"
_RATE = 16000
_CHUNK = 512        # Silero v5 consumes 512 samples at 16 kHz (~32 ms) per step
_CONTEXT = 64       # Silero v5 prepends the previous 64 samples -> model input is 576
_HISTORY = 64       # rolling levels kept for the waveform
_GAIN = 4.0         # RMS -> bar height, for the visual waveform ONLY
_VAD_ON = 0.5       # Silero speech-probability threshold
_HANGOVER = 12      # chunks (~384 ms) to hold 'voice detected' after the last speech frame
_MAX_UTT = _RATE * 30   # cap an utterance at 30 s so the buffer can't grow unbounded

_MODEL = Path(__file__).resolve().parents[2] / "assets" / "silero" / "silero_vad_16k.onnx"


def _card_index() -> int | None:
    """ALSA card index of the Reachy Mini Audio mic (parsed from /proc/asound/cards)."""
    try:
        txt = Path("/proc/asound/cards").read_text()
    except Exception:
        return None
    for line in txt.splitlines():
        if _CARD_NAME in line:
            m = re.match(r"\s*(\d+)\s", line)
            if m:
                return int(m.group(1))
    return None


class AudioMonitor:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._levels = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        self._running = False
        self._active = False
        self._muted = False   # half-duplex: ignore input while the robot is speaking (TTS)
        self._sess: ort.InferenceSession | None = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, _CONTEXT), dtype=np.float32)  # rolling 64-sample context
        # utterance buffering for transcription (consumed by transcribe.StreamingTranscriber)
        self._utt: list[np.ndarray] = []          # chunks of the in-progress utterance
        self._utt_len = 0
        self._final_q: deque = deque(maxlen=8)    # completed utterances awaiting transcription
        self._prev_active = False
        self._utt_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def _load_vad(self) -> None:
        if self._sess is None:
            try:
                ort.set_default_logger_severity(3)  # quiet device-discovery chatter
                so = ort.SessionOptions()
                so.log_severity_level = 3
                self._sess = ort.InferenceSession(
                    str(_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
                )
            except Exception as e:  # noqa: BLE001
                print(f"[audio_monitor] Silero VAD load failed: {e}")
                self._sess = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # reset recurrent state
        self._ctx = np.zeros((1, _CONTEXT), dtype=np.float32)

    def _vad_prob(self, chunk: np.ndarray) -> float:
        if self._sess is None:
            return 0.0
        try:
            x = np.concatenate([self._ctx, chunk.reshape(1, -1)], axis=1).astype(np.float32)  # 64+512
            out, self._state = self._sess.run(
                ["output", "stateN"],
                {"input": x, "state": self._state, "sr": np.array(_RATE, dtype=np.int64)},
            )
            self._ctx = x[:, -_CONTEXT:]                 # carry the last 64 samples forward
            return float(out[0, 0])
        except Exception:
            return 0.0

    def start(self) -> None:
        """Begin capturing the mic + running VAD (idempotent)."""
        with self._lock:
            if self._running:
                return
            idx = _card_index()
            if idx is None:
                return
            self._load_vad()
            try:
                self._proc = subprocess.Popen(
                    ["arecord", "-D", f"plughw:{idx},0", "-f", "S16_LE",
                     "-r", str(_RATE), "-c", "1", "-t", "raw", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    preexec_fn=_die_with_parent,
                )
            except Exception:
                self._proc = None
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        proc = self._proc
        nbytes = _CHUNK * 2  # int16
        since = _HANGOVER    # start inactive
        while self._running and proc is not None and proc.stdout is not None:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            f32 = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f32 * f32)))
            self._levels.append(min(1.0, rms * _GAIN))        # waveform amplitude
            if self._muted:                                     # robot is speaking -> ignore input
                self._active = False
                since = _HANGOVER
                continue
            prob = self._vad_prob(f32)                          # speech probability
            since = 0 if prob >= _VAD_ON else since + 1
            self._active = since < _HANGOVER
            self._buffer_utterance(f32)
        self._running = False
        self._active = False

    def _buffer_utterance(self, f32: np.ndarray) -> None:
        """Accumulate audio while speaking; queue the utterance when speech ends (or caps out)."""
        with self._utt_lock:
            if self._active:
                self._utt.append(f32.copy())
                self._utt_len += len(f32)
                if self._utt_len >= _MAX_UTT:           # force-flush over-long utterances
                    self._final_q.append(np.concatenate(self._utt))
                    self._utt, self._utt_len = [], 0
            elif self._prev_active and self._utt:        # speech just ended -> finalize
                self._final_q.append(np.concatenate(self._utt))
                self._utt, self._utt_len = [], 0
            self._prev_active = self._active

    def mute(self, on: bool) -> None:
        """Half-duplex: when muted, drop the in-progress utterance and ignore input (TTS playback)."""
        self._muted = bool(on)
        if on:
            with self._utt_lock:
                self._utt, self._utt_len = [], 0
                self._prev_active = False

    def partial_audio(self) -> np.ndarray | None:
        """Audio of the in-progress utterance (for streaming partials), or None if not speaking."""
        with self._utt_lock:
            return np.concatenate(self._utt) if self._utt else None

    def pop_final(self) -> np.ndarray | None:
        """The next completed utterance to transcribe, or None."""
        with self._utt_lock:
            return self._final_q.popleft() if self._final_q else None

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._active = False
            self._muted = False
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            self._levels = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        with self._utt_lock:
            self._utt, self._utt_len = [], 0
            self._final_q.clear()
            self._prev_active = False

    def snapshot(self) -> dict:
        """The rolling waveform, the latest level, and the Silero 'voice detected' flag."""
        lv = list(self._levels)
        level = lv[-1] if lv else 0.0
        return {
            "levels": [round(x, 3) for x in lv],
            "level": round(level, 3),
            "active": bool(self._active),
        }


monitor = AudioMonitor()
