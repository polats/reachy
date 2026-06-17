"""Play sound on the Reachy Mini speaker via direct ALSA.

Why not the SDK's ``mini.media.play_sound``? On this machine the SDK media stack needs
the GStreamer webrtc rust plugin, which isn't installed, so the daemon's media server
fails and the ``local`` client backend (camera+audio) can't initialize. The robot's
speaker is just a USB audio card ("Reachy Mini Audio"), so we play to it directly with
``aplay`` — independent of the daemon, GStreamer, and the motion control path.

This is the sound half of synchronized motion+sound behaviors: a generated move's paired
WAV plays here while the move plays through the daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np

SPEAKER_NAME = "Reachy Mini Audio"


def find_speaker_device(name: str = SPEAKER_NAME) -> Optional[str]:
    """Return an ALSA device string (e.g. ``plughw:5,0``) for the robot speaker.

    Resolves the card by *name* (via ``aplay -l``) so it survives card-number changes.
    Returns ``None`` if not found.
    """
    aplay = shutil.which("aplay")
    if not aplay:
        return None
    out = subprocess.run([aplay, "-l"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        # e.g. "card 5: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]"
        if name in line and line.strip().startswith("card "):
            try:
                card = line.split("card ", 1)[1].split(":", 1)[0].strip()
                dev = line.split("device ", 1)[1].split(":", 1)[0].strip()
                return f"plughw:{card},{dev}"
            except (IndexError, ValueError):
                continue
    return None


def play_wav(path: str | Path, *, device: Optional[str] = None, blocking: bool = True):
    """Play a WAV file on the robot speaker. Returns the Popen if non-blocking."""
    device = device or find_speaker_device()
    if device is None:
        raise RuntimeError(f"speaker '{SPEAKER_NAME}' not found in `aplay -l`")
    aplay = shutil.which("aplay")
    if not aplay:
        raise RuntimeError("`aplay` not found (install alsa-utils)")
    proc = subprocess.Popen(
        [aplay, "-q", "-D", device, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if blocking:
        proc.wait()
        return None
    return proc


def play_samples(
    samples: np.ndarray,
    samplerate: int,
    *,
    device: Optional[str] = None,
    blocking: bool = True,
):
    """Play a numpy PCM buffer (float32 [-1,1] or int16), mono or [N, channels].

    This is the entry point TTS will use: synthesize -> ``play_samples`` while a move
    plays. Streams raw PCM straight to ``aplay``.
    """
    device = device or find_speaker_device()
    if device is None:
        raise RuntimeError(f"speaker '{SPEAKER_NAME}' not found in `aplay -l`")
    aplay = shutil.which("aplay")
    if not aplay:
        raise RuntimeError("`aplay` not found (install alsa-utils)")

    arr = np.asarray(samples)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    channels = 1 if arr.ndim == 1 else arr.shape[1]
    proc = subprocess.Popen(
        [aplay, "-q", "-D", device, "-f", "S16_LE", "-r", str(samplerate), "-c", str(channels)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    if blocking:
        proc.wait()
        return None
    return proc


def wav_duration(path: str | Path) -> float:
    """Duration of a WAV file in seconds."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
