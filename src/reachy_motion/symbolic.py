"""Symbolic (procedural) motion — Phase 1 text-to-motion target.

A move is described by a compact, LLM-emittable JSON "spec": each robot channel is
the sum of one or more oscillator terms (sine/cosine/triangle/square) plus a constant
offset, optionally shaped by a fade-in/out envelope. This is the format an LLM produces
from a text prompt (see the SDK's ``skills/symbolic-motion.md``), and it bakes losslessly
into the canonical recorded-move schema (:mod:`reachy_motion.schema`).

Channels (LLM-friendly units):
    head_x, head_y, head_z          translation offset, millimetres
    head_roll, head_pitch, head_yaw head orientation, degrees
    antenna_left, antenna_right     antenna angle, degrees
    body_yaw                        body rotation, degrees

Example spec::

    {
      "description": "curious head tilt with perky antennas",
      "duration": 3.0,
      "envelope": "hann",
      "channels": {
        "head_pitch":    [{"amp": 12, "freq": 0.4, "phase": 0}],
        "head_yaw":      [{"amp": 18, "freq": 0.3, "phase": 90}],
        "antenna_left":  [{"amp": 25, "freq": 0.8, "phase": 0}],
        "antenna_right": [{"amp": 25, "freq": 0.8, "phase": 180}],
        "body_yaw":      [{"offset": 10}]
      }
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose

from .schema import MoveData

CHANNELS = (
    "head_x", "head_y", "head_z",
    "head_roll", "head_pitch", "head_yaw",
    "antenna_left", "antenna_right",
    "body_yaw",
)

_SHAPES = ("sin", "cos", "triangle", "square")


def _osc(shape: str, phase_rad: float, x: float) -> float:
    """Unit-amplitude oscillator value; x is the phase argument (2*pi*f*t + phase)."""
    if shape == "sin":
        return math.sin(x)
    if shape == "cos":
        return math.cos(x)
    if shape == "triangle":
        # period 2*pi, range [-1, 1]
        return 2.0 / math.pi * math.asin(math.sin(x))
    if shape == "square":
        return 1.0 if math.sin(x) >= 0 else -1.0
    raise ValueError(f"unknown shape {shape!r}, expected one of {_SHAPES}")


def _eval_channel(terms: list[dict[str, Any]], t: float) -> float:
    """Sum of oscillator terms + constant offsets for a channel at time t."""
    total = 0.0
    for term in terms:
        total += float(term.get("offset", 0.0))
        amp = float(term.get("amp", 0.0))
        if amp == 0.0:
            continue
        freq = float(term.get("freq", 0.0))
        phase = math.radians(float(term.get("phase", 0.0)))
        shape = term.get("shape", "sin")
        total += amp * _osc(shape, phase, 2.0 * math.pi * freq * t + phase)
    return total


def _envelope(name: Optional[str], t: float, duration: float) -> float:
    """Amplitude scaling in [0, 1] applied to all oscillating (non-offset) motion."""
    if not name or name == "none" or duration <= 0:
        return 1.0
    frac = min(max(t / duration, 0.0), 1.0)
    if name == "hann":  # smooth rise and fall
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * frac))
    if name == "fade_in":
        return frac
    if name == "fade_out":
        return 1.0 - frac
    if name == "ease":  # quick ease-in, hold, ease-out
        edge = 0.15
        if frac < edge:
            return frac / edge
        if frac > 1.0 - edge:
            return (1.0 - frac) / edge
        return 1.0
    raise ValueError(f"unknown envelope {name!r}")


@dataclass
class SymbolicMove(Move):
    """A :class:`reachy_mini.motion.move.Move` defined by a symbolic spec.

    Evaluates in real time (so it can drive ``play_move`` directly at ~100 Hz) and can
    be baked into the canonical keyframe schema via :meth:`bake`.
    """

    spec: dict[str, Any]
    _duration: float = field(init=False)
    _channels: dict[str, list[dict[str, Any]]] = field(init=False)
    _envelope: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        self._duration = float(self.spec.get("duration", 3.0))
        self._envelope = self.spec.get("envelope")
        chans = self.spec.get("channels", {})
        unknown = set(chans) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channel(s): {sorted(unknown)}; valid: {CHANNELS}")
        # normalize: every term-list, allow a bare dict for a single term
        self._channels = {
            k: ([v] if isinstance(v, dict) else list(v)) for k, v in chans.items()
        }

    @property
    def description(self) -> str:
        return str(self.spec.get("description", ""))

    @property
    def duration(self) -> float:
        return self._duration

    def _channel_value(self, name: str, t: float, env: float) -> float:
        terms = self._channels.get(name)
        if not terms:
            return 0.0
        # Constant offsets are not enveloped; oscillating part is.
        offset = sum(float(term.get("offset", 0.0)) for term in terms)
        oscillating = _eval_channel(terms, t) - offset
        return offset + env * oscillating

    def evaluate(
        self, t: float
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None]:
        env = _envelope(self._envelope, t, self._duration)
        head = create_head_pose(
            x=self._channel_value("head_x", t, env),
            y=self._channel_value("head_y", t, env),
            z=self._channel_value("head_z", t, env),
            roll=self._channel_value("head_roll", t, env),
            pitch=self._channel_value("head_pitch", t, env),
            yaw=self._channel_value("head_yaw", t, env),
            mm=True,
            degrees=True,
        )
        antennas = np.deg2rad(
            [
                self._channel_value("antenna_left", t, env),
                self._channel_value("antenna_right", t, env),
            ]
        )
        body_yaw = math.radians(self._channel_value("body_yaw", t, env))
        return head, antennas, body_yaw

    def bake(self, fps: float = 100.0) -> MoveData:
        """Sample the move into the canonical keyframe schema."""
        n = max(2, int(round(self._duration * fps)) + 1)
        times = np.linspace(0.0, self._duration, n)
        heads = np.empty((n, 4, 4), dtype=np.float64)
        antennas = np.empty((n, 2), dtype=np.float64)
        body = np.empty((n,), dtype=np.float64)
        for i, t in enumerate(times):
            h, a, b = self.evaluate(float(t))
            heads[i] = h
            antennas[i] = a
            body[i] = b if b is not None else 0.0
        return MoveData(
            description=self.description,
            time=times,
            head=heads,
            antennas=antennas,
            body_yaw=body,
        )
