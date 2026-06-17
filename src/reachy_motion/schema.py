"""Canonical Reachy Mini move schema — the lingua franca of this project.

A "move" is the exact JSON format produced by the Marionette recorder, stored in
the Hugging Face moves datasets (e.g. ``pollen-robotics/reachy-mini-emotions-library``),
and consumed by :class:`reachy_mini.motion.recorded_move.RecordedMove`::

    {
      "description": str,
      "time": [t0, t1, ...],                       # seconds, monotonically increasing
      "set_target_data": [
        {"head": <4x4 row-major list>, "antennas": [left, right], "body_yaw": float},
        ...                                         # one entry per timestamp
      ]
    }

Head is a 4x4 homogeneous transform (head frame). ``antennas`` and ``body_yaw`` are
radians. A move may be paired by filename with a ``<name>.wav`` for synchronized sound.

Everything in this project (symbolic generator now, learned model later) emits *this*
schema, so generated moves are first-class: playable via ``ReachyMini.play_move`` and
selectable by the conversation app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# Keys of the per-frame target dict, matching reachy_mini.motion.recorded_move.
HEAD_KEY = "head"
ANTENNAS_KEY = "antennas"
BODY_YAW_KEY = "body_yaw"


@dataclass
class MoveData:
    """An in-memory move in the canonical schema.

    Stores the head poses as a single ``[T, 4, 4]`` array for convenience; converts
    to/from the on-disk JSON (list-of-dicts) form on demand.
    """

    description: str
    time: npt.NDArray[np.float64]  # [T]
    head: npt.NDArray[np.float64]  # [T, 4, 4]
    antennas: npt.NDArray[np.float64]  # [T, 2] radians
    body_yaw: npt.NDArray[np.float64]  # [T] radians
    sound_path: Optional[Path] = field(default=None)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.head = np.asarray(self.head, dtype=np.float64)
        self.antennas = np.asarray(self.antennas, dtype=np.float64)
        self.body_yaw = np.asarray(self.body_yaw, dtype=np.float64)
        t = len(self.time)
        assert self.head.shape == (t, 4, 4), f"head must be [T,4,4], got {self.head.shape}"
        assert self.antennas.shape == (t, 2), f"antennas must be [T,2], got {self.antennas.shape}"
        assert self.body_yaw.shape == (t,), f"body_yaw must be [T], got {self.body_yaw.shape}"

    @property
    def duration(self) -> float:
        return float(self.time[-1]) if len(self.time) else 0.0

    @property
    def num_frames(self) -> int:
        return len(self.time)

    # --- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the canonical move dict (what RecordedMove consumes)."""
        return {
            "description": self.description,
            "time": [float(t) for t in self.time],
            "set_target_data": [
                {
                    HEAD_KEY: self.head[i].tolist(),
                    ANTENNAS_KEY: [float(self.antennas[i, 0]), float(self.antennas[i, 1])],
                    BODY_YAW_KEY: float(self.body_yaw[i]),
                }
                for i in range(self.num_frames)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], sound_path: Optional[Path] = None) -> "MoveData":
        traj = d["set_target_data"]
        return cls(
            description=d.get("description", ""),
            time=np.asarray(d["time"], dtype=np.float64),
            head=np.asarray([f[HEAD_KEY] for f in traj], dtype=np.float64),
            antennas=np.asarray([f[ANTENNAS_KEY] for f in traj], dtype=np.float64),
            body_yaw=np.asarray([f.get(BODY_YAW_KEY, 0.0) for f in traj], dtype=np.float64),
            sound_path=sound_path,
        )

    def save(self, path: str | Path) -> Path:
        """Write ``<stem>.json`` (the canonical schema). Returns the JSON path."""
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoveData":
        path = Path(path)
        d = json.loads(path.read_text())
        wav = path.with_suffix(".wav")
        return cls.from_dict(d, sound_path=wav if wav.exists() else None)

    def to_recorded_move(self):  # type: ignore[no-untyped-def]
        """Adapt to a ``reachy_mini`` ``RecordedMove`` so it can be played directly."""
        from reachy_mini.motion.recorded_move import RecordedMove

        return RecordedMove(self.to_dict(), sound_path=self.sound_path)
