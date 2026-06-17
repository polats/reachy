"""Load a Reachy Mini moves library and build cached audio+video previews.

A "preview" is the move rendered offscreen (EGL, via :mod:`reachy_motion.render`) and
muxed with its paired WAV into a single MP4 — i.e. exactly the synchronized motion+sound
behavior, watchable in a browser. Used by the Gradio dataset viewer (``app.py``).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves

from .render import render

DEFAULT_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "out" / "previews"


@lru_cache(maxsize=8)
def get_library(name: str = DEFAULT_LIBRARY) -> RecordedMoves:
    """Load (and cache) a moves library; downloads from HF on first use."""
    return RecordedMoves(name)


def list_moves(library: str = DEFAULT_LIBRARY) -> list[str]:
    return sorted(get_library(library).list_moves())


@dataclass
class MoveInfo:
    name: str
    description: str
    duration: float
    num_frames: int
    has_sound: bool


def move_info(name: str, library: str = DEFAULT_LIBRARY) -> MoveInfo:
    m = get_library(library).get(name)
    return MoveInfo(
        name=name,
        description=getattr(m, "description", "") or "",
        duration=float(m.duration),
        num_frames=len(getattr(m, "timestamps", []) or []),
        has_sound=m.sound_path is not None,
    )


def channels(name: str, library: str = DEFAULT_LIBRARY) -> dict[str, np.ndarray]:
    """Per-frame channel signals for plotting: time + head euler/pos + antennas + body."""
    from scipy.spatial.transform import Rotation as R

    m = get_library(library).get(name)
    t = np.asarray(m.timestamps, dtype=np.float64)
    traj = m.trajectory
    heads = np.array([f["head"] for f in traj], dtype=np.float64)  # [T,4,4]
    eul = R.from_matrix(heads[:, :3, :3]).as_euler("xyz", degrees=True)  # roll,pitch,yaw
    pos_mm = heads[:, :3, 3] * 1000.0
    ant = np.rad2deg(np.array([f["antennas"] for f in traj], dtype=np.float64))
    body = np.rad2deg(np.array([f.get("body_yaw", 0.0) for f in traj], dtype=np.float64))
    return {
        "t": t,
        "head_roll": eul[:, 0], "head_pitch": eul[:, 1], "head_yaw": eul[:, 2],
        "head_x": pos_mm[:, 0], "head_y": pos_mm[:, 1], "head_z": pos_mm[:, 2],
        "antenna_left": ant[:, 0], "antenna_right": ant[:, 1], "body_yaw": body,
    }


def _mux(silent_mp4: Path, wav: Path, out: Path) -> Path:
    """Combine a silent video with a WAV into one MP4 (re-encodes audio to AAC)."""
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(silent_mp4), "-i", str(wav),
         "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def preview(
    name: str,
    library: str = DEFAULT_LIBRARY,
    *,
    fps: int = 30,
    width: int = 512,
    height: int = 384,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Return a cached MP4 of the move with synchronized sound, rendering if needed."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / f"{name}.mp4"
    if final.exists() and not force:
        return final

    move = get_library(library).get(name)
    silent = render(move, cache_dir / f"{name}_silent.mp4", fps=fps, width=width, height=height)
    if move.sound_path is not None and Path(move.sound_path).exists():
        out = _mux(silent, Path(move.sound_path), final)
        silent.unlink(missing_ok=True)
        return out
    return silent.rename(final)
