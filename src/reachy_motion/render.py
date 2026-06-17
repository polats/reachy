"""Offscreen render a Reachy Mini move to a preview video (MP4/GIF).

Uses MuJoCo's EGL offscreen renderer (no window, no GLFW) — reliable on headless and
Wayland machines where the interactive viewer is flaky. This is the workhorse for
previewing generated moves and building a clip gallery for the generation pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

import reachy_mini
from reachy_mini.motion.move import Move

from .schema import MoveData

# EGL must be selected before mujoco creates a GL context.
os.environ.setdefault("MUJOCO_GL", "egl")


def _scene_path(scene: str) -> str:
    base = Path(reachy_mini.__file__).parent / "descriptions" / "reachy_mini" / "mjcf"
    cand = base / f"{scene}.xml"
    return str(cand if cand.exists() else base / "scene.xml")


def render(
    move: Union[Move, MoveData],
    out_path: str | Path,
    *,
    scene: str = "scene",
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    cam_distance: float = 0.8,
    cam_azimuth: float = 160.0,
    cam_elevation: float = -20.0,
    cam_lookat: tuple[float, float, float] = (0.0, 0.0, 0.15),
    settle_steps: int = 200,
) -> Path:
    """Render ``move`` to a video file (``.mp4`` or ``.gif``). Returns the path."""
    import imageio.v2 as imageio
    import mujoco
    from reachy_mini.kinematics import AnalyticalKinematics

    from .viewer import _apply_targets

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(_scene_path(scene))
    d = mujoco.MjData(m)
    kin = AnalyticalKinematics()

    rec = move.to_recorded_move() if isinstance(move, MoveData) else move
    duration = float(rec.duration)
    t_end = max(0.0, duration - 1e-4)
    sim_dt = float(m.opt.timestep)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation
    cam.lookat[:] = cam_lookat

    renderer = mujoco.Renderer(m, height=height, width=width)
    try:
        # settle on the t=0 pose
        head0, ant0, by0 = rec.evaluate(0.0)
        _apply_targets(d, kin, head0, ant0, by0)
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)

        n_frames = max(2, int(round(duration * fps)) + 1)
        frames = []
        for i in range(n_frames):
            t = t_end * i / (n_frames - 1)
            head, antennas, body_yaw = rec.evaluate(t)
            _apply_targets(d, kin, head, antennas, body_yaw)
            # advance sim to this frame's wall-clock time
            target_sim_t = t
            while d.time < target_sim_t:
                mujoco.mj_step(m, d)
            renderer.update_scene(d, cam)
            frames.append(renderer.render())

        if out_path.suffix.lower() == ".gif":
            imageio.mimsave(out_path, frames, fps=fps, loop=0)
        else:
            imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    finally:
        renderer.close()

    return out_path
