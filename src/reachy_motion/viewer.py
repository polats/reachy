"""Standalone MuJoCo viewer for Reachy Mini moves — decoupled from the daemon.

Why this exists: the official ``reachy-mini-daemon --sim`` opens its viewer *inside*
the daemon process, which also runs a GStreamer/GLib main loop for media. On some
Linux/Wayland setups those two windowing stacks corrupt the heap and the daemon
crashes (SIGSEGV/SIGABRT). A *clean* MuJoCo process (no GStreamer) renders fine, so
this viewer runs its own physics + window and drives the model directly through the
SDK's analytical kinematics. No daemon, no media stack, no crash.

It is purely a visualizer: it does not talk to a daemon and cannot drive real hardware.
For hardware (or the official sim), use :mod:`reachy_motion.player`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

import numpy as np

import reachy_mini
from reachy_mini.motion.move import Move

from .schema import MoveData


def model_path(scene: str = "scene") -> str:
    """Absolute path to a shipped Reachy Mini MJCF scene (e.g. ``scene``, ``empty``)."""
    base = Path(reachy_mini.__file__).parent / "descriptions" / "reachy_mini" / "mjcf"
    cand = base / f"{scene}.xml"
    if not cand.exists():
        cand = base / "scene.xml"
    return str(cand)


def _apply_targets(data, kin, head, antennas, body_yaw) -> None:  # type: ignore[no-untyped-def]
    """Write a move sample into MuJoCo actuator targets (mirrors the daemon backend)."""
    if head is not None:
        q = kin.ik(head, float(body_yaw or 0.0))  # 4x4 (+body yaw) -> 7 head joints
        data.ctrl[:7] = q
    if antennas is not None:
        data.ctrl[-2:] = -np.asarray(antennas, dtype=np.float64)  # note: negated


def view(
    move: Union[Move, MoveData],
    *,
    scene: str = "scene",
    loop: bool = False,
    realtime: bool = True,
    settle_steps: int = 200,
) -> None:
    """Open a MuJoCo window and play a move on the simulated Reachy Mini.

    Blocks until the move finishes (``loop=False``) or the window is closed.
    """
    import mujoco
    import mujoco.viewer
    from reachy_mini.kinematics import AnalyticalKinematics

    m = mujoco.MjModel.from_xml_path(model_path(scene))
    d = mujoco.MjData(m)
    kin = AnalyticalKinematics()

    rec = move.to_recorded_move() if isinstance(move, MoveData) else move
    duration = float(rec.duration)
    dt = float(m.opt.timestep)

    with mujoco.viewer.launch_passive(
        m, d, show_left_ui=False, show_right_ui=False
    ) as v:
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.distance = 0.8
            v.cam.azimuth = 160
            v.cam.elevation = -20
            v.cam.lookat[:] = [0, 0, 0.15]

        # RecordedMove.evaluate rejects t >= duration; clamp just under the end.
        t_end = max(0.0, duration - 1e-4)

        # settle to the move's t=0 pose so playback doesn't snap
        head0, ant0, by0 = rec.evaluate(0.0)
        _apply_targets(d, kin, head0, ant0, by0)
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)
        v.sync()

        sim_t = 0.0
        wall0 = time.time()
        while v.is_running():
            head, antennas, body_yaw = rec.evaluate(min(sim_t, t_end))
            _apply_targets(d, kin, head, antennas, body_yaw)
            mujoco.mj_step(m, d)
            v.sync()
            sim_t += dt
            if realtime:
                lag = (wall0 + sim_t) - time.time()
                if lag > 0:
                    time.sleep(lag)
            if sim_t >= duration:
                if loop:
                    sim_t = 0.0
                    wall0 = time.time()
                else:
                    break


def _dry_run(move: Union[Move, MoveData], steps: int = 50) -> dict:
    """Headless validation of the IK/ctrl path (no window). Returns diagnostics."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from reachy_mini.kinematics import AnalyticalKinematics

    m = mujoco.MjModel.from_xml_path(model_path())
    d = mujoco.MjData(m)
    kin = AnalyticalKinematics()
    rec = move.to_recorded_move() if isinstance(move, MoveData) else move
    duration = float(rec.duration)
    t_end = max(0.0, duration - 1e-4)
    ctrl_range = []
    for i in range(steps):
        t = t_end * i / max(1, steps - 1)
        head, antennas, body_yaw = rec.evaluate(t)
        _apply_targets(d, kin, head, antennas, body_yaw)
        mujoco.mj_step(m, d)
        ctrl_range.append(d.ctrl.copy())
    arr = np.array(ctrl_range)
    return {
        "nu": int(m.nu),
        "nq": int(m.nq),
        "ctrl_min": arr.min(axis=0).round(4).tolist(),
        "ctrl_max": arr.max(axis=0).round(4).tolist(),
        "moved": bool(np.ptp(arr, axis=0).max() > 1e-4),
    }
