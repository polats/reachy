"""Three.js viewer that loads the 8bitkick Reachy Mini model (urdf-loader + DRACO).

Uses the proven asset bundle from the `8bitkick/reachy_mini_3d_web_viz` Space (vendored
under assets/reachy_web): the URDF + per-part DRACO-compressed GLBs. We load it exactly
like they do (urdf-loader → GLTFLoader+DRACOLoader, per-link MeshPhysicalMaterial from
URDF colors), which renders cleanly (no splotches — the parts stay separate with their
own normals).

Animation is IK-free (we only have the head 4x4, not the 7 Stewart joint values): we
hide the leg linkage, reparent the head platform (xl_330) under a world-aligned control
frame, and per frame set body_yaw + antenna joints + the head control frame's 4x4.
"""

from __future__ import annotations

import numpy as np

from .dataset import get_library

_KIN = None


def _kin():
    """Cached analytic kinematics (head 4x4 + body_yaw -> 7 head joints)."""
    global _KIN
    if _KIN is None:
        from reachy_mini.kinematics import AnalyticalKinematics

        _KIN = AnalyticalKinematics()
    return _KIN


def _pose_head_matrix(pose: dict):
    from reachy_mini.utils import create_head_pose

    return create_head_pose(
        x=pose["x"], y=pose["y"], z=pose["z"],
        roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"],
        mm=False, degrees=False,
    )


# the canonical "ready" pose (SDK: INIT_HEAD_POSE = identity, INIT_ANTENNAS = ±10°, body 0)
READY_POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
              "antL": -0.1745, "antR": 0.1745, "body": 0.0}


def ready_render() -> dict:
    """The ready pose as an updateJoints payload for the viewer (single source of truth)."""
    return pose_to_render(READY_POSE)


def pose_to_render(pose: dict) -> dict:
    """Pose (command vector) -> payload the 3D viewer's updateJoints consumes."""
    H = _pose_head_matrix(pose)
    try:
        q = _kin().ik(H, float(pose.get("body", 0.0)))
        q = [float(x) for x in q] if np.all(np.isfinite(q)) else [0.0] * 7
    except Exception:
        q = [0.0] * 7
    return {
        "head_pose": [float(x) for x in H.flatten()],
        "head_joints": q,
        "antennas_position": [float(pose["antL"]), float(pose["antR"])],
        "body_yaw": float(pose.get("body", 0.0)),
    }


def pose_to_goto(pose: dict):
    """Pose -> (head 4x4, antennas array, body_yaw) for ReachyMini.goto_target."""
    return (
        _pose_head_matrix(pose),
        np.array([float(pose["antL"]), float(pose["antR"])]),
        float(pose.get("body", 0.0)),
    )


def move_trajectory(name: str, library: str, fps: int = 40) -> dict:
    """Downsample a move into a compact JSON trajectory for the browser viewer.

    Includes ``head_joints`` (the 7 Stewart/body joint values via IK, so the browser can
    drive the full platform like the 8bitkick viewer) and ``head`` as a 16-flat row-major
    matrix (the format their ``updateJoints`` consumes). IK is non-finite for ~5% of
    frames on some moves; those carry forward the last valid solution.
    """
    m = get_library(library).get(name)
    t = np.asarray(m.timestamps, dtype=np.float64)
    dur = float(t[-1])
    n = max(2, int(round(dur * fps)) + 1)
    grid = np.linspace(0.0, dur, n)
    idx = np.searchsorted(t, grid).clip(0, len(t) - 1)
    traj = m.trajectory
    kin = _kin()

    from scipy.spatial.transform import Rotation as R

    head_flat, head_joints, antennas, body = [], [], [], []
    ch = {k: [] for k in ("roll", "pitch", "yaw", "x", "y", "z", "antL", "antR", "body")}
    last_q = [0.0] * 7
    for i in idx:
        H = np.asarray(traj[i]["head"], dtype=np.float64)
        by = float(traj[i].get("body_yaw", 0.0))
        try:
            q = kin.ik(H, by)
            q = [float(x) for x in q] if np.all(np.isfinite(q)) else last_q
        except Exception:
            q = last_q
        last_q = q
        head_joints.append(q)
        head_flat.append([float(x) for x in H.flatten()])
        aL, aR = float(traj[i]["antennas"][0]), float(traj[i]["antennas"][1])
        antennas.append([aL, aR])
        body.append(by)
        roll, pitch, yaw = R.from_matrix(H[:3, :3]).as_euler("xyz", degrees=True)
        ch["roll"].append(float(roll)); ch["pitch"].append(float(pitch)); ch["yaw"].append(float(yaw))
        ch["x"].append(float(H[0, 3] * 1000)); ch["y"].append(float(H[1, 3] * 1000)); ch["z"].append(float(H[2, 3] * 1000))
        ch["antL"].append(float(np.rad2deg(aL))); ch["antR"].append(float(np.rad2deg(aR)))
        ch["body"].append(float(np.rad2deg(by)))
    return {
        "time": [float(x) for x in grid],
        "head": head_flat,
        "head_joints": head_joints,
        "antennas": antennas,
        "body_yaw": body,
        "channels": ch,
    }


HEAD_HTML = """
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
}}
</script>
"""

CONTAINER_HTML = """
<div style="display:flex;flex-direction:column;gap:6px">
  <div id="reachy3d" style="width:100%;height:420px;border-radius:10px;overflow:hidden;background:#12182a"></div>
  <div style="display:flex;align-items:center;gap:10px">
    <button id="reachy-play" style="padding:4px 12px;border-radius:6px;cursor:pointer">⏸ Pause</button>
    <input id="reachy-time" type="range" min="0" max="1000" value="0" style="flex:1"/>
  </div>
  <div style="font-size:12px;opacity:0.7">drag to orbit · scroll to zoom · scrub the timeline · Connect (top-left) mirrors the live USB robot</div>
  <audio id="reachy-audio" loop preload="auto" style="display:none"></audio>
</div>
"""

AUDIO_HTML = """
<div id="reachy-voice" style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
  <div style="display:flex;align-items:center;gap:8px">
    <span id="reachy-voice-dot" style="width:10px;height:10px;border-radius:50%;background:#444;display:inline-block"></span>
    <span style="font-size:13px;font-weight:600;opacity:0.85">🎤 Voice</span>
    <span id="reachy-voice-label" style="font-size:12px;opacity:0.6">connect to hear the robot's mic</span>
  </div>
  <canvas id="reachy-audio-mon" width="600" height="44"
          style="width:100%;height:44px;background:rgba(255,255,255,0.03);border-radius:6px"></canvas>
  <div id="reachy-transcript" style="font-size:13px;line-height:1.45;min-height:2.8em;margin-top:2px"></div>
</div>
"""

CAMERA_HTML = """
<div style="display:flex;flex-direction:column;gap:4px">
  <video id="reachy-cam" autoplay playsinline muted
         style="width:100%;border-radius:6px;background:#000;aspect-ratio:16/9;object-fit:contain"></video>
  <div id="reachy-cam-status" style="font-size:12px;opacity:0.6">connect to the robot, then expand to start the camera</div>
</div>
"""

CHART_HTML = """
<div style="display:flex;flex-direction:column;gap:4px">
  <div style="font-size:13px;font-weight:600;opacity:0.85">Channels (head pose · antennas · body)</div>
  <div id="reachy-chart" style="width:100%;height:330px"></div>
  <div style="font-size:12px;opacity:0.6">white line = playhead · click or drag on the chart to scrub</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">
    <div style="font-size:13px;font-weight:600;opacity:0.85">Voice</div>
    <div style="display:flex;align-items:center;gap:6px;font-size:12px;opacity:0.75">
      🔊 <input id="reachy-vol" type="range" min="0" max="100" value="80" style="width:96px" title="volume">
    </div>
  </div>
  <div id="reachy-wave" style="width:100%;height:70px">
    <div style="opacity:.45;font-size:12px;padding:24px 0">select a move to see its sound</div>
  </div>
  <div style="font-size:12px;opacity:0.6">waveform of the move's sound · click or drag to scrub the voice</div>
</div>
"""

GAMEPAD_HTML = """<div id="gamepad-viz" style="font-size:12px;min-height:24px">
  <div style="opacity:.6">No gamepad detected — press a button on it.</div>
</div>"""
