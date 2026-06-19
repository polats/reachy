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
<div id="viewer-block" style="display:flex;flex-direction:column;gap:6px">
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
<div id="chart-block" style="display:flex;flex-direction:column;gap:4px">
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

# Phase-1 authoring: pose the robot in 3D + a keyframe timeline (driven by viewer.js, client-side).
# A "keyframe" is a segment's target pose; the animation eases NEUTRAL -> kf1 -> kf2 -> ... .
_POSE_SLIDER = ('<label>{label}</label>'
                '<input class="apose" data-ch="{ch}" type="range" min="{lo}" max="{hi}" step="{step}" value="0">'
                '<span class="aval" data-for="{ch}">0</span>')
# yaw + pitch live on a 2D "aim pad" (like a thumbstick); the rest stay as sliders
_POSE_SLIDERS = "".join(_POSE_SLIDER.format(label=l, ch=c, lo=lo, hi=hi, step=st) for l, c, lo, hi, st in [
    ("Tilt (roll)", "roll", -30, 30, 1), ("Height", "z", -15, 15, 0.5),
    ("Body", "body", -90, 90, 1), ("L ear", "antL", -180, 180, 1), ("R ear", "antR", -180, 180, 1),
])
_EASE_OPTS = "".join(f'<option value="{e}">{e}</option>'
                     for e in ("smooth", "linear", "ease_out", "ease_in", "back", "anticipate", "hold"))
# the aim pad: a draggable dot = (yaw, pitch), gamepad-thumbstick style. data-x/data-y carry the
# channel + range so viewer.js maps the dot position <-> pose without hardcoding.
_AIM_PAD = """
<div style="display:flex;flex-direction:column;gap:3px;align-items:center">
  <span style="font-size:11px;opacity:0.7">Aim (drag)</span>
  <div id="aim-pad" data-x="yaw" data-xlo="-90" data-xhi="90" data-y="pitch" data-ylo="-25" data-yhi="20"
       style="position:relative;width:128px;height:128px;border-radius:12px;background:rgba(255,255,255,0.05);
              border:1px solid rgba(255,255,255,0.14);touch-action:none;cursor:grab;flex:none">
    <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:rgba(255,255,255,0.10)"></div>
    <div style="position:absolute;top:50%;left:0;height:1px;width:100%;background:rgba(255,255,255,0.10)"></div>
    <div id="aim-dot" style="position:absolute;left:50%;top:50%;width:18px;height:18px;border-radius:50%;
         background:#3b82f6;border:2px solid #fff;transform:translate(-50%,-50%);box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>
  </div>
  <span style="font-size:10px;opacity:0.55">← turn →&nbsp;·&nbsp;↑ nod ↓</span>
</div>
"""
POSE_PANEL_HTML = f"""
<div id="author-pose" style="display:flex;flex-direction:column;gap:8px">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
    <span style="font-size:13px;font-weight:600;opacity:0.85">Pose the keyframe</span>
    <button id="author-posemode" type="button" style="padding:3px 10px;border-radius:6px;cursor:pointer;font-size:12px">✋ Pose</button>
  </div>
  <div id="author-pose-hint" style="font-size:12px;opacity:0.6">
    Select or add a keyframe, then drag in 3D — <b>head</b> = aim, <b>antenna</b> = raise/lower,
    <b>body</b> = twist (the grabbed part glows). Or use the pad + sliders. Toggle 🔄 Orbit to look around.</div>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    {_AIM_PAD}
    <div style="flex:1;min-width:190px;display:grid;grid-template-columns:auto 1fr 2.4em;gap:7px 10px;align-items:center;font-size:12px">{_POSE_SLIDERS}</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px">
    <label>Dur</label><input id="author-dur" type="number" min="0.05" max="6" step="0.05" value="0.4" style="width:64px">s
    <label>Ease</label><select id="author-ease" style="padding:2px">{_EASE_OPTS}</select>
    <button id="author-addkf" type="button" style="padding:5px 12px;border-radius:6px;cursor:pointer;background:#2563eb;color:#fff;border:none;font-weight:600">＋ Keyframe</button>
  </div>
</div>
"""

SOUND_LAB_HTML = """
<div id="soundlab" style="display:flex;flex-direction:column;gap:10px">
  <div style="display:flex;align-items:center;gap:8px">
    <button id="sl-play" type="button" style="padding:5px 16px;border-radius:6px;cursor:pointer;background:#2563eb;color:#fff;border:none;font-weight:600">▶ Play</button>
    <button id="sl-loop" type="button" style="padding:5px 12px;border-radius:6px;cursor:pointer;background:rgba(255,255,255,0.08);color:#fff;border:none">⟳ Loop</button>
    <button id="sl-stop" type="button" style="padding:5px 12px;border-radius:6px;cursor:pointer;background:rgba(255,255,255,0.08);color:#fff;border:none">⏹</button>
    <span style="font-size:12px;opacity:0.55">instant browser preview · 📢 plays on the robot below</span>
  </div>
  <div id="sl-wave" style="width:100%;height:70px;background:rgba(255,255,255,0.03);border-radius:6px"></div>
  <div style="font-size:12px;opacity:0.7">Pitch contour (the melody) — <b>drag</b> a dot up/down · <b>double-click</b> to add · <b>right-click</b> to remove</div>
  <div id="sl-contour" style="width:100%;height:150px;background:rgba(255,255,255,0.03);border-radius:8px"></div>
  <div style="display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap">
    <div style="display:flex;flex-direction:column;gap:3px">
      <span style="font-size:11px;opacity:0.7">Voice (drag knobs ↕)</span>
      <div id="sl-knobs" style="display:flex;gap:12px"></div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:3px">
      <div id="sl-formants" style="width:130px;height:130px"></div>
      <span style="font-size:11px;opacity:0.7">Formants (timbre)</span>
    </div>
  </div>
</div>
"""

TIMELINE_HTML = """
<div style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
  <div style="font-size:13px;font-weight:600;opacity:0.85">Timeline — keyframes (tap to edit · ✕ to remove)</div>
  <div id="author-timeline" style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;min-height:42px;
       padding:6px;border-radius:8px;background:rgba(255,255,255,0.03)"></div>
</div>
"""
