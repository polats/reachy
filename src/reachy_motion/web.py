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
<script type="module">
import * as THREE from 'three';
import URDFLoader from 'https://cdn.jsdelivr.net/npm/urdf-loader@0.12.3/+esm';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// joint names (match the 8bitkick RobotManager)
const HEAD_JOINT_NAMES = ['yaw_body','stewart_1','stewart_2','stewart_3','stewart_4','stewart_5','stewart_6'];
const PASSIVE_JOINT_NAMES = [];
for (let i=1;i<=7;i++) PASSIVE_JOINT_NAMES.push(`passive_${i}_x`,`passive_${i}_y`,`passive_${i}_z`);

const V = { ready:false, traj:null, t:0, dur:0, playing:true, robot:null, jointMap:{},
            calcPassive:null, buildHeadPose:null, ik:null, audio:null, live:false, ws:null,
            audioCtx:null, waveUrl:null, waveW:0, cam:null,
            liveBuf:[], liveWindow:8, liveLast:0,
            joy:{ on:false, ws:null, timer:0, warned:false, tgt:null, phase:null, fps:null,
                  prevL3:false, prevR3:false } };

// gamepad teleop tuning (ranges/speeds mirror the desktop app)
const JOY = {
  dt:1/30,            // fixed tick (steady setInterval, not rAF → no dt spikes / render-stall jerks)
  dead:0.12,
  smooth:0.35,        // input low-pass (EMA) factor — higher = snappier, lower = smoother
  posSpeed:0.06, rotSpeed:0.9, zSpeed:0.04, bodySpeed:0.9, antSpeed:2.5,
  lim:{ pos:0.05, z:0.05, pitch:0.8, yaw:1.2, roll:0.5, body:1.0, ant:2.79 } };
const _clamp = (v,a) => Math.max(-a, Math.min(a, v));
const _dz = v => Math.abs(v) < JOY.dead ? 0 : v;

// ---- FPS teleop model, ported from ebubar/teleop (HF: "FPS-style gamepad control with
// dynamic safety limits"). State is in DEGREES + mm; head yaw is RELATIVE to the body. ----
const FPS = {
  hz:60, dead:0.15, axisLock:2.0, smooth:0.7,        // per-60Hz-tick constants (verbatim)
  speed:{ head:3.5, ant:5.0, z:1.0, base:2.5 },      // deg/tick (z in mm/tick)
  // verified all-reachable envelope (81/81 corners incl. body-yaw): no detach anywhere inside it
  lim:{ neck:30, pitch:15, zMin:-15, zMax:15, ant:60, base:90 } };
const _fdz = v => Math.abs(v) < FPS.dead ? 0 : v;
function fpsSnap(x, y){                                // axis-lock: kill accidental diagonals
  if (Math.abs(x) < FPS.dead && Math.abs(y) < FPS.dead) return [x, y];
  if (Math.abs(x) > Math.abs(y)*FPS.axisLock) y = 0;
  else if (Math.abs(y) > Math.abs(x)*FPS.axisLock) x = 0;
  return [x, y];
}
function fpsClampDyn(j){    // clamp to the verified-reachable fixed envelope (no FK / no detach inside)
  const L = FPS.lim;
  j.z     = Math.max(L.zMin, Math.min(j.z, L.zMax));
  j.neck  = Math.max(-L.neck,  Math.min(j.neck,  L.neck));
  j.pitch = Math.max(-L.pitch, Math.min(j.pitch, L.pitch));
  j.base  = Math.max(-L.base,  Math.min(j.base,  L.base));
  j.antL  = Math.max(-L.ant,   Math.min(j.antL,  L.ant));
  j.antR  = Math.max(-L.ant,   Math.min(j.antR,  L.ant));
}
function fpsReadyTargets(j){ j.neck=0; j.base=0; j.pitch=0; j.z=0; j.antL=-10; j.antR=10; }  // ready (±10° ant)
function fpsReady(j){ fpsReadyTargets(j); j.cNeck=0; j.cBase=0; j.cPitch=0; j.cZ=0; j.cAntL=-10; j.cAntR=10; }
function fpsFromPose(j, p){                            // SI command-vector pose -> degree/mm targets
  const G = 180/Math.PI;
  j.base = p.body*G; j.neck = p.yaw*G - j.base; j.pitch = p.pitch*G;
  j.z = p.z*1000; j.antL = p.antL*G; j.antR = p.antR*G;
}
function fpsToTgt(j, tg){                              // smoothed state -> our SI command vector
  const D = Math.PI/180;
  tg.x = 0; tg.y = 0; tg.roll = 0;
  tg.pitch = j.cPitch*D;
  tg.yaw = (j.cBase + j.cNeck)*D;                      // head pose yaw = body + neck (relative)
  tg.z = j.cZ/1000;                                    // mm -> m
  tg.by = j.cBase*D; tg.aL = j.cAntL*D; tg.aR = j.cAntR*D;
}
function fpsSmoothAndPush(j, tg){
  fpsClampDyn(j);
  const s = FPS.smooth;
  j.cNeck = j.neck*s + j.cNeck*(1-s);  j.cPitch = j.pitch*s + j.cPitch*(1-s);
  j.cZ    = j.z*s    + j.cZ*(1-s);     j.cBase  = j.base*s  + j.cBase*(1-s);
  j.cAntL = j.antL*s + j.cAntL*(1-s);  j.cAntR  = j.antR*s  + j.cAntR*(1-s);
  fpsToTgt(j, tg);
  sendTarget(tg);                                      // robot stream (sim display via rAF driveSim)
}

// fallback rig config if window.REACHY_READY is missing (head joints + identity head pose)
const SIM_INIT16 = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const SIM_INIT_JOINTS = [0, 0.62655, -0.62654, 0.62654, -0.62654, 0.62654, -0.62654];
function readyPayload(){
  return window.REACHY_READY ||
    { head_pose: SIM_INIT16, head_joints: SIM_INIT_JOINTS, antennas_position:[-0.1745,0.1745], body_yaw:0 };
}
const JOY_MSG = '🎮 Joystick — L stick look · R stick turn + height · L2/R2 antennas · L3 ready';

// head pose -> row-major 4x4 (matches the SDK's create_head_pose: intrinsic xyz euler + translation)
function poseMatrix(tg){
  const cr=Math.cos(tg.roll), sr=Math.sin(tg.roll);
  const cp=Math.cos(tg.pitch), sp=Math.sin(tg.pitch);
  const cy=Math.cos(tg.yaw), sy=Math.sin(tg.yaw);
  return [
    cp*cy,           -cp*sy,          sp,             tg.x,
    cr*sy+sr*sp*cy,  cr*cy-sr*sp*sy, -sr*cp,          tg.y,
    sr*sy-cr*sp*cy,  sr*cy+cr*sp*sy,  cr*cp,          tg.z,
    0, 0, 0, 1,
  ];
}

// static ready pose (shown when nothing is playing / dropdown empty / entering Simulator)
function showReady(){
  if (!V.robot || !V.ready) return;
  updateJoints(readyPayload());
  V.robot.updateMatrixWorld(true);
}

// start active joystick free-control from the ready pose
function startControl(){
  if (!V.joy.on || V.joy.phase === 'control') return;
  V.joy.fps = {}; fpsReady(V.joy.fps);              // start at the ready pose
  V.joy.lastGood = null;
  V.joy.tgt = { x:0,y:0,z:0,roll:0,pitch:0,yaw:0,aL:0,aR:0,by:0 };
  fpsToTgt(V.joy.fps, V.joy.tgt);
  V.joy.prevL3 = false; V.joy.prevR3 = false;
  V.joy.phase = 'control';
  if (!V.joy.timer) V.joy.timer = setInterval(joyTick, 1000/FPS.hz);   // ~60 Hz like the source
  if (!V.live) driveSim(V.joy.tgt);
}

// auto-enable joystick control when a gamepad is present AND the Control sub-tab is open
// (no checkbox). Works in both modes — setJoystick branches on V.live internally.
function autoControl(){
  if (V.ready){
    const viz = document.getElementById('gamepad-viz');
    const controlTab = !!(viz && viz.offsetParent);     // Control sub-tab visible
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    let gp = false; for (const p of pads){ if (p){ gp = true; break; } }
    if (controlTab && gp && !V.joy.on) window.ReachyViewer.setJoystick(true);
    else if ((!controlTab || !gp) && V.joy.on) window.ReachyViewer.setJoystick(false);
    else if (!V.live && controlTab && !gp && !V.joy.on) showReady();   // Control tab, waiting for a pad
  }
  setTimeout(autoControl, 250);
}

// live gamepad tester (axes bars + button grid), like html5-gamepad-test
function gamepadViz(){
  const el = document.getElementById('gamepad-viz');
  if (el && el.offsetParent !== null){            // only render when the panel is open/visible
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    let gp = null; for (const p of pads){ if (p){ gp = p; break; } }
    if (!gp){
      el.innerHTML = '<div style="opacity:.6;padding:6px 0">No gamepad detected — press a button on it.</div>';
    } else {
      let h = `<div style="font-size:11px;opacity:.7;margin-bottom:5px">${gp.id.slice(0,46)}</div>`;
      h += '<div style="display:flex;flex-direction:column;gap:3px">';
      gp.axes.forEach((v, i) => {
        const pct = ((v + 1) / 2 * 100).toFixed(1);
        h += `<div style="display:flex;align-items:center;gap:6px;font-size:11px">
                <span style="width:30px;opacity:.7">ax${i}</span>
                <div style="flex:1;height:8px;background:rgba(255,255,255,.08);border-radius:4px;position:relative">
                  <div style="position:absolute;left:50%;top:0;width:1px;height:8px;background:rgba(255,255,255,.2)"></div>
                  <div style="position:absolute;left:${pct}%;top:-2px;width:4px;height:12px;background:#fb923c;border-radius:2px;transform:translateX(-50%)"></div>
                </div>
                <span style="width:38px;text-align:right;font-variant-numeric:tabular-nums">${v.toFixed(2)}</span>
              </div>`;
      });
      h += '</div><div style="display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin-top:7px">';
      gp.buttons.forEach((b, i) => {
        const on = b.pressed, a = (0.06 + b.value * 0.5).toFixed(2);
        h += `<div style="font-size:10px;text-align:center;padding:4px 0;border-radius:4px;
                background:${on ? '#ea580c' : 'rgba(255,255,255,' + a + ')'};
                color:${on ? '#fff' : '#aaa'}">${i}</div>`;
      });
      h += '</div>';
      el.innerHTML = h;
    }
  }
  setTimeout(gamepadViz, 60);                      // ~16 Hz, cheap when panel closed
}

function sendTarget(tg){
  if (V.live && V.joy.ws && V.joy.ws.readyState === 1){
    V.joy.ws.send(JSON.stringify({
      target_head_pose:{x:tg.x,y:tg.y,z:tg.z,roll:tg.roll,pitch:tg.pitch,yaw:tg.yaw},
      target_antennas:[tg.aL,tg.aR], target_body_yaw:tg.by }));
  }
}

function joyTick(){
  if (!V.joy.on || V.joy.phase !== 'control') return;
  const tg = V.joy.tgt, j = V.joy.fps; if (!tg || !j) return;
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gp = null; for (const p of pads){ if (p){ gp = p; break; } }
  if (!gp){
    if (!V.joy.warned){ setStatus('🎮 no gamepad — connect one and press a button', '#d9a441'); V.joy.warned=true; }
    fpsSmoothAndPush(j, tg);   // no pad: still settle toward targets (L3/recall) without input
    return;
  }
  if (V.joy.warned){ V.joy.warned=false; setStatus(JOY_MSG, '#4cae4c'); }
  const ax = gp.axes, bt = gp.buttons;
  const v = (b) => (bt[b] && (bt[b].value || (bt[b].pressed ? 1 : 0))) || 0;

  // L3 -> reset to the ready pose
  const l3 = !!(bt[10] && bt[10].pressed);
  if (l3 && !V.joy.prevL3){ const D = window.REACHY_DEFAULT_POSE; if (D) fpsFromPose(j, D); else fpsReadyTargets(j); }
  V.joy.prevL3 = l3;
  // R3 -> save pose: disabled for now (pose UI is off; kept for reuse)
  // const r3 = !!(bt[11] && bt[11].pressed);
  // if (r3 && !V.joy.prevR3) captureAndSave();
  // V.joy.prevR3 = r3;

  // integrate (ebubar/teleop mapping): L stick = neck pan/tilt, R stick = body yaw + height,
  // L2/L1 + R2/R1 = antennas. State accumulates in degrees/mm.
  const [rx, ry] = fpsSnap(ax[0]||0, ax[1]||0);
  j.neck  -= _fdz(rx)*FPS.speed.head;
  j.pitch -= _fdz(ry)*FPS.speed.head;
  j.base  -= _fdz(ax[2]||0)*FPS.speed.base;
  j.z     -= _fdz(ax[3]||0)*FPS.speed.z;
  const L2 = v(6), R2 = v(7);
  if (L2 > 0.1) j.antL += FPS.speed.ant*L2;
  if (bt[4] && bt[4].pressed) j.antL -= FPS.speed.ant;
  if (R2 > 0.1) j.antR -= FPS.speed.ant*R2;
  if (bt[5] && bt[5].pressed) j.antR += FPS.speed.ant;

  fpsSmoothAndPush(j, tg);
}

// sim free-control: compute the Stewart joints from the head pose via the WASM IK, then drive
// the FULL rig (legs included) through updateJoints — same path as recorded playback.
const HEAD_Z_OFFSET = 0.177;   // SDK ik() adds this to z before solving (verified vs server IK)

function _ikStewart(M, by){                        // 6 stewart joints, or null if unreachable
  const Mik = M.slice(); Mik[11] += HEAD_Z_OFFSET;  // IK frame (z offset added, like the SDK)
  try {
    const st = V.ik.inverse_kinematics(new Float64Array(Mik), by);
    return (st && st.length === 6 && st.every(v => Number.isFinite(v))) ? st : null;  // NaN = unreachable
  } catch(e){ return null; }
}
function _headSnap(j){ return { neck:j.neck, base:j.base, pitch:j.pitch, z:j.z,
  cNeck:j.cNeck, cBase:j.cBase, cPitch:j.cPitch, cZ:j.cZ }; }

// sim free-control: Stewart joints from the head pose via WASM IK -> full rig (legs included).
// Joystick targets are clamped to a verified-reachable envelope (FPS.lim), so the IK is always
// solvable inside it; the cheap NaN backstop below only fires if something slips out.
function driveSim(tg){
  if (!V.robot || !V.ik) return;
  let M = poseMatrix(tg), st = _ikStewart(M, tg.by);     // null only if IK is NaN
  if (!st && V.joy.fps && V.joy.lastGood){               // stateless backstop (no FK -> no drift/stuck)
    Object.assign(V.joy.fps, V.joy.lastGood);
    fpsToTgt(V.joy.fps, tg); M = poseMatrix(tg); st = _ikStewart(M, tg.by);
  }
  if (!st) return;
  updateJoints({ head_pose: M, head_joints: [tg.by, st[0],st[1],st[2],st[3],st[4],st[5]],
                 antennas_position:[tg.aL, tg.aR], body_yaw: tg.by });
  if (V.joy.fps) V.joy.lastGood = _headSnap(V.joy.fps);
}

function parseUrdfColors(urdfText){
  const doc = new DOMParser().parseFromString(urdfText, 'application/xml');
  const map = {};
  doc.querySelectorAll('visual').forEach(v => {
    const mesh = v.querySelector('geometry mesh');
    const matEl = v.querySelector('material');
    const col = v.querySelector('material color');
    if (mesh && col){
      const name = mesh.getAttribute('filename').split('/').pop();
      const [r,g,b,a] = col.getAttribute('rgba').split(' ').map(Number);
      map[name] = { color: new THREE.Color(r,g,b), opacity: a,
                    name: matEl ? matEl.getAttribute('name') : null };
    }
  });
  return map;
}

// ---- channels chart (SVG) with live playhead + click/drag scrub ----
const CH_PANELS = [
  { label:'head °',   keys:[['roll','#fbbf24'],['pitch','#60a5fa'],['yaw','#f472b6']] },
  { label:'head mm',  keys:[['x','#fbbf24'],['y','#34d399'],['z','#60a5fa']] },
  { label:'°',        keys:[['antL','#34d399'],['antR','#a78bfa'],['body','#fb923c']] },
];

function buildChart(traj){
  const el = document.getElementById('reachy-chart');
  if (!el || !traj || !traj.channels) return;
  const w = el.clientWidth || 360, h = 330;
  const L=34, Rm=8, T=8, B=16, gap=12;
  const x0=L, x1=w-Rm, dur=traj.time[traj.time.length-1] || 1;
  const ph=(h-T-B-2*gap)/3;
  const tx = t => x0 + (t/dur)*(x1-x0);
  let svg = `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block">`;
  CH_PANELS.forEach((p, pi) => {
    const yTop = T + pi*(ph+gap);
    let lo=1e9, hi=-1e9;
    p.keys.forEach(([k]) => traj.channels[k].forEach(v => { if(v<lo)lo=v; if(v>hi)hi=v; }));
    if (hi-lo < 1e-6){ hi+=1; lo-=1; }
    const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
    const vy = v => yTop + ph - (v-lo)/(hi-lo)*ph;
    svg += `<rect x="${x0}" y="${yTop}" width="${x1-x0}" height="${ph}" fill="rgba(255,255,255,0.03)"/>`;
    svg += `<line x1="${x0}" y1="${vy(0)}" x2="${x1}" y2="${vy(0)}" stroke="rgba(255,255,255,0.12)"/>`;
    svg += `<text x="2" y="${yTop+9}" fill="rgba(255,255,255,0.5)" font-size="9">${p.label}</text>`;
    p.keys.forEach(([k,c]) => {
      const pts = traj.channels[k].map((v,i)=>`${tx(traj.time[i]).toFixed(1)},${vy(v).toFixed(1)}`).join(' ');
      svg += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.3"/>`;
    });
  });
  svg += `<line id="reachy-ph" x1="${x0}" y1="${T}" x2="${x0}" y2="${h-B}" stroke="#fff" stroke-width="1.2" opacity="0.85"/>`;
  svg += `</svg>`;
  el.innerHTML = svg;
  V.chart = { x0, x1, dur, el };
}

function updatePlayhead(){
  if (!V.chart) return;
  const ph = document.getElementById('reachy-ph');
  if (!ph) return;
  const x = V.chart.x0 + (Math.min(V.t, V.chart.dur)/V.chart.dur)*(V.chart.x1-V.chart.x0);
  ph.setAttribute('x1', x); ph.setAttribute('x2', x);
}

// ---- live channels (rolling scope) while connected to the robot ----
function mat16euler(m){            // intrinsic xyz (matches scipy as_euler('xyz'))
  const s = Math.max(-1, Math.min(1, m[2]));
  const d = 180/Math.PI;
  return [Math.atan2(-m[6], m[10])*d, Math.asin(s)*d, Math.atan2(-m[1], m[0])*d];
}

function pushLiveSample(data){
  if (!data) return;
  const now = performance.now()/1000;
  let m = null; const hp = data.head_pose;
  if (Array.isArray(hp) && hp.length===16) m = hp; else if (hp && hp.m) m = hp.m;
  let roll=0,pitch=0,yaw=0,x=0,y=0,z=0;
  if (m){ [roll,pitch,yaw] = mat16euler(m); x=m[3]*1000; y=m[7]*1000; z=m[11]*1000; }
  const ant = data.antennas_position || [0,0], D = 180/Math.PI;
  V.liveBuf.push({ t:now, roll, pitch, yaw, x, y, z,
                   antL:ant[0]*D, antR:ant[1]*D, body:(data.body_yaw||0)*D });
  while (V.liveBuf.length && V.liveBuf[0].t < now - V.liveWindow) V.liveBuf.shift();
  if (now - V.liveLast > 0.06){ V.liveLast = now; buildLiveChart(); }  // ~16 Hz redraw
}

function buildLiveChart(){
  const el = document.getElementById('reachy-chart');
  if (!el || V.liveBuf.length < 2) return;
  const w = el.clientWidth || 360, h = 330, L=34, Rm=8, T=8, B=16, gap=12;
  const x0=L, x1=w-Rm, ph=(h-T-B-2*gap)/3;
  const now = performance.now()/1000, win = V.liveWindow, t0 = now - win, buf = V.liveBuf;
  const tx = t => x0 + ((t-t0)/win)*(x1-x0);
  let svg = `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block">`;
  CH_PANELS.forEach((p, pi) => {
    const yTop = T + pi*(ph+gap);
    let lo=1e9, hi=-1e9;
    p.keys.forEach(([k]) => buf.forEach(s => { if(s[k]<lo)lo=s[k]; if(s[k]>hi)hi=s[k]; }));
    if (hi-lo < 1e-6){ hi+=1; lo-=1; }
    const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
    const vy = v => yTop + ph - (v-lo)/(hi-lo)*ph;
    svg += `<rect x="${x0}" y="${yTop}" width="${x1-x0}" height="${ph}" fill="rgba(255,255,255,0.03)"/>`;
    svg += `<line x1="${x0}" y1="${vy(0)}" x2="${x1}" y2="${vy(0)}" stroke="rgba(255,255,255,0.12)"/>`;
    svg += `<text x="2" y="${yTop+9}" fill="rgba(255,255,255,0.5)" font-size="9">${p.label}</text>`;
    p.keys.forEach(([k,c]) => {
      const pts = buf.map(s => `${tx(s.t).toFixed(1)},${vy(s[k]).toFixed(1)}`).join(' ');
      svg += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.3"/>`;
    });
  });
  svg += `<line x1="${x1}" y1="${T}" x2="${x1}" y2="${h-B}" stroke="#fff" stroke-width="1" opacity="0.5"/>`;
  svg += `</svg>`;
  el.innerHTML = svg;
}

function chartScrub(clientX){
  if (V.live || !V.chart || !V.traj) return;
  const r = V.chart.el.getBoundingClientRect();
  const px = (clientX - r.left) / r.width * (V.chart.el.clientWidth || r.width);
  let t = (px - V.chart.x0)/(V.chart.x1 - V.chart.x0) * V.chart.dur;
  t = Math.max(0, Math.min(V.chart.dur, t));
  V.t = t; V.playing = false; V.scrubbing = true;
  if (V.audio && V.audio.src){ V.audio.pause(); V.audio.currentTime = t; }
  const sl = document.getElementById('reachy-time'); if (sl) sl.value = (t/V.chart.dur*1000)|0;
  const pb = document.getElementById('reachy-play'); if (pb) pb.textContent = '▶ Play';
  if (V.ready && V.traj && !V.live) applyFrame();   // immediately reflect the scrubbed pose
}

// ---- voice waveform (decoded via Web Audio) + scrub ----
async function buildWaveform(url){
  const el = document.getElementById('reachy-wave'); if (!el) return;
  if (!url){ el.innerHTML = '<div style="opacity:.45;font-size:12px;padding:24px 0">no sound for this move</div>'; V.waveUrl = null; V.waveW = 0; return; }
  if (url === V.waveUrl) return;
  V.waveUrl = url;
  el.innerHTML = '<div style="opacity:.45;font-size:12px;padding:24px 0">loading waveform…</div>';
  try {
    if (!V.audioCtx) V.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await V.audioCtx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    if (url !== V.waveUrl) return;   // a newer move was selected while decoding
    const data = buf.getChannelData(0);
    const w = el.clientWidth || 360, h = 70, N = Math.min(w, 600), step = Math.max(1, Math.floor(data.length/N)), mid = h/2;
    let bars = '';
    for (let i=0;i<N;i++){
      let mx = 0; for (let j=0;j<step;j++){ const v = Math.abs(data[i*step+j]||0); if (v>mx) mx = v; }
      const x = (i/N)*w, bh = Math.max(1, mx*h*0.9);
      bars += `<rect x="${x.toFixed(1)}" y="${(mid-bh/2).toFixed(1)}" width="${(w/N*0.8).toFixed(2)}" height="${bh.toFixed(1)}" fill="#60a5fa" opacity="0.7"/>`;
    }
    el.innerHTML = `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block;cursor:col-resize">
      <rect x="0" y="0" width="${w}" height="${h}" fill="rgba(255,255,255,0.03)"/>${bars}
      <line id="reachy-wave-ph" x1="0" y1="0" x2="0" y2="${h}" stroke="#fff" stroke-width="1.2" opacity="0.85"/></svg>`;
    V.waveW = w;
  } catch(e){ console.error('waveform decode failed', e); el.innerHTML = '<div style="opacity:.45;font-size:12px;padding:24px 0">waveform unavailable</div>'; V.waveW = 0; }
}
function updateWavePlayhead(){
  const ph = document.getElementById('reachy-wave-ph');
  if (!ph || !V.waveW || !V.dur) return;
  const x = Math.min(V.t, V.dur)/V.dur * V.waveW;
  ph.setAttribute('x1', x); ph.setAttribute('x2', x);
}
function waveScrub(clientX){
  if (V.live || !V.waveW || !V.dur) return;
  const el = document.getElementById('reachy-wave'); const r = el.getBoundingClientRect();
  let t = (clientX - r.left)/r.width * V.dur;
  t = Math.max(0, Math.min(V.dur, t));
  V.t = t; V.playing = false; V.scrubbing = true;
  if (V.audio && V.audio.src){ V.audio.pause(); V.audio.currentTime = t; }
  const sl = document.getElementById('reachy-time'); if (sl) sl.value = (t/V.dur*1000)|0;
  const pb = document.getElementById('reachy-play'); if (pb) pb.textContent = '▶ Play';
  if (V.ready && V.traj && !V.live) applyFrame();
}

// ---- robot camera: consume the daemon's WebRTC stream into the <video> ----
// The daemon publishes a producer ('reachy_mini') on its GStreamer signaling server (:8443).
// gstwebrtc-api (loaded in <head>) connects, finds the producer, and gives us a MediaStream.
function camStatus(m){ const e = document.getElementById('reachy-cam-status'); if (e) e.textContent = m; }
function startCamera(){
  if (V.cam) return;                                   // already running
  if (!window.GstWebRTCAPI){ camStatus('camera library not loaded'); return; }
  const host = location.hostname || 'localhost';
  camStatus('connecting…');
  const cam = { api:null, session:null, prodL:null };
  V.cam = cam;
  try {
    cam.api = new window.GstWebRTCAPI({
      signalingServerUrl: `ws://${host}:8443`,
      reconnectionTimeout: 2000,
      meta: { name: 'reachy-motion-viewer' },
      webrtcConfig: { iceServers: [{ urls:'stun:stun.l.google.com:19302' }] },
    });
    const onProducer = (producer) => {
      if (V.cam !== cam || cam.session) return;          // one session; ignore if torn down
      const session = cam.api.createConsumerSession(producer.id);
      if (!session){ camStatus('no consumer session'); return; }
      cam.session = session;
      session.addEventListener('streamsChanged', () => {
        const s = session.streams;
        if (s && s.length){ const v = document.getElementById('reachy-cam');
          if (v){ v.srcObject = s[0]; v.play().catch(()=>{}); } camStatus(''); }
      });
      session.addEventListener('error', () => camStatus('stream error'));
      session.addEventListener('closed', () => { if (cam.session === session) cam.session = null; });
      session.connect();
    };
    cam.prodL = { producerAdded: onProducer, producerRemoved: () => {} };
    cam.api.registerPeerListener(cam.prodL);   // fires producerAdded for existing + new producers
    try { (cam.api.getAvailableProducers() || []).forEach(onProducer); } catch(_){}  // backup for already-listed
  } catch(e){ console.error('camera', e); camStatus('camera unavailable'); V.cam = null; }
}
function stopCamera(){
  const cam = V.cam; if (!cam) return; V.cam = null;
  try { if (cam.session) cam.session.close(); } catch(_){}
  try { if (cam.api && cam.prodL) cam.api.unregisterPeerListener(cam.prodL); } catch(_){}
  try { if (cam.api && cam.api._channel) cam.api._channel.close(); } catch(_){}   // close signaling ws
  const v = document.getElementById('reachy-cam'); if (v) v.srcObject = null;
  camStatus('camera off — expand to start');
}
function camTick(){
  const v = document.getElementById('reachy-cam');
  const open = !!(v && v.offsetParent !== null);        // accordion expanded -> content visible
  if (V.live && open){ if (!V.cam) startCamera(); }
  else if (V.cam){ stopCamera(); }
  else if (open && !V.live){ camStatus('connect to the robot (Connected tab) to start the camera'); }
}

// ---- poses: capture the current pose and push it to the gradio backend ----
function currentPose(){
  if (V.joy.on && V.joy.tgt){
    const t = V.joy.tgt;
    return {x:t.x,y:t.y,z:t.z,roll:t.roll,pitch:t.pitch,yaw:t.yaw,antL:t.aL,antR:t.aR,body:t.by};
  }
  if (V.traj){
    const ts = V.traj.time; let i=1; while (i<ts.length && ts[i]<V.t) i++;
    const k = Math.max(0, i-1), m = V.traj.head[k], e = mat16euler(m), R = Math.PI/180;
    return {x:m[3],y:m[7],z:m[11], roll:e[0]*R, pitch:e[1]*R, yaw:e[2]*R,
            antL:V.traj.antennas[k][0], antR:V.traj.antennas[k][1], body:V.traj.body_yaw[k]};
  }
  return null;
}

function captureAndSave(){
  const p = currentPose();
  if (!p){ setStatus('nothing to save yet', '#d9a441'); return; }
  const btn = document.querySelector('#pose_save_btn button') || document.getElementById('pose_save_btn');
  if (btn){ btn.click(); setStatus('💾 pose saved', '#4cae4c'); }   // js getCurrentPose() supplies the value
}

// their updateJoints(): drive the full Stewart platform + passive joints + antennas
function updateJoints(data){
  if (!V.robot) return;
  let headPose = null;
  if (data.head_pose){
    if (Array.isArray(data.head_pose) && data.head_pose.length===16) headPose = data.head_pose;
    else if (data.head_pose.m) headPose = data.head_pose.m;        // {m:[16]} from the daemon
    else if (V.buildHeadPose) headPose = V.buildHeadPose(data.head_pose);  // {x,y,z,roll,..}
  }
  const headJoints = (data.head_joints?.length===7) ? data.head_joints : [data.body_yaw||0,0,0,0,0,0,0];
  for (let i=0;i<7;i++){ const j=V.jointMap[HEAD_JOINT_NAMES[i]]; if (j) j.setJointValue(headJoints[i]); }
  if (headPose && V.calcPassive){
    const pj = V.calcPassive(headJoints, headPose);
    for (let i=0;i<21;i++){ const j=V.jointMap[PASSIVE_JOINT_NAMES[i]]; if (j) j.setJointValue(pj[i]); }
  }
  if (data.antennas_position?.length>=2){
    V.jointMap['right_antenna']?.setJointValue(-data.antennas_position[0]);
    V.jointMap['left_antenna']?.setJointValue(-data.antennas_position[1]);
  }
}

function applyFrame(){
  const tr = V.traj, ts = tr.time;
  let i=1; while (i<ts.length && ts[i]<V.t) i++;
  const k = Math.max(0, i-1);
  updateJoints({ head_pose: tr.head[k], head_joints: tr.head_joints[k],
                 body_yaw: tr.body_yaw[k], antennas_position: tr.antennas[k] });
  const sl = document.getElementById('reachy-time');
  if (sl && !V.scrubbing) sl.value = (V.t / V.dur * 1000) | 0;
  updatePlayhead();
  updateWavePlayhead();
}

window.ReachyViewer = {
  async init(containerId, urdfUrl, meshBase, kinUrl){
    const el = document.getElementById(containerId);
    if (!el || V.scene) return;
    const w = el.clientWidth || 640, h = 420;

    // --- scene + lighting + shadows (their SceneManager) ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x12182a);
    const cam = new THREE.PerspectiveCamera(38, w/h, 0.02, 10);
    cam.position.set(0.34, 0.26, 0.40);
    const r = new THREE.WebGLRenderer({antialias:true});
    r.setSize(w, h); r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.0;
    el.appendChild(r.domElement);
    const ctrl = new OrbitControls(cam, r.domElement);
    ctrl.target.set(0, 0.11, 0); ctrl.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(2,1,2);
    key.castShadow = true; key.shadow.mapSize.set(1024,1024);
    Object.assign(key.shadow.camera, {near:0.1, far:10, left:-1, right:1, top:1, bottom:-1});
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xFFB366, 0.6); fill.position.set(-2,0.5,1.5); scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.4); rim.position.set(0,1.2,-2); scene.add(rim);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3); scene.add(hemi);
    // NOTE: no environment map — 8bitkick leaves it off; an envMap lifts black matte
    // surfaces to gray (washes out the antennas/face). Direct lights only, like theirs.
    // ground shadow catcher
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(2,2), new THREE.ShadowMaterial({opacity:0.3}));
    ground.rotation.x = -Math.PI/2; ground.receiveShadow = true; scene.add(ground);
    V.grid = new THREE.GridHelper(1.0, 20, 0x2c3a57, 0x1c2740); scene.add(V.grid);
    Object.assign(V, {scene, cam, r, ctrl});
    V.audio = document.getElementById('reachy-audio');

    // kinematics (their passive-joint solver)
    try { const K = await import(kinUrl); V.calcPassive = K.calculatePassiveJoints; V.buildHeadPose = K.buildHeadPoseMatrix; }
    catch(e){ console.error('Kinematics import failed', e); }

    // load the Rust IK (WASM) so joystick control can compute Stewart joints -> full rig renders
    (async () => {
      try {
        const ik = await import(window.REACHY_IK_WASM_URL);
        await ik.default();                                   // init wasm (resolves _bg.wasm same dir)
        const data = await (await fetch(window.REACHY_IK_DATA_URL)).text();
        ik.init_kinematics(data);
        V.ik = ik;
        console.log('IK wasm ready');
      } catch(e){ console.error('IK wasm load failed', e); }
    })();

    // --- load URDF + per-part DRACO meshes (their RobotManager) ---
    const urdfText = await (await fetch(urdfUrl)).text();
    const colors = parseUrdfColors(urdfText);
    const draco = new DRACOLoader();
    draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
    const gltf = new GLTFLoader(); gltf.setDRACOLoader(draco);

    const loader = new URDFLoader();
    loader.packages = { 'assets': meshBase, 'reachy_mini_description': meshBase };
    loader.workingPath = meshBase;
    loader.loadMeshCb = (path, manager, onComplete) => {
      const filename = path.split('/').pop();
      const md = colors[filename] || {};
      const opacity = md.opacity ?? 1, transp = opacity < 0.4;
      const isAnt = filename.includes('antenna_V2') || md.name === 'antenna_material';
      const material = new THREE.MeshPhysicalMaterial({
        color: (isAnt || transp) ? 0x202020 : (md.color || new THREE.Color(0.8,0.8,0.8)),
        metalness: 0.0, roughness: (isAnt || transp) ? 0.05 : 0.7,
        transparent: transp, opacity, side: transp ? THREE.DoubleSide : THREE.FrontSide,
      });
      if (filename.includes('link')){ material.color.setHex(0xffffff); material.metalness=1.0; material.roughness=0.3; }
      if (md.name === 'antenna_material'){ material.clearcoat=1.0; material.clearcoatRoughness=0.0; material.reflectivity=1.0; material.envMapIntensity=1.5; }
      gltf.load(meshBase + filename, (g) => {
        let geom=null; g.scene.traverse(c=>{ if(c.isMesh && !geom) geom=c.geometry; });
        const mesh = geom ? new THREE.Mesh(geom, material) : g.scene;
        mesh.castShadow = true; mesh.receiveShadow = true;
        onComplete(mesh);
      }, undefined, (e)=>{ console.error('mesh load', filename, e); onComplete(null, e); });
    };

    const blob = URL.createObjectURL(new Blob([urdfText], {type:'application/xml'}));
    loader.load(blob, (robot) => {
      URL.revokeObjectURL(blob);
      robot.rotation.x = -Math.PI/2;            // URDF Z-up -> three.js Y-up
      scene.add(robot);
      V.robot = robot;
      robot.traverse(c => { if (c.isURDFJoint) V.jointMap[c.name] = c;
                            if (c.isMesh){ c.castShadow=true; c.receiveShadow=true; } });
      V.ready = true;
      if (V.traj){ if (V.traj.audio && V.audio){ V.audio.src = V.traj.audio; V.audio.play().catch(()=>{}); } }
      else showReady();   // empty dropdown -> static ready pose
    }, undefined, (e)=> console.error('URDF load error', e));

    const clock = new THREE.Clock();
    const loop = () => {
      requestAnimationFrame(loop);
      const dt = clock.getDelta();
      if (V.ready && !V.live){
        const mp = V.moveEl || (V.moveEl = document.getElementById('move-pick'));
        const animateActive = !!(mp && mp.offsetParent);   // Animate sub-tab visible
        if (V.joy.on && V.joy.phase === 'control'){
          driveSim(V.joy.tgt);                      // sim free-control: IK -> full rig, every frame
        } else if (V.traj && !V.joy.on && animateActive){
          // drive playback on its own clock (NOT slaved to audio.currentTime — that freezes the
          // move at frame 0 whenever audio stalls / has no device). Audio plays alongside, best-effort.
          if (V.playing){
            V.t += dt;
            if (V.t > V.dur){ V.t = 0; if (V.audio && V.audio.src){ try { V.audio.currentTime = 0; } catch(e){} } }
          }
          applyFrame();
        }
      }
      ctrl.update(); r.render(scene, cam);
    };
    loop();

    const sl = document.getElementById('reachy-time');
    if (sl){
      sl.addEventListener('input', () => { V.scrubbing=true; V.playing=false;
        V.t=(sl.value/1000)*V.dur; if(V.audio){V.audio.pause(); V.audio.currentTime=V.t;} });
      sl.addEventListener('change', () => { V.scrubbing=false; });
    }
    const pb = document.getElementById('reachy-play');
    if (pb) pb.addEventListener('click', () => { V.playing=!V.playing;
      if (V.audio && V.audio.src){ if(V.playing) V.audio.play().catch(()=>{}); else V.audio.pause(); }
      pb.textContent = V.playing ? '⏸ Pause' : '▶ Play'; });

    // channels chart: click/drag to scrub the timeline
    const chartEl = document.getElementById('reachy-chart');
    if (chartEl){
      let dragging = false;
      chartEl.style.cursor = 'col-resize';
      chartEl.addEventListener('pointerdown', e => { dragging = true; chartScrub(e.clientX);
        try { chartEl.setPointerCapture(e.pointerId); } catch(_){} });
      chartEl.addEventListener('pointermove', e => { if (dragging) chartScrub(e.clientX); });
      const end = () => { dragging = false; V.scrubbing = false; };
      chartEl.addEventListener('pointerup', end);
      chartEl.addEventListener('pointercancel', end);
      window.addEventListener('resize', () => { if (V.traj) buildChart(V.traj); });
    }
    if (V.traj) buildChart(V.traj);

    // voice: volume slider + waveform click/drag to scrub
    const vol = document.getElementById('reachy-vol');
    if (vol && V.audio){ V.audio.volume = vol.value/100;
      vol.addEventListener('input', () => { if (V.audio) V.audio.volume = vol.value/100; }); }
    const waveEl = document.getElementById('reachy-wave');
    if (waveEl){
      let wd = false;
      waveEl.addEventListener('pointerdown', e => { wd = true; waveScrub(e.clientX);
        try { waveEl.setPointerCapture(e.pointerId); } catch(_){} });
      waveEl.addEventListener('pointermove', e => { if (wd) waveScrub(e.clientX); });
      const wend = () => { wd = false; V.scrubbing = false; };
      waveEl.addEventListener('pointerup', wend);
      waveEl.addEventListener('pointercancel', wend);
    }

    // spacebar -> save pose: disabled for now (pose UI is off; kept for reuse)
    // document.addEventListener('keydown', (e) => {
    //   if (e.code === 'Space' && !/^(INPUT|TEXTAREA|SELECT)$/.test((e.target && e.target.tagName) || '')){
    //     e.preventDefault(); captureAndSave();
    //   }
    // });

    gamepadViz();    // start the live gamepad tester (self-schedules)
    autoControl();   // auto-enable control when a gamepad is on the Control sub-tab
    setInterval(camTick, 700);   // start/stop the robot camera with the Camera accordion + connection
  },

  playMove(traj){
    if (typeof traj === 'string'){ traj = traj ? JSON.parse(traj) : null; }
    if (!traj || !traj.time){ V.traj = null; showReady(); return; }   // empty selection -> ready
    V.traj = traj; V.dur = traj.time[traj.time.length-1]; V.t = 0; V.playing = true;
    if (!V.live) buildChart(traj);   // while live, the rolling scope owns the chart
    if (V.audio){
      if (traj.audio){ V.audio.src = traj.audio; V.audio.currentTime = 0; V.audio.play().catch(()=>{}); }
      else { V.audio.removeAttribute('src'); V.audio.load(); }
    }
    buildWaveform(traj.audio || null);   // show the sound's waveform beneath the channels
    const pb = document.getElementById('reachy-play'); if (pb) pb.textContent = '⏸ Pause';
  },

  // --- live connect: mirror the real robot via the daemon state WebSocket ---
  connectRobot(host){
    if (V.ws) return;
    host = host || (location.hostname || '127.0.0.1');
    const url = `ws://${host}:8000/api/state/ws/full?frequency=30&with_head_joints=true&use_pose_matrix=true`;
    setStatus('connecting…', '#d9a441');
    let ws;
    try { ws = new WebSocket(url); } catch(e){ setStatus('connect failed', '#d9534f'); return; }
    V.ws = ws;
    ws.onopen = () => { V.live = true; V.liveBuf = []; V.liveLast = 0;
      setStatus('● live — mirroring robot', '#4cae4c');
      const b=document.getElementById('reachy-connect'); if(b) b.textContent='Disconnect robot';
      if (V.audio) V.audio.pause(); };
    ws.onmessage = (ev) => { if (!V.ready) return;
      try { const d = JSON.parse(ev.data); updateJoints(d); pushLiveSample(d); } catch(e){} };
    ws.onerror = () => setStatus('connection error (is the daemon running on :8000?)', '#d9534f');
    ws.onclose = () => { V.live = false; V.ws = null;
      const b=document.getElementById('reachy-connect'); if(b) b.textContent='Connect robot';
      setStatus('disconnected', '#888'); };
  },
  disconnectRobot(){ if (V.ws){ V.ws.close(); V.ws = null; } V.live = false;
    V.liveBuf = []; if (V.traj) buildChart(V.traj); },  // restore recorded chart
  toggleRobot(){ if (V.live || V.ws){ this.setJoystick(false); this.disconnectRobot(); } else this.connectRobot(); },

  setMode(connected){
    if (connected){ this.connectRobot(); }   // robot was set to ready by go_connected; mirror shows it
    else { this.setJoystick(false); this.disconnectRobot(); if (!V.traj) showReady(); }  // enter Simulator -> ready
    this.setTheme(connected);
  },

  setTheme(connected){
    document.body.classList.toggle('reachy-connected', !!connected);
    if (!V.scene) return;
    V.scene.background = new THREE.Color(connected ? 0x2a1606 : 0x12182a);
    if (V.grid) V.scene.remove(V.grid);
    V.grid = new THREE.GridHelper(1.0, 20,
      connected ? 0x7c2d12 : 0x2c3a57, connected ? 0x431407 : 0x1c2740);
    V.scene.add(V.grid);
  },

  getCurrentPose(){ const p = currentPose(); return p ? JSON.stringify(p) : ""; },

  pushAudio(j){    // render the robot mic's level waveform + voice-detected indicator
    const cv = document.getElementById('reachy-audio-mon');
    const dot = document.getElementById('reachy-voice-dot');
    const label = document.getElementById('reachy-voice-label');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    let d = null; try { d = j ? JSON.parse(j) : null; } catch(e){}
    if (!d){
      if (dot) dot.style.background = '#444';
      if (label){ label.textContent = "connect to hear the robot's mic"; }
      const tr = document.getElementById('reachy-transcript'); if (tr) tr.innerHTML = '';
      return;
    }
    const levels = d.levels || [], n = levels.length, w = cv.width, h = cv.height, bw = w / Math.max(1, n);
    ctx.fillStyle = d.active ? '#fb923c' : '#60a5fa';
    for (let i = 0; i < n; i++){
      const bh = Math.max(1, levels[i] * h);
      ctx.fillRect(i * bw, (h - bh) / 2, bw * 0.7, bh);   // centered bars
    }
    if (dot) dot.style.background = d.active ? '#22c55e' : '#3b4252';
    if (label) label.textContent = d.active ? 'voice detected' : 'listening…';
    // live transcript: committed lines (solid) + the in-progress utterance (grey/italic)
    const tr = document.getElementById('reachy-transcript');
    if (tr){
      const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
      let html = (d.committed || []).slice(-3).map(t => '<div>' + esc(t) + '</div>').join('');
      if (d.interim) html += '<div style="opacity:0.55;font-style:italic">' + esc(d.interim) + '</div>';
      if (!html) html = '<div style="opacity:0.4">' + (d.stt_ready === false ? 'loading speech model…' : 'say something…') + '</div>';
      tr.innerHTML = html;
    }
  },

  applyPose(payload){
    if (typeof payload === 'string'){ if (!payload) return; payload = JSON.parse(payload.split('|')[0]); }
    if (!payload || !V.ready) return;
    updateJoints(payload);                       // show it in 3D
    if (V.robot) V.robot.updateMatrixWorld(true);
    // if joysticking, retarget the FPS state so control eases to the recalled pose
    if (V.joy.on && V.joy.fps && payload.head_pose){
      const m = payload.head_pose, e = mat16euler(m), j = V.joy.fps, G = 180/Math.PI;
      j.base = payload.body_yaw*G; j.neck = e[2] - j.base; j.pitch = e[1];
      j.z = m[11]*1000; j.antL = payload.antennas_position[0]*G; j.antR = payload.antennas_position[1]*G;
    }
  },

  setJoystick(on){
    if (on){
      if (V.joy.on) return;
      V.joy.on = true; V.joy.warned = false;
      if (V.live){
        // connected: stream targets to the real robot; control immediately from neutral
        const host = location.hostname || '127.0.0.1';
        try { V.joy.ws = new WebSocket(`ws://${host}:8000/api/move/ws/set_target`); }
        catch(e){ setStatus('🎮 joystick connection failed', '#d9534f'); V.joy.on = false; return; }
        startControl();
      } else {
        startControl();   // simulator: snap to the ready pose, then free-control (3a)
      }
      setStatus(JOY_MSG, '#4cae4c');
    } else {
      V.joy.on = false; V.joy.phase = null;
      if (V.joy.timer) clearInterval(V.joy.timer);
      V.joy.timer = 0;
      if (V.joy.ws){ try { V.joy.ws.close(); } catch(e){} V.joy.ws = null; }
      V.joy.tgt = null; V.joy.fps = null;   // clear so each enable starts fresh
      if (!V.live && !V.traj) showReady();   // no move loaded -> static ready (else Animate resumes it)
    }
  },
};

function setStatus(text, color){
  const el = document.getElementById('reachy-status');
  if (el){ el.textContent = text; el.style.color = color || '#888'; }
}

(function(){
  const poll = setInterval(() => {
    if (!window.ReachyViewer) return;
    const el = document.getElementById('reachy3d');
    if (el && !V.scene && window.REACHY_URDF_URL && window.REACHY_MESH_BASE && window.REACHY_KIN_URL){
      window.ReachyViewer.init('reachy3d', window.REACHY_URDF_URL, window.REACHY_MESH_BASE, window.REACHY_KIN_URL);
      clearInterval(poll);
    }
  }, 200);
})();
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
<div style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
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
