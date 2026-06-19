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
            edit:{ spec:{fps:40, segments:[], layers:[]}, sel:-1, posing:true },  // Phase-1 authoring
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
  // verified all-reachable envelope (81/81 corners incl. body-yaw): no detach anywhere inside it.
  // antennas are free rotational joints (not part of the Stewart IK) -> a full ±180 (360 span).
  lim:{ neck:30, pitch:15, zMin:-15, zMax:15, ant:180, base:90 } };
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
  const dn = _fdz(rx), dp = _fdz(ry), db = _fdz(ax[2]||0), dz = _fdz(ax[3]||0);
  j.neck  -= dn*FPS.speed.head;
  j.pitch -= dp*FPS.speed.head;
  j.base  -= db*FPS.speed.base;
  j.z     -= dz*FPS.speed.z;
  const L2 = v(6), R2 = v(7);
  const aL1 = !!(bt[4] && bt[4].pressed), aR1 = !!(bt[5] && bt[5].pressed);
  if (L2 > 0.1) j.antL += FPS.speed.ant*L2;
  if (aL1) j.antL -= FPS.speed.ant;
  if (R2 > 0.1) j.antR -= FPS.speed.ant*R2;
  if (aR1) j.antR += FPS.speed.ant;

  // Authoring: the keyframe is the source of truth. Only fold the gamepad in while the stick is
  // ACTIVELY moving — when idle, sync the FPS state FROM the keyframe so manual edits (pad/sliders/
  // 3D drag, possibly beyond the joystick's envelope) aren't clobbered and the next move continues from there.
  const active = dn || dp || db || dz || L2 > 0.1 || R2 > 0.1 || aL1 || aR1 || l3;
  if (authoringJoy()){
    if (active){ fpsSmoothAndPush(j, tg); joyToKeyframe(); }
    else keyframeToJoy();
    return;
  }
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

// ---- client-side animation bake (the JS twin of reachy_motion/anim.py) ----
// Authored behaviors arrive as a tiny SPEC (eased pose segments + additive "life" layers),
// not a baked trajectory. We bake them HERE — pose interpolation + the WASM IK that gamepad
// control already uses — into the same {time,head,head_joints,antennas,body_yaw,channels}
// structure recorded moves produce, so all the playback/chart/scrub code below is unchanged.
// Keeping the bake on the client means edits/selections ship hundreds of bytes, not ~60KB,
// and never run server-side IK — which is what made browsing/editing hang.
const ANIM_CH = ['x','y','z','roll','pitch','yaw','antL','antR','body'];
const ANIM_EASES = {
  linear:    u => u,
  smooth:    u => u*u*u*(u*(u*6-15)+10),
  ease_out:  u => 1-Math.pow(1-u,3),
  ease_in:   u => u*u*u,
  back:      u => { const c1=1.70158, c3=c1+1; return 1 + c3*Math.pow(u-1,3) + c1*Math.pow(u-1,2); },
  anticipate:u => { const c1=1.70158; return u*u*((c1+1)*u - c1); },
  hold:      u => u,
};
const ANIM_LAYERS = {
  breath: (p={}) => { const az=p.amp_z??2.0, ap=p.amp_pitch??1.2, per=p.period??4.0;
    return t => { const s=Math.sin(2*Math.PI*t/per); return { z:az*s, pitch:ap*s }; }; },
  ear_idle: (p={}) => { const a=p.amp??4.0, per=p.period??2.3;
    return t => ({ antL:a*Math.sin(2*Math.PI*t/per), antR:a*Math.sin(2*Math.PI*t/per+0.7) }); },
};
function animLerp(a, b, u){ const o={}; for (const c of ANIM_CH) o[c]=a[c]+(b[c]-a[c])*u; return o; }
function buildAnim(spec){                              // spec -> { fps, dur, sample(t) }
  const fps = +spec.fps || 40, ch = ANIM_CH;
  let t = 0, prev = Object.fromEntries(ch.map(c=>[c,0]));   // start at NEUTRAL
  const segs = [];
  for (const s of (spec.segments||[])){
    const dur = Math.max(1e-3, +s.dur || 0.001);
    const pose = Object.fromEntries(ch.map(c=>[c, +((s.pose||{})[c]) || 0]));
    segs.push({ t0:t, t1:t+dur, a:prev, b:pose, ease:s.ease||'smooth' });
    t += dur; prev = pose;
  }
  const layers = (spec.layers||[]).filter(l=>ANIM_LAYERS[l.type]).map(l => {
    const { type, ...p } = l; return ANIM_LAYERS[type](p);
  });
  function sample(time){
    let p = segs.length ? segs[segs.length-1].b : prev;   // past the end -> hold last pose
    for (const s of segs){ if (time <= s.t1){
      const u = Math.max(0, Math.min(1, (time-s.t0)/(s.t1-s.t0)));
      p = animLerp(s.a, s.b, (ANIM_EASES[s.ease]||ANIM_EASES.smooth)(u)); break; } }
    const d = Object.assign({}, p);
    for (const f of layers){ const dd=f(time); for (const k in dd) d[k]=(d[k]||0)+dd[k]; }
    return d;
  }
  return { fps, dur: t || 0.5, sample };
}
// pose (deg/mm) -> the head matrix + IK joints + antennas (radians), matching server pose_to_render
function animFrame(p){
  const D = Math.PI/180;
  const tg = { x:(p.x||0)/1000, y:(p.y||0)/1000, z:(p.z||0)/1000,
               roll:(p.roll||0)*D, pitch:(p.pitch||0)*D, yaw:(p.yaw||0)*D };
  const by = (p.body||0)*D, M = poseMatrix(tg);
  const st = V.ik ? _ikStewart(M, by) : null;
  return { head_pose:M, body_yaw:by, antennas:[(p.antL||0)*D, (p.antR||0)*D],
           joints: st ? [by, ...st] : null };
}
function bakeSpec(spec){                                // -> same shape as a baked recorded move
  const anim = buildAnim(spec);
  const n = Math.max(2, Math.round(anim.dur*anim.fps)+1);
  const time=[], head=[], head_joints=[], antennas=[], body_yaw=[], channels={};
  ANIM_CH.forEach(c=>channels[c]=[]);
  let lastJoints = null;
  for (let i=0;i<n;i++){
    const t = i/anim.fps, p = anim.sample(t), fr = animFrame(p);
    if (fr.joints) lastJoints = fr.joints;             // IK NaN -> carry the last valid (like the server)
    time.push(t); head.push(fr.head_pose); body_yaw.push(fr.body_yaw); antennas.push(fr.antennas);
    head_joints.push(fr.joints || lastJoints || [fr.body_yaw,0,0,0,0,0,0]);
    ANIM_CH.forEach(c=>channels[c].push(+(+(p[c]||0)).toFixed(2)));
  }
  return { time, head, head_joints, antennas, body_yaw, channels };
}

// ===== Phase-1 authoring: pose-in-3D + keyframe timeline (the viewer owns the spec) =====
// A keyframe IS a segment's target pose. Editing mutates V.edit.spec client-side and re-bakes
// (~1-3ms) — no server round-trip until Save. The selected keyframe is shown by parking the
// playhead at its time, so the normal rAF applyFrame() renders it (incl. life-layer offsets).
// match the gamepad's reachable envelope (FPS.lim: body/base ±90, etc.); antennas spin a full 360
const EDIT_RANGES = { pitch:[-25,20], yaw:[-90,90], roll:[-30,30], z:[-15,15], body:[-90,90], antL:[-180,180], antR:[-180,180] };
function clampCh(ch, v){ const r = EDIT_RANGES[ch]; return r ? Math.max(r[0], Math.min(r[1], v)) : v; }
function zerosPose(){ const p = {}; for (const c of ANIM_CH) p[c] = 0; return p; }
function editSegs(){ return V.edit.spec.segments; }
function selSeg(){ return V.edit.sel >= 0 ? editSegs()[V.edit.sel] : null; }
function kfTime(i){ let t = 0, s = editSegs(); for (let k = 0; k <= i && k < s.length; k++) t += Math.max(1e-3, +s[k].dur || 0.001); return t; }
function authorVisible(){ const ap = document.getElementById('author-pane'); return ap && !ap.classList.contains('pane-hidden'); }
function editMode(){ return V.edit.sel >= 0 && V.edit.posing && authorVisible(); }

function rebakeEdit(){                                  // spec -> baked V.traj for playback/scrub
  const segs = editSegs();
  if (!segs.length){ V.traj = null; showReady(); return; }
  V.traj = bakeSpec(V.edit.spec);
  V.dur = V.traj.time[V.traj.time.length - 1];
  if (!V.live) buildChart(V.traj);
}
function parkOnSelected(){                              // pause + put the playhead on the selected kf
  if (V.edit.sel < 0) return;
  V.playing = false;
  const pb = document.getElementById('reachy-play'); if (pb) pb.textContent = '▶ Play';
  V.t = Math.min(kfTime(V.edit.sel), V.dur || 0);
}
function commitEdit(){ rebakeEdit(); parkOnSelected(); keyframeToJoy(); }   // re-bake, park, keep gamepad in sync

// ---- gamepad <-> keyframe (authoring with a controller; see joyTick / the rAF loop) ----
function authoringJoy(){ return V.joy && V.joy.on && V.joy.phase === 'control' && authorVisible() && V.edit.sel >= 0; }
// fold the live joystick pose into the selected keyframe (the gamepad has no roll/x/y, so leave those)
function joyToKeyframe(){
  const s = selSeg(); if (!s) return;
  const tg = V.joy.tgt, G = 180 / Math.PI;
  s.pose.pitch = clampCh('pitch', tg.pitch * G);
  s.pose.yaw   = clampCh('yaw',   tg.yaw * G);
  s.pose.z     = clampCh('z',     tg.z * 1000);
  s.pose.body  = clampCh('body',  tg.by * G);
  s.pose.antL  = clampCh('antL',  tg.aL * G);
  s.pose.antR  = clampCh('antR',  tg.aR * G);
  V.traj = bakeSpec(V.edit.spec); V.dur = V.traj.time[V.traj.time.length - 1];   // light re-bake (no chart)
  V.t = Math.min(kfTime(V.edit.sel), V.dur);
  syncPosePanel();
}
// seed the FPS state from the keyframe so the rate-control joystick continues from a manual edit
// (and so enabling the joystick doesn't snap the keyframe to the ready pose)
function keyframeToJoy(){
  const j = V.joy && V.joy.fps, s = selSeg(); if (!j || !s) return;
  j.base = s.pose.body || 0; j.neck = (s.pose.yaw || 0) - j.base; j.pitch = s.pose.pitch || 0;
  j.z = s.pose.z || 0; j.antL = s.pose.antL || 0; j.antR = s.pose.antR || 0;
  j.cBase = j.base; j.cNeck = j.neck; j.cPitch = j.pitch; j.cZ = j.z; j.cAntL = j.antL; j.cAntR = j.antR;
}

function syncPosePanel(){
  const s = selSeg();
  document.querySelectorAll('#author-pose .apose').forEach(inp => {
    const ch = inp.dataset.ch, v = s ? (s.pose[ch] || 0) : 0;
    inp.value = v; inp.disabled = !s;
    const lbl = document.querySelector(`#author-pose .aval[data-for="${ch}"]`); if (lbl) lbl.textContent = Math.round(v);
  });
  const dur = document.getElementById('author-dur'), ease = document.getElementById('author-ease');
  if (s){ if (dur) dur.value = s.dur; if (ease) ease.value = s.ease; }
  // aim pad: place the dot from (yaw, pitch)
  const pad = document.getElementById('aim-pad'), dot = document.getElementById('aim-dot');
  if (pad && dot){
    const xlo = +pad.dataset.xlo, xhi = +pad.dataset.xhi, ylo = +pad.dataset.ylo, yhi = +pad.dataset.yhi;
    const xv = s ? (s.pose[pad.dataset.x] || 0) : 0, yv = s ? (s.pose[pad.dataset.y] || 0) : 0;
    dot.style.left = ((xv - xlo) / (xhi - xlo) * 100) + '%';
    dot.style.top = ((yhi - yv) / (yhi - ylo) * 100) + '%';
    pad.style.opacity = s ? '1' : '0.45';
  }
}
function renderTimeline(){
  const el = document.getElementById('author-timeline'); if (!el) return;
  const segs = editSegs();
  if (!segs.length){ el.innerHTML = '<span style="opacity:.5;font-size:12px">No keyframes yet — pose the robot, then ＋ Keyframe</span>'; return; }
  el.innerHTML = '';
  segs.forEach((s, i) => {
    const chip = document.createElement('div');
    chip.style.cssText = 'display:flex;align-items:center;gap:7px;padding:6px 10px;border-radius:8px;cursor:pointer;font-size:12px;user-select:none;' +
      (i === V.edit.sel ? 'background:#2563eb;color:#fff' : 'background:rgba(255,255,255,0.09)');
    chip.innerHTML = `<b>${i + 1}</b><span style="opacity:.75">${(+s.dur).toFixed(2)}s</span>` +
                     `<span class="kf-del" data-i="${i}" style="opacity:.65;padding:0 2px;font-weight:700">✕</span>`;
    chip.addEventListener('click', ev => {
      if (ev.target.classList.contains('kf-del')){ ev.stopPropagation(); window.ReachyViewer.deleteKeyframe(i); return; }
      window.ReachyViewer.selectKeyframe(i);
    });
    el.appendChild(chip);
  });
}
function setCh(ch, v){ const s = selSeg(); if (!s) return; s.pose[ch] = clampCh(ch, v); commitEdit(); syncPosePanel(); }

// ---- grab a body part in 3D and drag the WHOLE part (with a highlight) ----
// Group every mesh by which URDF joint subtree it lives in (a mesh belongs to exactly one link, so
// no head/body overlap): under left/right_antenna -> that antenna; under a stewart/head joint ->
// head; everything else (torso under yaw_body, base) -> body. Each mesh caches userData.part, so
// picking = the hit mesh's tag and highlighting = the whole group.
function indexParts(){
  V.parts = { head: [], antennaL: [], antennaR: [], body: [] };
  if (!V.robot) return false;
  V.robot.traverse(o => {
    if (!o.isMesh) return;
    let part = 'body', n = o;
    while (n){
      const nm = (n.name || '').toLowerCase();
      if (n.isURDFJoint && nm === 'left_antenna'){ part = 'antennaL'; break; }
      if (n.isURDFJoint && nm === 'right_antenna'){ part = 'antennaR'; break; }
      if (n.isURDFJoint && (nm.includes('stewart') || nm.includes('head'))){ part = 'head'; break; }
      n = n.parent;
    }
    o.userData.part = part;
    V.parts[part].push(o);
  });
  return (V.parts.head.length + V.parts.body.length) > 0;
}
function clearHighlight(){ if (V._hl){ V._hl.forEach(({ m, mat }) => { m.material = mat; }); V._hl = null; } }
function highlightPart(part){
  clearHighlight();
  V._hl = ((V.parts && V.parts[part]) || []).map(m => {
    const mat = m.material, hm = mat.clone();
    hm.emissive = new THREE.Color(0x3b82f6); hm.emissiveIntensity = 0.5; m.material = hm;
    return { m, mat };
  });
}
// the antenna's screen-space bounding rect (its world bbox projected) — covers the whole stalk,
// which is long and thin, so a point+radius wouldn't (the joint origin sits at the base).
function antennaScreenRect(part){
  const meshes = (V.parts && V.parts[part]) || []; if (!meshes.length || !V.cam3d || !V.r) return null;
  const box = new THREE.Box3(); meshes.forEach(m => box.expandByObject(m));
  if (box.isEmpty()) return null;
  const rect = V.r.domElement.getBoundingClientRect(), v = new THREE.Vector3();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, any = false;
  for (let i = 0; i < 8; i++){
    v.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).project(V.cam3d);
    if (v.z > 1) continue;
    const x = rect.left + (v.x * 0.5 + 0.5) * rect.width, y = rect.top + (-v.y * 0.5 + 0.5) * rect.height;
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); any = true;
  }
  return any ? { x0, y0, x1, y1 } : null;
}
function rectDist(r, x, y){ const dx = Math.max(r.x0 - x, 0, x - r.x1), dy = Math.max(r.y0 - y, 0, y - r.y1); return Math.hypot(dx, dy); }
// returns { part, mesh } for the pixel (part: head | body | antennaL | antennaR)
const ANT_GRAB_PX = 18;   // margin around the antenna's screen rect, so the thin stalk is easy to grab
function pickPart(clientX, clientY){
  const miss = { part: 'head', mesh: null };
  if (!V.robot || !V.cam3d || !V.r) return miss;
  if (!V.parts || !(V.parts.head.length + V.parts.body.length)) indexParts();
  const el = V.r.domElement, rect = el.getBoundingClientRect();
  const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1,
                                -((clientY - rect.top) / rect.height) * 2 + 1);
  const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, V.cam3d);
  const hits = rc.intersectObject(V.robot, true);
  let part = null, mesh = null;
  if (hits.length){ let o = hits[0].object; mesh = o; while (o){ if (o.userData && o.userData.part){ part = o.userData.part; break; } o = o.parent; } }
  // antennas are 2px thin — grab one if the click is within ANT_GRAB_PX of its screen rect
  if (part !== 'antennaL' && part !== 'antennaR'){
    let best = null;
    for (const ap of ['antennaL', 'antennaR']){
      const r = antennaScreenRect(ap); if (!r) continue;
      const d = rectDist(r, clientX, clientY);
      if (d < ANT_GRAB_PX && (!best || d < best.d)) best = { part: ap, d };
    }
    if (best) return { part: best.part, mesh };
  }
  if (part) return { part, mesh };
  return { part: 'head', mesh: hits.length ? hits[0].object : null };
}
// channel a grabbed antenna controls: the 'left_antenna' joint is driven by antR, 'right_antenna' by
// antL (see updateJoints) — so grabbing meshes under left_antenna writes antR, moving THAT antenna.
const PART_CH = { antennaL: 'antR', antennaR: 'antL' };
// apply a pixel drag to whatever part is grabbed
function dragPart(part, dx, dy){
  const s = selSeg(); if (!s) return;
  if (part === 'head'){
    s.pose.yaw = clampCh('yaw', (s.pose.yaw || 0) + dx * 0.6);
    s.pose.pitch = clampCh('pitch', (s.pose.pitch || 0) - dy * 0.6);
  } else if (part === 'body'){
    s.pose.body = clampCh('body', (s.pose.body || 0) + dx * 0.7);
  } else {                                   // an antenna: vertical drag = raise/lower (360 range)
    const ch = PART_CH[part]; s.pose[ch] = clampCh(ch, (s.pose[ch] || 0) - dy * 1.5);
  }
  commitEdit(); syncPosePanel();
}
function setPoseMode(on){
  V.edit.posing = on;
  const pm = document.getElementById('author-posemode'); if (pm) pm.textContent = on ? '✋ Pose' : '🔄 Orbit';
  if (V.ctrl) V.ctrl.enabled = !(on && authorVisible());   // posing disables camera orbit
}
function setupAuthor(){
  document.querySelectorAll('#author-pose .apose').forEach(inp => {
    inp.addEventListener('input', () => {
      const s = selSeg(); if (!s) return; const ch = inp.dataset.ch;
      s.pose[ch] = clampCh(ch, parseFloat(inp.value) || 0);
      const lbl = document.querySelector(`#author-pose .aval[data-for="${ch}"]`); if (lbl) lbl.textContent = Math.round(s.pose[ch]);
      commitEdit();
    });
  });
  const dur = document.getElementById('author-dur');
  if (dur) dur.addEventListener('change', () => { const s = selSeg(); if (s){ s.dur = Math.max(0.05, parseFloat(dur.value) || 0.4); commitEdit(); renderTimeline(); } });
  const ease = document.getElementById('author-ease');
  if (ease) ease.addEventListener('change', () => { const s = selSeg(); if (s){ s.ease = ease.value; commitEdit(); } });
  const add = document.getElementById('author-addkf'); if (add) add.addEventListener('click', () => window.ReachyViewer.addKeyframe());
  const pm = document.getElementById('author-posemode'); if (pm) pm.addEventListener('click', () => setPoseMode(!V.edit.posing));
  // aim pad: drag the dot -> (yaw, pitch) of the selected keyframe (thumbstick style)
  const pad = document.getElementById('aim-pad');
  if (pad){
    let dragging = false;
    const apply = (cx, cy) => {
      const s = selSeg(); if (!s) return;
      const r = pad.getBoundingClientRect();
      const fx = Math.max(0, Math.min(1, (cx - r.left) / r.width));
      const fy = Math.max(0, Math.min(1, (cy - r.top) / r.height));
      const xlo = +pad.dataset.xlo, xhi = +pad.dataset.xhi, ylo = +pad.dataset.ylo, yhi = +pad.dataset.yhi;
      s.pose[pad.dataset.x] = clampCh(pad.dataset.x, xlo + fx * (xhi - xlo));   // turn (yaw)
      s.pose[pad.dataset.y] = clampCh(pad.dataset.y, yhi - fy * (yhi - ylo));   // nod (pitch); top = max
      commitEdit(); syncPosePanel();
    };
    pad.addEventListener('pointerdown', e => { if (!selSeg()) return; dragging = true; try { pad.setPointerCapture(e.pointerId); } catch (_) {} apply(e.clientX, e.clientY); });
    pad.addEventListener('pointermove', e => { if (dragging) apply(e.clientX, e.clientY); });
    const padEnd = () => { dragging = false; };
    pad.addEventListener('pointerup', padEnd); pad.addEventListener('pointercancel', padEnd);
  }
  // grab a part in 3D and drag it (left button / single touch); 2nd pointer cancels -> orbit
  const canvas = V.r && V.r.domElement;
  if (canvas){
    let active = false, lx = 0, ly = 0, pts = 0, part = null;
    canvas.addEventListener('pointerdown', e => {
      pts++; if (!editMode() || pts > 1 || e.button !== 0){ active = false; clearHighlight(); return; }
      part = pickPart(e.clientX, e.clientY).part; highlightPart(part);
      active = true; lx = e.clientX; ly = e.clientY; try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    });
    canvas.addEventListener('pointermove', e => {
      if (active){ const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY; dragPart(part, dx, dy); return; }
      if (editMode()){ const p = pickPart(e.clientX, e.clientY).part; if (p !== V.hoverPart){ V.hoverPart = p; highlightPart(p); } }   // hover preview
    });
    const end = () => { pts = Math.max(0, pts - 1); if (active){ active = false; clearHighlight(); V.hoverPart = null; rebakeEdit(); } };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', () => { if (!active){ clearHighlight(); V.hoverPart = null; } });
  }
  setPoseMode(V.edit.posing);
  renderTimeline(); syncPosePanel();
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

// tell the backend (via a hidden button) when the Voice loop should run — only on change,
// so the backend flag is stable (the gradio Timer can't carry a per-tick js value reliably).
let _voiceWantLast = null;
function voiceTick(){
  const w = window.ReachyViewer.voiceWanted();
  if (w !== _voiceWantLast){
    _voiceWantLast = w;
    const el = document.getElementById('voice-set-btn');
    const b = el && (el.tagName === 'BUTTON' ? el : el.querySelector('button'));
    if (b) b.click();
  }
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
    Object.assign(V, {scene, cam3d: cam, r, ctrl});   // cam3d: the THREE camera (V.cam is the WebRTC cam!)
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
        mesh.userData.isAntenna = /antenna/i.test(filename);   // tag for grab-in-3D part picking
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
        const au = V.authorEl || (V.authorEl = document.getElementById('reachy-author'));
        // visible = laid out AND not inside a collapsed top pane (.pane-hidden keeps width but
        // hides via height:0, so offsetParent alone stays truthy — check the pane too).
        const vis = el => el && el.offsetParent && !el.closest('.pane-hidden');
        const playActive = !!(vis(mp) || vis(au));   // Animate sub-tab, or the Author pane
        const authoring = vis(au) && V.edit.sel >= 0;
        if (authoring){
          // the selected keyframe is the single source of truth: the gamepad is folded into it by
          // joyTick (joyToKeyframe), manual edits write it too — so just render it, never driveSim.
          if (V.traj) applyFrame();
        } else if (V.joy.on && V.joy.phase === 'control'){
          driveSim(V.joy.tgt);                      // sim free-control: IK -> full rig, every frame
        } else if (V.traj && !V.joy.on && playActive){
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
    setInterval(voiceTick, 600); // start/stop the voice loop with the Voice accordion + connection
    setupAuthor();   // wire the pose panel, timeline, and pose-drag (Phase-1 authoring)
  },

  playMove(traj){
    if (typeof traj === 'string'){ traj = traj ? JSON.parse(traj) : null; }
    // Authored behaviors arrive as a SPEC (eased segments + layers) — bake on the client.
    // Recorded library moves already arrive baked (have .time) and pass straight through.
    if (traj && traj.segments !== undefined){
      traj = traj.segments.length ? bakeSpec(traj) : null;
    }
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

  // ▶ Preview button: restart the loaded animation from the top. The live edit/select preview
  // keeps V.traj current, so Preview just rewinds + plays (re-emitting the same spec to the
  // hidden traj box wouldn't fire its change event, so we drive playback directly here).
  replay(){
    if (!V.traj){ showReady(); return; }
    V.t = 0; V.playing = true;
    const pb = document.getElementById('reachy-play'); if (pb) pb.textContent = '⏸ Pause';
    if (V.audio && V.audio.src){ try { V.audio.currentTime = 0; V.audio.play().catch(()=>{}); } catch(e){} }
  },

  // refit the renderer + camera to #reachy3d's current size (it changes when the viewer is
  // relocated between the wide Play layout and the smaller Author preview, see relocate()).
  resize(){
    const el = document.getElementById('reachy3d');
    if (!el || !V.r || !V.cam3d) return;
    const w = el.clientWidth, h = el.clientHeight;
    if (w < 2 || h < 2) return;   // hidden tab -> 0 size; skip until it's visible
    V.r.setSize(w, h); V.cam3d.aspect = w/h; V.cam3d.updateProjectionMatrix();
    if (V.traj && !V.live) buildChart(V.traj);   // chart re-fits to its (also relocated) container
  },

  // ---- Phase-1 authoring API (called from the Gradio author pane) ----
  loadSpec(json){                       // load a behavior spec into the editor (from the dropdown)
    let spec; try { spec = typeof json === 'string' ? (json ? JSON.parse(json) : null) : json; } catch (_) { spec = null; }
    if (!spec || !Array.isArray(spec.segments)) spec = { fps:40, segments:[], layers:[] };
    spec.segments = spec.segments.map(s => ({ dur:+s.dur || 0.4, ease:s.ease || 'smooth',
      pose: Object.assign(zerosPose(), s.pose || {}) }));
    spec.layers = spec.layers || [];
    V.edit.spec = spec;
    V.edit.sel = spec.segments.length ? 0 : -1;
    rebakeEdit(); parkOnSelected(); renderTimeline(); syncPosePanel();
  },
  newSpec(){                            // start a fresh behavior with one editable keyframe
    V.edit.spec = { fps:40, segments:[{ dur:0.4, ease:'ease_out', pose:zerosPose() }], layers:[] };
    V.edit.sel = 0; rebakeEdit(); parkOnSelected(); renderTimeline(); syncPosePanel();
  },
  getEditSpec(){ return JSON.stringify(V.edit.spec); },
  setLayers(list){ V.edit.spec.layers = (list || []).map(t => ({ type:t })); commitEdit(); },
  addKeyframe(){
    const prev = selSeg() || editSegs()[editSegs().length - 1];
    const pose = prev ? Object.assign({}, prev.pose) : zerosPose();
    const dur = parseFloat((document.getElementById('author-dur') || {}).value) || 0.4;
    const ease = (document.getElementById('author-ease') || {}).value || 'smooth';
    editSegs().push({ dur, ease, pose });
    V.edit.sel = editSegs().length - 1;
    rebakeEdit(); parkOnSelected(); renderTimeline(); syncPosePanel();
  },
  selectKeyframe(i){ V.edit.sel = i; rebakeEdit(); parkOnSelected(); renderTimeline(); syncPosePanel(); },
  deleteKeyframe(i){
    editSegs().splice(i, 1);
    if (V.edit.sel >= editSegs().length) V.edit.sel = editSegs().length - 1;
    rebakeEdit(); parkOnSelected(); renderTimeline(); syncPosePanel();
  },

  // There is ONE 3D viewer + channels chart. Move those DOM blocks into the active top tab's slot
  // (Play's home spots, or the Author tab's preview/channels slots) so each tab gets its own layout
  // without a second WebGL context. The canvas survives the move; resize() refits afterward.
  relocate(toAuthor){
    const vb = document.getElementById('viewer-block'), cb = document.getElementById('chart-block');
    const vSlot = document.getElementById(toAuthor ? 'author-viewer-slot' : 'play-viewer-home');
    const cSlot = document.getElementById(toAuthor ? 'author-chart-slot' : 'play-chart-home');
    if (vb && vSlot && vb.parentElement !== vSlot) vSlot.appendChild(vb);
    if (cb && cSlot && cb.parentElement !== cSlot) cSlot.appendChild(cb);
    // posing disables camera orbit (Author only); Play always orbits
    if (V.ctrl) V.ctrl.enabled = toAuthor ? !V.edit.posing : true;
    // frame the robot: Author starts zoomed out (more room to pose); Play sits closer
    if (V.cam3d && V.ctrl){
      const t = V.ctrl.target, f = toAuthor ? 1.7 : 1.0;   // play offset from target = (0.34,0.15,0.40)
      V.cam3d.position.set(t.x + 0.34 * f, t.y + 0.15 * f, t.z + 0.40 * f);
      V.ctrl.update();
    }
    if (toAuthor){ parkOnSelected(); syncPosePanel(); renderTimeline(); }
    setTimeout(() => window.ReachyViewer.resize(), 50);   // let the new layout settle, then refit
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
    if (dot) dot.style.background = d.speaking ? '#fb923c' : (d.active ? '#22c55e' : '#3b4252');
    if (label) label.textContent = d.speaking ? 'Reachy speaking…' : (d.active ? 'voice detected' : 'listening…');
    // conversation: You / Reachy lines + the in-progress utterance (grey italic)
    const tr = document.getElementById('reachy-transcript');
    if (tr){
      const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
      let html = (d.dialogue || []).slice(-4).map(m => {
        const you = m.role === 'user';
        const style = you ? 'opacity:0.9' : 'color:#fb923c';
        return '<div style="' + style + '"><b>' + (you ? 'You' : 'Reachy') + ':</b> ' + esc(m.text) + '</div>';
      }).join('');
      if (d.interim) html += '<div style="opacity:0.5;font-style:italic"><b>You:</b> ' + esc(d.interim) + '</div>';
      if (!html) html = '<div style="opacity:0.4">' + (d.stt_ready === false ? 'loading speech model…' : 'say something…') + '</div>';
      tr.innerHTML = html;
    }
  },

  // the voice loop (mic/VAD/STT/TTS) should run only when connected AND the Voice
  // accordion is expanded (its content visible) — mirrors how the camera gates itself
  voiceWanted(){
    const el = document.getElementById('reachy-voice');
    return !!(V.live && el && el.offsetParent !== null);
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
