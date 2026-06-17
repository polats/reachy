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
            calcPassive:null, buildHeadPose:null, audio:null, live:false, ws:null,
            liveBuf:[], liveWindow:8, liveLast:0,
            joy:{ on:false, ws:null, raf:0, last:0, t:0, warned:false, tgt:null } };

// gamepad teleop tuning (ranges/speeds mirror the desktop app)
const JOY = { dead:0.12, posSpeed:0.06, rotSpeed:0.9, zSpeed:0.04, bodySpeed:0.9, antSpeed:2.5,
  lim:{ pos:0.05, z:0.05, pitch:0.8, yaw:1.2, roll:0.5, body:1.0, ant:2.79 } };
const _clamp = (v,a) => Math.max(-a, Math.min(a, v));
const _dz = v => Math.abs(v) < JOY.dead ? 0 : v;

function joyLoop(){
  if (!V.joy.on) return;
  V.joy.raf = requestAnimationFrame(joyLoop);
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gp = null; for (const p of pads){ if (p){ gp = p; break; } }
  const now = performance.now()/1000; const dt = Math.min(0.05, now-(V.joy.t||now)); V.joy.t = now;
  if (!gp){ if (!V.joy.warned){ setStatus('🎮 no gamepad — connect one and press a button', '#d9a441'); V.joy.warned=true; } return; }
  if (V.joy.warned){ V.joy.warned=false; setStatus('🎮 Joystick — L stick move · R stick look · bumpers turn · triggers height', '#4cae4c'); }
  const ax = gp.axes, bt = gp.buttons, tg = V.joy.tgt, L = JOY.lim;
  const lx=_dz(ax[0]||0), ly=_dz(ax[1]||0), rx=_dz(ax[2]||0), ry=_dz(ax[3]||0);
  tg.x     = _clamp(tg.x     + (-ly)*JOY.posSpeed*dt, L.pos);
  tg.y     = _clamp(tg.y     + ( lx)*JOY.posSpeed*dt, L.pos);
  tg.pitch = _clamp(tg.pitch + (-ry)*JOY.rotSpeed*dt, L.pitch);
  tg.yaw   = _clamp(tg.yaw   + ( rx)*JOY.rotSpeed*dt, L.yaw);
  const lb=(bt[4]&&bt[4].value)||0, rb=(bt[5]&&bt[5].value)||0;   // bumpers -> body yaw
  tg.by    = _clamp(tg.by + (rb-lb)*JOY.bodySpeed*dt, L.body);
  const lt=(bt[6]&&bt[6].value)||0, rt=(bt[7]&&bt[7].value)||0;   // triggers -> height (z)
  tg.z     = _clamp(tg.z + (rt-lt)*JOY.zSpeed*dt, L.z);
  const aUp=(bt[3]&&bt[3].pressed)?1:0, aDn=(bt[0]&&bt[0].pressed)?1:0;  // Y/A -> antennas
  const ad=(aUp-aDn)*JOY.antSpeed*dt; tg.aL=_clamp(tg.aL+ad,L.ant); tg.aR=_clamp(tg.aR+ad,L.ant);
  if (now - V.joy.last > 0.04 && V.joy.ws && V.joy.ws.readyState === 1){
    V.joy.last = now;
    V.joy.ws.send(JSON.stringify({
      target_head_pose:{x:tg.x,y:tg.y,z:tg.z,roll:tg.roll,pitch:tg.pitch,yaw:tg.yaw},
      target_antennas:[tg.aL,tg.aR], target_body_yaw:tg.by }));
  }
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
    scene.add(new THREE.GridHelper(1.0, 20, 0x2c3a57, 0x1c2740));
    Object.assign(V, {scene, cam, r, ctrl});
    V.audio = document.getElementById('reachy-audio');

    // kinematics (their passive-joint solver)
    try { const K = await import(kinUrl); V.calcPassive = K.calculatePassiveJoints; V.buildHeadPose = K.buildHeadPoseMatrix; }
    catch(e){ console.error('Kinematics import failed', e); }

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
      if (V.traj && V.traj.audio && V.audio){ V.audio.src = V.traj.audio; V.audio.play().catch(()=>{}); }
    }, undefined, (e)=> console.error('URDF load error', e));

    const clock = new THREE.Clock();
    const loop = () => {
      requestAnimationFrame(loop);
      const dt = clock.getDelta();
      if (V.ready && V.traj && !V.live){   // recorded playback (paused while mirroring live)
        if (V.audio && V.audio.src && !V.audio.paused) V.t = Math.min(V.audio.currentTime, V.dur);
        else if (V.playing){ V.t += dt; if (V.t > V.dur) V.t = 0; }
        applyFrame();
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
  },

  playMove(traj){
    if (typeof traj === 'string'){ if (!traj) return; traj = JSON.parse(traj); }
    if (!traj || !traj.time) return;
    V.traj = traj; V.dur = traj.time[traj.time.length-1]; V.t = 0; V.playing = true;
    if (!V.live) buildChart(traj);   // while live, the rolling scope owns the chart
    if (V.audio){
      if (traj.audio){ V.audio.src = traj.audio; V.audio.currentTime = 0; V.audio.play().catch(()=>{}); }
      else { V.audio.removeAttribute('src'); V.audio.load(); }
    }
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

  setJoystick(on){
    if (on){
      if (V.joy.on) return;
      const s = V.liveBuf[V.liveBuf.length-1], R = Math.PI/180;
      V.joy.tgt = s
        ? { x:s.x/1000, y:s.y/1000, z:s.z/1000, roll:s.roll*R, pitch:s.pitch*R, yaw:s.yaw*R,
            aL:s.antL*R, aR:s.antR*R, by:s.body*R }
        : { x:0,y:0,z:0,roll:0,pitch:0,yaw:0,aL:0,aR:0,by:0 };
      const host = location.hostname || '127.0.0.1';
      try { V.joy.ws = new WebSocket(`ws://${host}:8000/api/move/ws/set_target`); }
      catch(e){ setStatus('🎮 joystick connection failed', '#d9534f'); return; }
      V.joy.on = true; V.joy.warned = false; V.joy.t = 0;
      setStatus('🎮 Joystick — L stick move · R stick look · bumpers turn · triggers height', '#4cae4c');
      V.joy.raf = requestAnimationFrame(joyLoop);
    } else {
      V.joy.on = false;
      if (V.joy.raf) cancelAnimationFrame(V.joy.raf);
      if (V.joy.ws){ try { V.joy.ws.close(); } catch(e){} V.joy.ws = null; }
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

CHART_HTML = """
<div style="display:flex;flex-direction:column;gap:4px">
  <div style="font-size:13px;font-weight:600;opacity:0.85">Channels (head pose · antennas · body)</div>
  <div id="reachy-chart" style="width:100%;height:330px"></div>
  <div style="font-size:12px;opacity:0.6">white line = playhead · click or drag on the chart to scrub</div>
</div>
"""
