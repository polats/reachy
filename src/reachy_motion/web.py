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

    head_flat, head_joints, antennas, body = [], [], [], []
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
        antennas.append([float(traj[i]["antennas"][0]), float(traj[i]["antennas"][1])])
        body.append(by)
    return {
        "time": [float(x) for x in grid],
        "head": head_flat,
        "head_joints": head_joints,
        "antennas": antennas,
        "body_yaw": body,
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
            calcPassive:null, buildHeadPose:null, audio:null, live:false, ws:null };

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
  },

  playMove(traj){
    if (typeof traj === 'string'){ if (!traj) return; traj = JSON.parse(traj); }
    if (!traj || !traj.time) return;
    V.traj = traj; V.dur = traj.time[traj.time.length-1]; V.t = 0; V.playing = true;
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
    ws.onopen = () => { V.live = true; setStatus('● live — mirroring robot', '#4cae4c');
      const b=document.getElementById('reachy-connect'); if(b) b.textContent='Disconnect robot';
      if (V.audio) V.audio.pause(); };
    ws.onmessage = (ev) => { if (!V.ready) return;
      try { updateJoints(JSON.parse(ev.data)); } catch(e){} };
    ws.onerror = () => setStatus('connection error (is the daemon running on :8000?)', '#d9534f');
    ws.onclose = () => { V.live = false; V.ws = null;
      const b=document.getElementById('reachy-connect'); if(b) b.textContent='Connect robot';
      setStatus('disconnected', '#888'); };
  },
  disconnectRobot(){ if (V.ws){ V.ws.close(); V.ws = null; } V.live = false; },
  toggleRobot(){ if (V.live || V.ws) this.disconnectRobot(); else this.connectRobot(); },
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
