// Reachy sound lab — client-side Web Audio synth + DAW-style rack (the JS twin of sound.py).
// Synthesizes in the browser so every tweak is instant (fixes the stale gr.Audio preview), and
// gives proper controls: a draggable pitch-CONTOUR editor (the melody), rotary KNOBS for the voice,
// an XY pad for the formants, a waveform, and transport. window.SoundLab.getSpec() hands the same
// spec to the server for "On robot" / Save.

const SL = { ctx: null, spec: null, src: null, playing: false, loop: false, sel: 0 };

const S_EASES = {
  linear: u => u, smooth: u => u * u * u * (u * (u * 6 - 15) + 10),
  ease_out: u => 1 - Math.pow(1 - u, 3), ease_in: u => u * u * u,
  back: u => { const c = 1.70158; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); },
  anticipate: u => { const c = 1.70158; return u * u * ((c + 1) * u - c); }, hold: u => u,
};
const F_LO = 80, F_HI = 1000;   // contour pitch range (Hz)

// ---- synth: port of sound.py bake_spec -> Float32Array ----
function biquadPeak(x, sr, fc, q) {
  const w0 = 2 * Math.PI * Math.min(fc, sr * 0.45) / sr, A = 3.2, al = Math.sin(w0) / (2 * q);
  const a0 = 1 + al / A, B0 = (1 + al * A) / a0, B1 = (-2 * Math.cos(w0)) / a0, B2 = (1 - al * A) / a0,
        A1 = (-2 * Math.cos(w0)) / a0, A2 = (1 - al / A) / a0;
  const y = new Float32Array(x.length); let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) { const yi = B0 * x[i] + B1 * x1 + B2 * x2 - A1 * y1 - A2 * y2; x2 = x1; x1 = x[i]; y2 = y1; y1 = yi; y[i] = yi; }
  return y;
}
function bakeSpec(spec) {
  const sr = spec.sr || 44100, v = spec.voice || {}, harm = v.harmonics || 8, tilt = v.tilt || 1,
        forms = v.formants || [760, 1500], fq = v.formant_q || 5, segs = spec.segments || [];
  if (!segs.length) return { sr, data: new Float32Array(1) };
  const f0 = [], amp = [], vr = [], vd = [], ra = []; let pf = segs[0].f0, pa = 0;
  for (const s of segs) {
    const n = Math.max(1, Math.round(s.dur * sr)), e = S_EASES[s.ease] || S_EASES.smooth;
    for (let i = 0; i < n; i++) { const u = e(i / n); f0.push(pf + (s.f0 - pf) * u); amp.push(pa + (s.amp - pa) * u); vr.push(s.vib_rate || 0); vd.push(s.vib_depth || 0); ra.push(s.rasp || 0); }
    pf = s.f0; pa = s.amp;
  }
  const N = f0.length, sig = new Float32Array(N); let phase = 0, mx = 0;
  for (let i = 0; i < N; i++) {
    const fm = f0[i] * (1 + vd[i] * Math.sin(2 * Math.PI * vr[i] * i / sr)); phase += 2 * Math.PI * fm / sr;
    let s = 0; for (let k = 1; k <= harm; k++) s += (1 / Math.pow(k, tilt)) * Math.sin(k * phase); sig[i] = s; if (Math.abs(s) > mx) mx = Math.abs(s);
  }
  for (let i = 0; i < N; i++) sig[i] /= (mx || 1);
  if (forms.length) { const fs = new Float32Array(N); for (const fc of forms) { const fb = biquadPeak(sig, sr, fc, fq); for (let i = 0; i < N; i++) fs[i] += fb[i] / forms.length; } for (let i = 0; i < N; i++) sig[i] = 0.5 * sig[i] + 0.9 * fs[i]; }
  let maxra = 0; for (let i = 0; i < N; i++) if (ra[i] > maxra) maxra = ra[i];
  if (maxra > 0) { const noise = new Float32Array(N); for (let i = 0; i < N; i++) noise[i] = Math.random() * 2 - 1; const nb = biquadPeak(noise, sr, 2600, 1.2); for (let i = 0; i < N; i++) sig[i] += ra[i] * 0.5 * nb[i]; }
  const a = Math.floor(0.006 * sr); for (let i = 0; i < N; i++) sig[i] *= amp[i];
  if (N > 2 * a) for (let i = 0; i < a; i++) { sig[i] *= i / a; sig[N - 1 - i] *= i / a; }
  mx = 0; for (let i = 0; i < N; i++) if (Math.abs(sig[i]) > mx) mx = Math.abs(sig[i]);
  for (let i = 0; i < N; i++) sig[i] = 0.9 * sig[i] / (mx || 1);
  return { sr, data: sig };
}

// ---- transport ----
function ctxOf() { if (!SL.ctx) SL.ctx = new (window.AudioContext || window.webkitAudioContext)(); return SL.ctx; }
function play() {
  stop(); if (!SL.spec) return;
  const { sr, data } = bakeSpec(SL.spec), ctx = ctxOf(); ctx.resume();
  const buf = ctx.createBuffer(1, data.length, sr); buf.copyToChannel(data, 0);
  const src = ctx.createBufferSource(); src.buffer = buf; src.loop = SL.loop; src.connect(ctx.destination); src.start();
  src.onended = () => { if (SL.src === src && !SL.loop) { SL.playing = false; transport(); } };
  SL.src = src; SL.playing = true; drawWave(data); transport();
}
function stop() { if (SL.src) { try { SL.src.stop(); } catch (_) {} SL.src = null; } SL.playing = false; transport(); }
function transport() {
  const p = document.getElementById('sl-play'), l = document.getElementById('sl-loop');
  if (p) p.textContent = SL.playing ? '⏸ Stop' : '▶ Play';
  if (l) l.style.background = SL.loop ? '#2563eb' : 'rgba(255,255,255,0.08)';
}

// ---- waveform ----
function drawWave(data) {
  const el = document.getElementById('sl-wave'); if (!el) return;
  const w = el.clientWidth || 600, h = 70, N = Math.min(w, 900), step = Math.max(1, Math.floor(data.length / N));
  let bars = '';
  for (let i = 0; i < N; i++) { let mx = 0; for (let j = 0; j < step; j++) { const a = Math.abs(data[i * step + j] || 0); if (a > mx) mx = a; } const bh = Math.max(1, mx * h * 0.92), x = (i / N) * w; bars += `<rect x="${x.toFixed(1)}" y="${((h - bh) / 2).toFixed(1)}" width="${(w / N * 0.85).toFixed(2)}" height="${bh.toFixed(1)}" fill="#60a5fa" opacity="0.75"/>`; }
  el.innerHTML = `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block"><rect width="${w}" height="${h}" fill="rgba(255,255,255,0.03)"/>${bars}</svg>`;
}

// ---- pitch-contour editor (the melody) ----
function cumTimes() { const t = [0]; for (const s of SL.spec.segments) t.push(t[t.length - 1] + s.dur); return t; }
function renderContour() {
  const el = document.getElementById('sl-contour'); if (!el || !SL.spec) return;
  const w = el.clientWidth || 600, h = 150, pad = 8;
  const segs = SL.spec.segments, tt = cumTimes(), total = tt[tt.length - 1] || 1;
  const X = t => pad + (t / total) * (w - 2 * pad);
  const Y = f => h - pad - ((Math.max(F_LO, Math.min(F_HI, f)) - F_LO) / (F_HI - F_LO)) * (h - 2 * pad);
  // gridlines at a few pitches
  let g = '';
  for (const f of [200, 400, 600, 800]) g += `<line x1="${pad}" y1="${Y(f)}" x2="${w - pad}" y2="${Y(f)}" stroke="rgba(255,255,255,0.07)"/><text x="2" y="${Y(f) + 3}" fill="rgba(255,255,255,0.35)" font-size="9">${f}</text>`;
  // eased polyline (sample the baked f0 path for accuracy)
  let pts = `${X(0)},${Y(segs[0].f0)}`; let pf = segs[0].f0;
  segs.forEach((s, i) => { const e = S_EASES[s.ease] || S_EASES.smooth; for (let k = 1; k <= 8; k++) { const u = e(k / 8), f = pf + (s.f0 - pf) * u; pts += ` ${X(tt[i] + s.dur * k / 8).toFixed(1)},${Y(f).toFixed(1)}`; } pf = s.f0; });
  // draggable keyframe handles
  let dots = '';
  segs.forEach((s, i) => { const sel = i === SL.sel; dots += `<circle class="sl-pt" data-i="${i}" cx="${X(tt[i + 1])}" cy="${Y(s.f0)}" r="${sel ? 7 : 5}" fill="${sel ? '#2563eb' : '#60a5fa'}" stroke="#fff" stroke-width="1.5" style="cursor:grab"/>`; });
  el.innerHTML = `<svg width="100%" viewBox="0 0 ${w} ${h}" style="display:block;touch-action:none">${g}<polyline points="${pts}" fill="none" stroke="#3b82f6" stroke-width="2"/>${dots}</svg>`;
}
function contourFromXY(cx, cy) {   // screen px -> {time, f0} in spec space
  const el = document.getElementById('sl-contour'), r = el.getBoundingClientRect();
  const w = r.width, h = 150 * (r.width / (el.clientWidth || r.width)) / (r.width / 150) || 150; // viewBox h=150 scaled
  const pad = 8, total = cumTimes().slice(-1)[0] || 1;
  const fx = (cx - r.left) / r.width, fy = (cy - r.top) / r.height;
  const time = Math.max(0, Math.min(1, (fx * (1) - pad / r.width))) * total;   // approx
  const f0 = F_LO + (1 - Math.max(0, Math.min(1, fy))) * (F_HI - F_LO);
  return { time, f0 };
}

// ---- rotary knob (drag up/down) + XY pad ----
function knobSVG(val, lo, hi) {
  const f = (val - lo) / (hi - lo), a0 = 135, a1 = 405, ang = (a0 + f * (a1 - a0)) * Math.PI / 180, cx = 26, cy = 26, R = 18;
  const arc = (s, e) => { const sa = s * Math.PI / 180, ea = e * Math.PI / 180; const large = (e - s) > 180 ? 1 : 0; return `M ${cx + R * Math.cos(sa)} ${cy + R * Math.sin(sa)} A ${R} ${R} 0 ${large} 1 ${cx + R * Math.cos(ea)} ${cy + R * Math.sin(ea)}`; };
  return `<svg width="52" height="52" viewBox="0 0 52 52"><path d="${arc(135, 405)}" stroke="rgba(255,255,255,0.12)" stroke-width="4" fill="none"/><path d="${arc(135, 135 + f * 270)}" stroke="#3b82f6" stroke-width="4" fill="none"/><line x1="${cx}" y1="${cy}" x2="${cx + (R - 3) * Math.cos(ang)}" y2="${cy + (R - 3) * Math.sin(ang)}" stroke="#fff" stroke-width="2"/></svg>`;
}
const KNOBS = [
  { ch: 'harmonics', label: 'Buzz', lo: 1, hi: 16, step: 1, fmt: v => v | 0 },
  { ch: 'tilt', label: 'Soft', lo: 0.3, hi: 2.5, step: 0.05, fmt: v => v.toFixed(2) },
  { ch: 'formant_q', label: 'Res', lo: 1, hi: 12, step: 0.5, fmt: v => v.toFixed(1) },
  { ch: 'rasp', label: 'Rasp', lo: 0, hi: 0.6, step: 0.02, fmt: v => v.toFixed(2) },
  { ch: 'vib_rate', label: 'Vib Hz', lo: 0, hi: 14, step: 0.5, fmt: v => v.toFixed(1) },
  { ch: 'vib_depth', label: 'Vib', lo: 0, hi: 0.08, step: 0.002, fmt: v => v.toFixed(3) },
];
function renderKnobs() {
  const el = document.getElementById('sl-knobs'); if (!el || !SL.spec) return;
  el.innerHTML = KNOBS.map(k => { const v = SL.spec.voice[k.ch] ?? k.lo; return `<div class="sl-knob" data-ch="${k.ch}" style="display:flex;flex-direction:column;align-items:center;gap:1px;cursor:ns-resize;user-select:none">${knobSVG(v, k.lo, k.hi)}<span style="font-size:10px;opacity:.7">${k.label}</span><span class="sl-kv" style="font-size:10px;font-variant-numeric:tabular-nums">${k.fmt(v)}</span></div>`; }).join('');
}
function setVoice(ch, val) {
  const k = KNOBS.find(x => x.ch === ch); if (k) val = Math.max(k.lo, Math.min(k.hi, val));
  SL.spec.voice[ch] = val;
  if (ch === 'vib_rate' || ch === 'vib_depth' || ch === 'rasp') SL.spec.segments.forEach(s => { if (s.amp > 0) s[ch] = val; });
  renderKnobs(); drawWave(bakeSpec(SL.spec).data);
}
function renderPad() {
  const el = document.getElementById('sl-formants'); if (!el || !SL.spec) return;
  const f = SL.spec.voice.formants || [760, 1500], w = 130, h = 130;
  const fx = (f[0] - 300) / 1200, fy = (f[1] - 800) / 2700;
  el.innerHTML = `<svg width="100%" viewBox="0 0 ${w} ${h}" style="display:block;touch-action:none"><rect width="${w}" height="${h}" rx="10" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.12)"/><circle cx="${fx * w}" cy="${(1 - fy) * h}" r="9" fill="#3b82f6" stroke="#fff" stroke-width="2"/></svg>`;
}

// ---- shared pointer drag ----
function onDrag(el, move) {
  let on = false, lx = 0, ly = 0;
  el.addEventListener('pointerdown', e => { on = true; lx = e.clientX; ly = e.clientY; try { el.setPointerCapture(e.pointerId); } catch (_) {} move(e, 0, 0, true); });
  el.addEventListener('pointermove', e => { if (!on) return; move(e, e.clientX - lx, e.clientY - ly, false); lx = e.clientX; ly = e.clientY; });
  const end = () => { if (on) { on = false; play(); } };   // auto-play on release for instant feedback
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
}

function setup() {
  if (SL._setup || !document.getElementById('sl-contour')) return;   // wire once, only when the DOM exists
  SL._setup = true;
  // transport
  const p = document.getElementById('sl-play'); if (p) p.onclick = () => SL.playing ? stop() : play();
  const l = document.getElementById('sl-loop'); if (l) l.onclick = () => { SL.loop = !SL.loop; if (SL.src) SL.src.loop = SL.loop; transport(); };
  const st = document.getElementById('sl-stop'); if (st) st.onclick = stop;
  // contour: drag points (y=f0, x=time); double-click empty=add; right-click point=remove
  const ce = document.getElementById('sl-contour');
  if (ce) {
    onDrag(ce, (e, dx, dy, down) => {
      const r = ce.getBoundingClientRect();
      if (down) { const t = e.target; SL.sel = t && t.classList.contains('sl-pt') ? +t.dataset.i : SL.sel; renderContour(); if (!(t && t.classList.contains('sl-pt'))) return; }
      const s = SL.spec.segments[SL.sel]; if (!s) return;
      const fy = (e.clientY - r.top) / r.height;
      s.f0 = Math.round(F_LO + (1 - Math.max(0, Math.min(1, fy))) * (F_HI - F_LO));
      renderContour(); drawWave(bakeSpec(SL.spec).data);
    });
    ce.addEventListener('dblclick', e => {
      const r = ce.getBoundingClientRect(), fy = (e.clientY - r.top) / r.height;
      const f0 = Math.round(F_LO + (1 - Math.max(0, Math.min(1, fy))) * (F_HI - F_LO));
      const seg = JSON.parse(JSON.stringify(SL.spec.segments[SL.sel] || { dur: 0.2, ease: 'smooth', amp: 1, vib_rate: 6, vib_depth: 0.015, rasp: 0 }));
      seg.f0 = f0; SL.spec.segments.splice(SL.sel + 1, 0, seg); SL.sel++; renderContour(); play();
    });
    ce.addEventListener('contextmenu', e => { e.preventDefault(); if (SL.spec.segments.length > 1) { SL.spec.segments.splice(SL.sel, 1); SL.sel = Math.min(SL.sel, SL.spec.segments.length - 1); renderContour(); play(); } });
  }
  // knobs: resolve which knob on pointerdown (e.target is the container on later moves), drag ↕
  const kc = document.getElementById('sl-knobs');
  if (kc) onDrag(kc, (e, dx, dy, down) => {
    if (down) { const kn = e.target.closest('.sl-knob'); kc._k = kn ? KNOBS.find(x => x.ch === kn.dataset.ch) : null; return; }
    const kk = kc._k; if (!kk) return;
    const cur = SL.spec.voice[kk.ch] ?? kk.lo;
    setVoice(kk.ch, cur - dy * (kk.hi - kk.lo) / 180);
  });
  // formant XY pad
  const fp = document.getElementById('sl-formants');
  if (fp) onDrag(fp, (e) => { const r = fp.getBoundingClientRect(); const fx = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), fy = Math.max(0, Math.min(1, 1 - (e.clientY - r.top) / r.height)); SL.spec.voice.formants = [Math.round(300 + fx * 1200), Math.round(800 + fy * 2700)]; renderPad(); drawWave(bakeSpec(SL.spec).data); });
}

window.SoundLab = {
  init() { setup(); if (SL.spec) this.render(); },
  loadSpec(json) {
    let spec; try { spec = typeof json === 'string' ? (json ? JSON.parse(json) : null) : json; } catch (_) { spec = null; }
    if (!spec || !spec.segments) return;
    spec.voice = Object.assign({ harmonics: 8, tilt: 1, formants: [760, 1500], formant_q: 5, rasp: 0, vib_rate: 6, vib_depth: 0.015 }, spec.voice || {});
    SL.spec = spec; SL.sel = 0; setup(); this.render(); play();
  },
  getSpec() { return JSON.stringify(SL.spec || {}); },
  render() { renderContour(); renderKnobs(); renderPad(); if (SL.spec) drawWave(bakeSpec(SL.spec).data); transport(); },
};

// Gradio lazy-renders accordion content, so the rack's DOM appears only when the user opens it
// (after loadSpec may have already run). Wire + render the moment it shows up.
new MutationObserver(() => {
  if (SL.spec && document.getElementById('sl-contour') && !document.querySelector('#sl-contour svg')) {
    setup(); window.SoundLab.render();
  }
}).observe(document.documentElement, { childList: true, subtree: true });
