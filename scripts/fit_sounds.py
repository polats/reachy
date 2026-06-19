"""Analysis-by-synthesis: fit a sound.py spec to each emotion in the Reachy library.

For every emotion we measure the real .wav and emit a declarative SoundSpec whose MELODY is the
measured pitch contour (real prosody, not hand-guessed) and whose VOICE is set from the measured
formants / rasp / brightness. Writes src/reachy_motion/fitted_sounds.json = {emotion: spec}.

That JSON is two things at once: data-grounded presets the synth can play, AND the {emotion -> spec}
training corpus for the autonomous "describe it -> generate the sound" goal.

Run:  uv run python scripts/fit_sounds.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import wave
from pathlib import Path

import numpy as np

DS = "/home/paul/.cache/huggingface/hub/datasets--pollen-robotics--reachy-mini-emotions-library/snapshots/152e84b8f46b88c4b52dd34bbef6975637366177"
OUT = Path(__file__).resolve().parents[1] / "src" / "reachy_motion" / "fitted_sounds.json"
MAX_DUR = 3.0          # cap a fitted sound's length (library clips run 3-16s; we want snappy ones)
N_TONES = 12           # melody resolution (tone keyframes)


def load(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels(); sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 3:        # 24-bit
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        x = np.where(x & 0x800000, x - 0x1000000, x).astype(np.float64)
    else:
        x = np.frombuffer(raw, dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x / (np.abs(x).max() + 1e-9), sr


def pitch_contour(x, sr):
    """Autocorrelation pitch per 50ms frame -> (times, f0) for voiced frames."""
    hop, win = int(0.02 * sr), int(0.05 * sr)
    ts, f0 = [], []
    for i in range(0, len(x) - win, hop):
        s = x[i:i + win] * np.hanning(win)
        if (s ** 2).mean() < 2e-4:
            continue
        ac = np.correlate(s, s, "full")[win - 1:]
        lo, hi = int(sr / 900), int(sr / 80)
        if hi >= len(ac):
            continue
        p = lo + int(np.argmax(ac[lo:hi]))
        if ac[p] > 0.3 * ac[0]:
            ts.append(i / sr); f0.append(sr / p)
    return np.array(ts), np.array(f0)


def formants(x, sr):
    """Two strongest peaks of the smoothed long-term average spectrum, in 350-3200 Hz."""
    win = int(0.05 * sr); acc = None; k = 0
    for i in range(0, len(x) - win, win // 2):
        X = np.abs(np.fft.rfft(x[i:i + win] * np.hanning(win)))
        acc = X if acc is None else acc + X; k += 1
    if acc is None:
        return [760.0, 1500.0]
    fr = np.fft.rfftfreq(win, 1 / sr); mag = acc / k
    sm = np.convolve(mag, np.ones(9) / 9, "same")            # smooth -> spectral envelope
    band = (fr > 350) & (fr < 3200)
    fb, mb = fr[band], sm[band]
    peaks = [j for j in range(1, len(mb) - 1) if mb[j] > mb[j - 1] and mb[j] > mb[j + 1]]
    peaks.sort(key=lambda j: -mb[j])
    f = sorted(float(fb[j]) for j in peaks[:2]) or [760.0, 1500.0]
    while len(f) < 2:
        f.append(min(f[-1] * 2, 3000.0))
    return [round(f[0]), round(f[1])]


def fit(paths):
    xs = [load(p) for p in paths]
    sr = xs[0][1]
    # use the variant closest to the median duration as the representative
    durs = [len(x) / s for x, s in xs]
    rep_x, _ = xs[int(np.argmin(np.abs(np.array(durs) - np.median(durs))))]
    ts, f0 = pitch_contour(rep_x, sr)
    dur = min(len(rep_x) / sr, MAX_DUR)
    if len(f0) < 3:                                          # mostly noise -> a flat midrange utter
        contour = [500.0] * N_TONES
    else:
        contour = np.clip(np.interp(np.linspace(0, 1, N_TONES), ts / ts[-1], f0), 150, 880)
    # vibrato from the pitch residual (oscillation around a smoothed contour)
    vd = 0.015
    if len(f0) > 8:
        base = np.convolve(f0, np.ones(5) / 5, "same")
        resid = (f0 - base) / (np.median(f0) + 1e-9)
        vd = float(np.clip(np.std(resid) * 1.2, 0.005, 0.05))
    zcr = float(((rep_x[:-1] * rep_x[1:]) < 0).mean())
    rasp = float(np.clip((zcr - 0.05) / 0.22, 0.0, 0.55))
    fr = np.fft.rfftfreq(len(rep_x), 1 / sr); X = np.abs(np.fft.rfft(rep_x * np.hanning(len(rep_x))))
    cen = float((fr * X).sum() / (X.sum() + 1e-9))
    tilt = float(np.clip(1.7 - cen / 2400, 0.6, 1.6))       # brighter -> less spectral tilt
    f1, f2 = formants(rep_x, sr)
    tdur = round(dur / N_TONES, 3)
    segs = [{"f0": round(float(c), 1), "dur": tdur, "ease": "smooth", "amp": 1.0,
             "vib_rate": 6.0, "vib_depth": round(vd, 3), "rasp": round(rasp, 3)} for c in contour]
    return {"sr": sr,
            "voice": {"harmonics": 9, "tilt": round(tilt, 2), "formants": [f1, f2],
                      "formant_q": 5.0, "rasp": round(rasp, 3), "vib_rate": 6.0, "vib_depth": round(vd, 3)},
            "segments": segs}


def main():
    groups = {}
    for w in sorted(glob.glob(os.path.join(DS, "*.wav"))):
        lab = re.sub(r"\d+$", "", os.path.splitext(os.path.basename(w))[0])
        groups.setdefault(lab, []).append(w)
    fitted = {lab: fit(ws) for lab, ws in groups.items()}
    OUT.write_text(json.dumps(fitted, indent=1))
    print(f"fitted {len(fitted)} emotions -> {OUT}")
    for lab in ["curious", "sad", "surprised", "cheerful", "no", "boredom"]:
        if lab in fitted:
            v = fitted[lab]["voice"]; c = [s["f0"] for s in fitted[lab]["segments"]]
            print(f"  {lab:10s} formants={v['formants']} tilt={v['tilt']} rasp={v['rasp']} "
                  f"vib={v['vib_depth']} contour={c}")


if __name__ == "__main__":
    main()
