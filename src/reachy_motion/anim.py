"""Code-authored animations for Reachy Mini — a tiny "Remotion for the robot".

An animation is a declarative **spec** (JSON-serializable): a timeline of eased pose
segments (+ additive "life" layers). ``bake()`` samples it at fps into the canonical
move schema (per-frame head 4x4 + antennas + body_yaw + timestamps) — the same format
the 3D viewer and robot player consume. Because the spec is plain data, the same
animations can be authored in code, edited in the UI, saved, and round-tripped.

Authoring units are friendly: degrees for roll/pitch/yaw/antennas/body, millimetres for
x/y/z. bake() converts to the SI command vector and runs IK.

Pose channels (an animal-ish vocabulary):
    pitch = nod (down/up)      roll = head tilt ("huh?")     yaw = neck turn (head only)
    body  = whole-body turn    z = lift/crane (mm)           x,y = small head shift (mm)
    antL/antR = the two antennas ("ears"); + up/perked, - back/down
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .web import pose_to_render

FPS = 40
CHANNELS = ("x", "y", "z", "roll", "pitch", "yaw", "antL", "antR", "body")


@dataclass
class Pose:
    x: float = 0.0; y: float = 0.0; z: float = 0.0          # mm
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0  # deg
    antL: float = 0.0; antR: float = 0.0                     # deg
    body: float = 0.0                                        # deg

    def lerp(self, other: "Pose", u: float) -> "Pose":
        return Pose(**{c: getattr(self, c) + (getattr(other, c) - getattr(self, c)) * u
                       for c in CHANNELS})

    def as_dict(self) -> dict:
        return {c: round(float(getattr(self, c)), 2) for c in CHANNELS}


# --- pose palette (anchors / seeds) ---
NEUTRAL = Pose()
CURIOUS = Pose(roll=14, pitch=-6, z=8, yaw=8, antL=34, antR=28)
ALERT = Pose(z=12, pitch=8, antL=48, antR=48)
SLEEPY = Pose(z=-10, pitch=-16, antL=-30, antR=-34)
PERK_EARS = Pose(antL=45, antR=45)


# --- easing (the craft layer) ---
def _linear(u): return u
def _smooth(u): return u * u * u * (u * (u * 6 - 15) + 10)
def _ease_out(u): return 1 - (1 - u) ** 3
def _ease_in(u): return u ** 3
def _back(u):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2
def _anticipate(u):
    c1 = 1.70158
    return u * u * ((c1 + 1) * u - c1)

EASES = {"linear": _linear, "smooth": _smooth, "ease_out": _ease_out,
         "ease_in": _ease_in, "back": _back, "anticipate": _anticipate, "hold": _linear}


# --- additive "life" layers: f(t_seconds) -> dict of channel deltas ---
def _breath(amp_z=2.0, amp_pitch=1.2, period=4.0):
    def f(t):
        s = math.sin(2 * math.pi * t / period)
        return {"z": amp_z * s, "pitch": amp_pitch * s}
    return f

def _ear_idle(amp=4.0, period=2.3):
    def f(t):
        return {"antL": amp * math.sin(2 * math.pi * t / period),
                "antR": amp * math.sin(2 * math.pi * t / period + 0.7)}
    return f

LAYERS = {"breath": _breath, "ear_idle": _ear_idle}


@dataclass
class Anim:
    """A timeline of eased pose segments, baked to the move schema."""
    fps: int = FPS
    _kf: list = field(default_factory=lambda: [(0.0, NEUTRAL)])   # (end_time, pose)
    _segs: list = field(default_factory=list)                     # (t0, t1, from, to, ease_name)
    _layers: list = field(default_factory=list)                   # (type, params)

    @property
    def _t(self): return self._kf[-1][0]
    @property
    def _pose(self): return self._kf[-1][1]

    def to(self, target: Pose, dur: float, ease: str = "smooth") -> "Anim":
        t0, p0 = self._t, self._pose
        self._segs.append((t0, t0 + max(1e-3, dur), p0, target, ease))
        self._kf.append((t0 + max(1e-3, dur), target))
        return self

    def hold(self, dur: float) -> "Anim":
        return self.to(self._pose, dur, "hold")

    def layer(self, kind: str, **params) -> "Anim":
        self._layers.append((kind, params))
        return self

    def _sample(self, t: float) -> Pose:
        p = self._pose
        for t0, t1, a, b, ease in self._segs:
            if t <= t1:
                u = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
                p = a.lerp(b, EASES.get(ease, _smooth)(u)); break
        d = {c: getattr(p, c) for c in CHANNELS}
        for kind, params in self._layers:
            for k, v in LAYERS[kind](**params)(t).items():
                d[k] += v
        return Pose(**d)

    def bake(self) -> dict:
        dur = self._t or 0.5
        n = max(2, int(round(dur * self.fps)) + 1)
        times = [i / self.fps for i in range(n)]
        head, head_joints, antennas, body_yaw = [], [], [], []
        ch = {c: [] for c in CHANNELS}
        for t in times:
            p = self._sample(t)
            cmd = {"x": p.x / 1000, "y": p.y / 1000, "z": p.z / 1000,
                   "roll": math.radians(p.roll), "pitch": math.radians(p.pitch),
                   "yaw": math.radians(p.yaw), "antL": math.radians(p.antL),
                   "antR": math.radians(p.antR), "body": math.radians(p.body)}
            r = pose_to_render(cmd)
            head.append(r["head_pose"]); head_joints.append(r["head_joints"])
            antennas.append(r["antennas_position"]); body_yaw.append(math.radians(p.body))
            for c in CHANNELS:
                ch[c].append(round(float(getattr(p, c)), 2))
        return {"time": times, "head": head, "head_joints": head_joints,
                "antennas": antennas, "body_yaw": body_yaw, "channels": ch}

    def spec(self) -> dict:
        return {
            "fps": self.fps,
            "segments": [{"dur": round(t1 - t0, 3), "ease": ease, "pose": b.as_dict()}
                         for (t0, t1, a, b, ease) in self._segs],
            "layers": [{"type": k, **p} for (k, p) in self._layers],
        }


def from_spec(spec: dict) -> Anim:
    a = Anim(fps=int(spec.get("fps", FPS)))
    for seg in spec.get("segments", []):
        a.to(Pose(**{c: float(seg.get("pose", {}).get(c, 0.0)) for c in CHANNELS}),
             float(seg.get("dur", 0.5)), seg.get("ease", "smooth"))
    for lyr in spec.get("layers", []):
        kind = lyr.get("type")
        if kind in LAYERS:
            a.layer(kind, **{k: v for k, v in lyr.items() if k != "type"})
    return a


def bake_spec(spec: dict) -> dict:
    return from_spec(spec).bake()


# ---- the built-in animations (authored in code, like a Remotion composition) ----

def _curious_look() -> Anim:
    return (Anim()
            .to(PERK_EARS, 0.18, "ease_out")
            .to(CURIOUS, 0.45, "back")
            .hold(0.6)
            .to(Pose(roll=10, pitch=-6, z=8, yaw=-6, antL=34, antR=28), 0.5, "smooth")
            .hold(0.5)
            .layer("breath"))

def _yes_nod() -> Anim:
    down = Pose(pitch=-16, antL=10, antR=10); up = Pose(pitch=8, antL=10, antR=10)
    return (Anim()
            .to(up, 0.15, "anticipate")
            .to(down, 0.22, "back").to(up, 0.22, "back")
            .to(down, 0.22, "back").to(NEUTRAL, 0.3, "smooth")
            .layer("breath"))

def _no_shake() -> Anim:
    L = Pose(yaw=-14, body=-12, antL=-12, antR=-12); R = Pose(yaw=14, body=12, antL=-12, antR=-12)
    return (Anim()
            .to(L, 0.2, "ease_out").to(R, 0.3, "smooth").to(L, 0.3, "smooth")
            .to(R, 0.3, "smooth").to(NEUTRAL, 0.3, "smooth")
            .layer("breath"))

def _alert() -> Anim:
    return (Anim()
            .to(ALERT, 0.12, "ease_out")
            .to(Pose(z=12, pitch=8, yaw=-16, antL=48, antR=48), 0.25, "smooth")
            .to(Pose(z=12, pitch=8, yaw=16, antL=48, antR=48), 0.4, "smooth")
            .to(ALERT, 0.25, "smooth").hold(0.4)
            .layer("ear_idle", amp=2.5))

def _sleepy() -> Anim:
    return (Anim()
            .to(Pose(z=-4, pitch=-8, antL=-30, antR=-34), 0.8, "ease_in")
            .to(SLEEPY, 1.0, "smooth").hold(0.8)
            .layer("breath", amp_z=3.0, amp_pitch=2.0, period=5.0))


def _greeting() -> Anim:
    """Friendly hello — ears pop up, two small quick nods, a little body turn, settle."""
    perk = Pose(antL=42, antR=42, z=4)
    return (Anim()
            .to(perk, 0.18, "ease_out")
            .to(Pose(pitch=-10, antL=42, antR=42, z=4, body=8), 0.18, "back")
            .to(Pose(pitch=4, antL=42, antR=42, z=4, body=8), 0.18, "back")
            .to(Pose(pitch=-8, antL=42, antR=42, z=4, body=-6), 0.18, "back")
            .to(NEUTRAL, 0.4, "smooth")
            .layer("ear_idle", amp=3.0))

def _look_around() -> Anim:
    """Scans the room — turns to look left, then right, ears following, back to center."""
    left = Pose(yaw=-18, body=-14, roll=-6, antL=30, antR=20)
    right = Pose(yaw=18, body=14, roll=6, antL=20, antR=30)
    return (Anim()
            .to(Pose(antL=30, antR=30, z=4), 0.2, "ease_out")
            .to(left, 0.6, "smooth").hold(0.4)
            .to(right, 0.9, "smooth").hold(0.4)
            .to(NEUTRAL, 0.5, "smooth")
            .layer("breath"))

def _happy_bounce() -> Anim:
    """Excited — bouncy ups with ears perked and a wiggle."""
    up = Pose(z=11, pitch=6, antL=44, antR=44); dn = Pose(z=-2, pitch=-4, antL=40, antR=40)
    return (Anim()
            .to(up, 0.16, "back").to(dn, 0.16, "back")
            .to(up, 0.16, "back").to(dn, 0.16, "back")
            .to(Pose(z=6, antL=44, antR=44, body=8), 0.18, "back")
            .to(Pose(z=6, antL=44, antR=44, body=-8), 0.2, "back")
            .to(NEUTRAL, 0.35, "smooth")
            .layer("ear_idle", amp=4.0, period=0.8))

def _scared() -> Anim:
    """Flinch — quick shrink down, ears flatten back, a small tremble, stay low."""
    cower = Pose(z=-12, pitch=-14, antL=-44, antR=-46)
    return (Anim()
            .to(cower, 0.12, "ease_out")
            .to(Pose(z=-12, pitch=-14, yaw=-5, antL=-44, antR=-46), 0.12, "smooth")
            .to(Pose(z=-12, pitch=-14, yaw=5, antL=-44, antR=-46), 0.12, "smooth")
            .to(cower, 0.12, "smooth").hold(0.6)
            .layer("breath", amp_z=1.0, amp_pitch=0.8, period=1.5))


BUILTINS = {
    "curious_look": _curious_look,
    "yes_nod": _yes_nod,
    "no_shake": _no_shake,
    "alert": _alert,
    "sleepy": _sleepy,
    "greeting": _greeting,
    "look_around": _look_around,
    "happy_bounce": _happy_bounce,
    "scared": _scared,
}


def builtin_specs() -> dict:
    return {name: fn().spec() for name, fn in BUILTINS.items()}
