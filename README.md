# reachy-motion

Text-to-motion **+ sound** behaviors for the [Reachy Mini](https://huggingface.co/reachy-mini)
robot — generate expressive moves from text and play them in simulation or on hardware.

This is the Reachy Mini analog of our [kimodo](../kimodo) /
[kimodo-zerogpu-space](../kimodo-zerogpu-space) text-to-motion work, retargeted to
Reachy Mini's 9-DOF body (6-DOF head + body yaw + 2 antennas) instead of a humanoid
skeleton.

## The keystone: one schema, swappable brains

Everything emits the **canonical recorded-move schema** — the same JSON format produced
by the Marionette recorder, stored in the HF moves datasets, and consumed by
`reachy_mini`'s `RecordedMove` / `play_move`:

```json
{
  "description": "...",
  "time": [t0, t1, ...],
  "set_target_data": [{"head": <4x4 matrix>, "antennas": [l, r], "body_yaw": f}, ...]
}
```
…paired by filename with an optional `<name>.wav` for synchronized sound. Because
generated moves are in this format, they're first-class: playable on the robot and
selectable by the conversation app. See [`src/reachy_motion/schema.py`](src/reachy_motion/schema.py).

The plan is staged behind this schema (so the generator can be swapped without touching
playback/UI/caching):

- **Phase 1 (here now): LLM → symbolic motion.** A compact, LLM-emittable spec describes
  each channel as a sum of oscillators; it bakes losslessly into the schema. No training
  data needed. See [`src/reachy_motion/symbolic.py`](src/reachy_motion/symbolic.py).
- **Phase 2: collect a dataset.** Kinesthetic recording (Marionette) on the real robot +
  the generation cache, seeded by `pollen-robotics/reachy-mini-emotions-library` (81 moves).
- **Phase 3: learned model.** Train a small text-conditioned model on the 9-DOF trajectory
  space and drop it in behind the same schema-emitting interface.

## Setup

```bash
uv sync          # installs reachy-mini[mujoco] + this package
```

System note (Linux): `reachy-mini` needs PyGObject, which compiles against
`gobject-introspection-1.0`. If `uv sync` fails building PyGObject:
`sudo apt install -y libgirepository1.0-dev gobject-introspection libcairo2-dev pkg-config`.

## Dataset viewer (Gradio + interactive 3D)

Browse a moves library as an **interactive 3D robot** — orbit, zoom, scrub the timeline —
driven directly from each move's recorded head 4×4 + antenna + body-yaw data, alongside
its description and per-DOF channel plots. The dataset inspector for curating and (later)
generating moves.

```bash
uv run python app.py            # http://127.0.0.1:7861  (GRADIO_SERVER_PORT to override)
```

How it works: the viewer (`src/reachy_motion/web.py`) loads the **proven 8bitkick model
bundle** (URDF + per-part DRACO-compressed GLBs, vendored under `assets/reachy_web/`, from
the `8bitkick/reachy_mini_3d_web_viz` Space) via **`urdf-loader` + `DRACOLoader`**, with a
per-link `MeshPhysicalMaterial` (matte plastic + a `RoomEnvironment` image-based light) —
exactly their render path, which is clean (no splotches). Animation is **IK-free**: we only
have the head 4×4 (not the 7 Stewart joint values, and IK to them is non-finite for some
moves), so we hide the leg linkage, reparent the head platform (`xl_330`) under a
world-aligned control frame, and per frame set `body_yaw` + antenna joints + the head 4×4.

**Connect mode (top-left, above the Move picker):** "Connect robot" does two things —
(1) the browser opens the daemon's live state WebSocket
(`ws://<host>:8000/api/state/ws/full?frequency=30&with_head_joints=true&use_pose_matrix=true`)
and feeds each message into `updateJoints`, so the 3D robot **mirrors the physical one** in
real time (Stewart legs and all, since `head_joints` come from the encoders); and (2) a
server-side SDK client connects (`reachy_motion.robot_control.RobotController`). While
connected, **selecting a move plays it on the physical robot** — motion via `play_move` in a
background thread, sound via the ALSA path (`reachy_motion.audio`) — and the viewer shows the
robot performing it. Changing the move interrupts the previous one (`cancel_move`). Needs the
hardware daemon running (`reachy-mini-daemon`). Recorded (off-robot) playback resumes on Disconnect.

Connect is a **single toggle** (one button: Connect ⇄ Disconnect). Two checkboxes control
hand-guiding:
- **Hand-guide (move by hand)** → motors off (`control_mode: disabled`) — freest, easiest to move.
- **↳ Compliant (firmer; holds position)** → gravity compensation (`gravity_compensation`) —
  resists more but holds its pose. Checking it implies hand-guide; unchecking hand-guide clears both.

While hand-guiding, the live mirror shows the robot following your hand. Disconnecting restores
normal hold (`enable_motors`). **Compliant** needs the daemon on Placo
(`uv add "reachy-mini[placo-kinematics]"` + system `liburdfdom-dev`, then
`reachy-mini-daemon --kinematics-engine Placo`); free hand-guide works on any daemon.

> Earlier approaches and why they were dropped: reconstructing a single GLB from the MJCF
> meshes (`src/reachy_motion/glb.py`, now unused) looked splotchy — GLTF's default material
> is `metalness=1`, and merging parts smeared/inverted normals. The MuJoCo *video* path
> (`dataset.preview` / `render_move.py`) still exists for shareable motion+sound MP4s, but
> its head-pose→Stewart IK is non-finite on some moves.

## See a single move (render to video)

The most reliable way to *see* a move on this machine is to **render it offscreen** to
an MP4/GIF — no window, no GLFW, no daemon. Uses MuJoCo's EGL renderer + the SDK's
analytical IK:

```bash
uv run python scripts/render_move.py --spec examples/curious.json -o out/curious.mp4
uv run python scripts/render_move.py --move  out/curious.json     -o out/curious.gif
uv run python scripts/render_move.py --emotion happy              -o out/happy.mp4
```

There's also a live MuJoCo window (`scripts/view_move.py --spec ...`), but the
interactive GLFW viewer is flaky on this Wayland box (see below) — rendering is the
dependable path, and a preview gallery is what the generation pipeline wants anyway.

> **Why not `reachy-mini-daemon --sim`?** The official sim opens its viewer *inside*
> the daemon process, which also runs a GStreamer/GLib main loop. On this Wayland
> machine the two windowing stacks corrupt the heap and the daemon crashes
> (SIGSEGV/SIGABRT), even with `--no-media`. A clean MuJoCo process (rendering or the
> standalone viewer) has no GStreamer and renders fine. The daemon still works
> **headless** (`--sim --headless`) for driving motion over the SDK — it just can't
> show a window here.

## Play a move (drives the daemon → real robot or headless sim)

```bash
# Terminal 1 — headless sim daemon (owns the sim + SDK; no window)
uv run reachy-mini-daemon --sim --headless

# Terminal 2 — generate a symbolic move and play it through the daemon
uv run python scripts/play_symbolic.py examples/curious.json
uv run python scripts/play_symbolic.py examples/curious.json --save out/curious.json

# Play a move from the official emotions library (downloads ~110MB on first use)
uv run python scripts/play_emotion.py --list
uv run python scripts/play_emotion.py happy
```

On real hardware, drop `--sim --headless` and run the daemon against the robot.

### Sound + synchronized behaviors

Motion and sound are decoupled, which is what makes them robust here:

- **Motion** goes through the daemon with the `no_media` backend — never touches GStreamer.
- **Sound** plays straight to the robot's USB speaker ("Reachy Mini Audio" ALSA card) via
  `reachy_motion.audio` (direct `aplay`), bypassing the SDK media stack entirely.

```bash
# move + sound together (sound starts in sync with the motion)
uv run python scripts/play_symbolic.py examples/curious.json \
    --wav .venv/lib/python3.12/site-packages/reachy_mini/assets/confused1.wav
```

`play(move, wav=...)` pre-positions to the move's start pose, then starts the WAV and the
motion together. A `MoveData` that carries a `sound_path` plays its sound automatically.
`audio.play_samples(np_buffer, sr)` is the entry point TTS will use later.

> **Why not the SDK's `mini.media.play_sound` / `sound=True`?** The SDK media stack
> (camera + audio) needs the GStreamer **webrtc rust plugin**, which isn't installed here,
> so the daemon's media server fails and the `local` client backend can't init. The
> speaker is just a USB audio card, so we play to it directly — independent of the daemon,
> GStreamer, and the motion path. (Installing the webrtc plugin would re-enable the
> official `sound=True` path, but it's unnecessary for this.)

## Symbolic spec format

Channels (LLM-friendly units): `head_x/y/z` (mm), `head_roll/pitch/yaw` (deg),
`antenna_left/right` (deg), `body_yaw` (deg). Each channel is a list of terms, summed;
each term is `{"amp", "freq" (Hz), "phase" (deg), "shape": sin|cos|triangle|square}`
and/or a constant `{"offset"}`. An optional `"envelope"` (`hann`/`fade_in`/`fade_out`/`ease`)
shapes the oscillating motion in/out. See [`examples/curious.json`](examples/curious.json).

## Layout

```
app.py                 # Gradio dataset viewer + interactive 3D + connect/hand-guide
src/reachy_motion/
  schema.py     # canonical move schema: MoveData (build/save/load/round-trip, RecordedMove adapter)
  symbolic.py   # SymbolicMove(Move): parametric spec -> evaluate() / bake() to schema
  render.py     # offscreen EGL render of a move -> MP4/GIF
  viewer.py     # standalone MuJoCo window driven via SDK IK (flaky on Wayland; render preferred)
  audio.py      # play sound on the robot speaker via direct ALSA (bypasses SDK media stack)
  player.py     # connect to daemon (sim or real) and play a Move/MoveData (+ optional sound)
  dataset.py    # load a moves library; metadata, channel signals, (legacy) muxed previews
  web.py        # the three.js viewer module (urdf-loader + DRACO) + trajectory builder
  robot_control.py  # server-side SDK client: play moves on the robot, hand-guide modes
  glb.py        # (unused) MJCF->single-GLB exporter, kept for reference
scripts/
  render_move.py · view_move.py · play_symbolic.py · play_emotion.py
examples/curious.json
assets/reachy_web/     # vendored 8bitkick model bundle (URDF + DRACO GLBs + Kinematics.js)
```

## Attributions

- **Robot model & web-render path** — `assets/reachy_web/` (URDF, per-part DRACO GLBs,
  `Kinematics.js`) is vendored from the [`8bitkick/reachy_mini_3d_web_viz`](https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz)
  Hugging Face Space; the meshes derive from Pollen Robotics' Reachy Mini description.
- **SDK, simulation, moves** — [`pollen-robotics/reachy_mini`](https://github.com/pollen-robotics/reachy_mini)
  and the [emotions library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library).
- **Libraries** — three.js, [urdf-loader](https://github.com/gkjohnson/urdf-loaders), MuJoCo, Gradio.

Review the upstream licenses before redistributing the vendored assets.
