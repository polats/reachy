# Robot camera over WebRTC

The **📷 Camera** panel (Connected tab, above the channels chart) shows the robot's live
camera feed. Expand the accordion to start it, collapse to stop it. It only runs when
connected to a physical robot.

This document records everything required to make it work, because most of the setup lives
**outside this repo** (system packages, locally-built GStreamer plugins, and one patch to the
installed `reachy-mini` SDK). On a fresh machine the panel will say "camera unavailable" until
these are in place.

## How it works

The daemon does **not** expose the camera over HTTP. It publishes the camera as a **WebRTC
producer** on a GStreamer signalling server (`ws://<host>:8443`). The browser consumes it:

```
libcamera ──► daemon media_server (webrtcsink, run-signalling-server=true) ──► :8443 signaller
                                                                                    │
browser:  gstwebrtc-api  ──connect ws://host:8443──► registerPeerListener ──────────┘
          producerAdded → createConsumerSession(id) → streamsChanged → <video>.srcObject
```

- Browser side: `assets/reachy_web/gstwebrtc-api.js` (loaded via `<script>` in the page head;
  exposes `window.GstWebRTCAPI`). Consumer logic is `startCamera()` / `stopCamera()` /
  `camTick()` in `src/reachy_motion/web.py`; the `<video id="reachy-cam">` lives in `CAMERA_HTML`.
- `camTick` (700 ms poll) starts the stream when `V.live` (connected) **and** the Camera
  accordion is expanded (`#reachy-cam` is visible), and stops it otherwise.

## In-repo files (committed)

| File | Role |
|------|------|
| `assets/reachy_web/gstwebrtc-api.js` | Vendored WebRTC consumer lib, **v3.0.0**, built from `gst-plugins-rs` (see below). Must match the signaller's protocol. |
| `src/reachy_motion/web.py` | `CAMERA_HTML`, `startCamera`/`stopCamera`/`camTick`, the `#reachy-cam` `<video>`, accordion-driven on/off. |
| `app.py` | `GSTWEBRTC_URL`, the `<script>` head injection, and the `📷 Camera` accordion above `CHART_HTML`. |
| `src/reachy_motion/robot_control.py` | `connect()` calls `self._mini.acquire_media()` so the control connection doesn't shut the camera off (see "Gotcha: media release"). |

## Out-of-repo prerequisites (NOT committed — must be set up per machine)

Verified on **Ubuntu 25.10, GStreamer 1.26.6, x86_64**.

### 1. System packages (apt)

```bash
sudo apt install -y \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
  pkg-config libssl-dev \
  gstreamer1.0-nice          # nicesrc/nicesink — webrtcbin ICE; without it: "Failed to request pad from webrtcbin"
```

`gstreamer1.0-plugins-{base,good,bad,ugly}` (encoders/payloaders, dtls, srtp) are assumed present.

### 2. The `webrtcsink` + `rtpgccbwe` GStreamer plugins (built from source)

These come from `gst-plugins-rs` and are **not** packaged for Ubuntu. Build them with `cargo-c`
(needs rustc ≥ 1.94 — `rustup update stable` if older):

```bash
cargo install cargo-c
git clone --depth 1 --branch 0.14.1 \
  https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git ~/gst-plugins-rs
cd ~/gst-plugins-rs
cargo cinstall -p gst-plugin-webrtc --prefix=$HOME/.local/gst-plugins-rs --release
cargo cinstall -p gst-plugin-rtp    --prefix=$HOME/.local/gst-plugins-rs --release
```

Installs `libgstrswebrtc.so` (webrtcsink) and `libgstrsrtp.so` (rtpgccbwe — congestion control)
under `~/.local/gst-plugins-rs/lib/x86_64-linux-gnu/gstreamer-1.0/`.

### 3. Run the daemon with `GST_PLUGIN_PATH`

The daemon must see the locally-built plugins:

```bash
GST_PLUGIN_PATH=$HOME/.local/gst-plugins-rs/lib/x86_64-linux-gnu \
  uv run reachy-mini-daemon --kinematics-engine Placo
```

On success the daemon log shows the media server starting and `:8443` opens
(`ss -tlnp | grep 8443`).

### 4. The vendored `gstwebrtc-api.js` must match the signaller

`assets/reachy_web/gstwebrtc-api.js` was built from the **same** `gst-plugins-rs` 0.14.1 tree
(`net/webrtc/gstwebrtc-api`, package **v3.0.0**) so its signalling protocol matches the
`webrtcsink` server:

```bash
cd ~/gst-plugins-rs/net/webrtc/gstwebrtc-api && npm install && npm run build
cp dist/gstwebrtc-api-3.0.0.min.js <repo>/assets/reachy_web/gstwebrtc-api.js
```

> An older vendored copy (e.g. from the desktop app) uses the pre-3.0 `registerProducersListener`
> API and **silently fails** against a 0.14.1 server. If you rebuild the plugins at a different
> version, rebuild this lib from the matching tree.

## What's missing / known issues

### Temporary SDK patch — video-only (NOT durable)

The daemon's WebRTC pipeline includes an audio capture branch (`pulsesrc`). On this machine that
source **times out at runtime** (no working capture device) and tears down the **whole** pipeline —
so the video producer never registers and there's nothing to consume.

Workaround: the installed SDK is patched to skip audio (video-only):

- File: `.venv/lib/python3.12/site-packages/reachy_mini/media/media_server.py`
- Change: `_build_audio_source()` returns `None` at the top (marker comment `TEMP-REACHY-MOTION`).

**This is lost whenever `reachy-mini` is reinstalled/upgraded.** A durable fix is still needed —
e.g. an env/config switch to disable the daemon's audio branch, fixing the audio device so
`pulsesrc` succeeds, or upstreaming "skip audio on runtime failure". The camera is **video-only**;
there is no audio track.

### Gotcha: media release on connect

Connecting the control client with `media_backend="no_media"` makes the daemon **release** its
camera/mic, which shuts down the media server and `:8443`. `RobotController.connect()` therefore
calls `acquire_media()` right after connecting to bring the camera back. (Restarting the daemon
while the Gradio app holds a connection will also drop it — restart the app after the daemon.)

## Camera modes / lens

The Reachy Mini Lite has **one physical lens** — a single wide-angle camera (the two
`/dev/video*` nodes are the same device's capture + metadata, not two lenses). It is distorted
in hardware and undistorted in software via the intrinsics (`K`) and distortion (`D`) the daemon
publishes at `GET /api/camera/specs`.

There are **four resolution modes**. Each has a different `crop_factor`, so picking a mode is the
closest thing to "changing the lens" (it changes the field of view / zoom):

| Mode | Resolution | FPS | crop_factor | View |
|------|-----------|-----|-------------|------|
| **default** | 1920×1080 | 60 | 1.115 | narrowest |
| | 3840×2592 | 30 | **1.0** | **widest** (full sensor) |
| | 3840×2160 | 30 | 1.109 | |
| | 3264×2448 | 30 | 1.115 | |

**There is no supported way to switch modes at runtime.** Checked all three surfaces:

- **Daemon API** — `/api/camera/specs` is read-only; there is no `set-resolution` endpoint.
- **Env var / config** — `media_server` hardcodes `self._resolution = camera_specs.default_resolution`;
  nothing reads an override.
- **CLI** — `reachy-mini-daemon` has no camera/resolution flag (only `--no-media`).

The WebRTC pipeline is therefore locked to **1920×1080@60 (crop 1.115, the narrowest view)** when the
media server starts.

### What changing it would take

The resolution is baked into the GStreamer pipeline at build time, so switching means rebuilding the
pipeline (restart the daemon's media) with a different mode. Practical path, **not implemented**:

1. Patch the SDK's `media_server` to read the resolution from an override (e.g. env var
   `REACHY_CAM_RESOLUTION`) instead of always `default_resolution` — another `.venv` patch, as
   fragile as the audio one above.
2. Add a mode dropdown in the Camera panel that sets the override and restarts the daemon media.

Trade-offs: each switch costs a media restart (a few seconds, brief feed drop), and the wider 4K
modes run at 30 fps and use much more WebRTC bandwidth than the default 1080p@60.

## Troubleshooting

| Symptom | Cause / check |
|---------|---------------|
| Panel: "camera library not loaded" | `gstwebrtc-api.js` didn't load; check the `<script>` URL. |
| Panel stuck "connecting…", console "cannot connect to signaling server" | `:8443` not up. Daemon launched without `GST_PLUGIN_PATH`, or `webrtcsink` missing (`gst-inspect-1.0 webrtcsink`). |
| Daemon log: "Failed to create webrtcsink element" | Plugin not built/visible — see step 2 + `GST_PLUGIN_PATH`. |
| Daemon log: "Failed to request pad from webrtcbin" | `gstreamer1.0-nice` not installed (no ICE). |
| `:8443` up but producer list empty | Audio branch tearing down the pipeline — apply the video-only patch (above). |
| Producer found but no video, `registerProducersListener is not a function` | Vendored `gstwebrtc-api.js` is the wrong (pre-3.0) version. |
