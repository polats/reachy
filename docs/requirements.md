# System requirements

Measured on the dev machine (Ubuntu 25.10, x86_64, RTX 3090, i9-12900K). Numbers are live
measurements, not estimates.

## Requirements by usage tier

The app degrades cleanly — you only need the heavy bits for the features you use.

| Tier | What you get | Needs |
|------|--------------|-------|
| **Simulator only** | 3D move-dataset viewer, sim joystick | Python venv, ~1.3 GB RAM. **No GPU, no daemon, no camera.** |
| **+ Connected** | drive the physical robot, hand-guide/compliant, live camera | + `reachy-mini-daemon` (~3.9 GB RAM), GStreamer + camera stack, a USB camera |
| **+ Live transcription** | streaming speech-to-text under the camera | + NVIDIA GPU (~2.5 GB VRAM) + faster-whisper model (1.6 GB) |

## Hardware

- **CPU:** any modern multicore (dev: i9-12900K, 24 threads). Silero VAD + audio run on CPU, cheap.
- **RAM:** ~8 GB for the full Connected experience.
  - daemon (`reachy-mini-daemon`, Placo + MuJoCo + media): **~3.9 GB RSS**
  - Gradio app (`app.py`): **~1.3 GB RSS**
- **GPU:** NVIDIA + CUDA, **~2.5 GB free VRAM** — **only for live transcription** (faster-whisper
  `large-v3-turbo` fp16 measured at **2386 MiB**). Nothing else uses the GPU. Use a smaller model
  (`distil-large-v3`, `small.en`) or `device="cpu"` in `transcribe.py` to avoid it.
- **Mic:** the Reachy Mini Audio USB card (captured directly via ALSA `arecord`).
- **Camera:** the Reachy Mini USB camera (served over WebRTC by the daemon).

## Disk footprint (what this project adds)

| Item | Size | For |
|---|---|---|
| Python venv (`.venv`) | 1.5 GB | everything |
| faster-whisper `large-v3-turbo` (HF cache) | 1.6 GB | transcription (auto-downloaded on first connect) |
| gst-plugins-rs runtime `.so` (`~/.local/gst-plugins-rs`) | 198 MB | camera (webrtcsink + rtpgccbwe) |
| emotions dataset (HF cache) | 106 MB | move viewer |
| repo + vendored assets | ~6 MB | everything |
| **Total to run** | **~3.4 GB** | |

The model + dataset live in `~/.cache/huggingface`, not the repo.

## OS / system packages

- **Linux x86_64** (tested: Ubuntu 25.10), **PipeWire/ALSA** (`arecord` / `alsa-utils`).
- apt: `gstreamer1.0-nice`, `gstreamer1.0-plugins-{base,good,bad,ugly}`, `libgstreamer1.0-dev`,
  `libgstreamer-plugins-base1.0-dev`, `libgstreamer-plugins-bad1.0-dev`, `pkg-config`, `libssl-dev`.
- **gst-plugins-rs** `webrtcsink` + `rtpgccbwe`, built locally to `~/.local/gst-plugins-rs` — the
  daemon must run with `GST_PLUGIN_PATH=~/.local/gst-plugins-rs/lib/x86_64-linux-gnu`. Full build
  steps are in [camera-webrtc.md](camera-webrtc.md).

## Key Python deps

`gradio`, `reachy-mini[mujoco,placo-kinematics]`, `onnxruntime` (Silero VAD), `faster-whisper`
(STT, pulls ctranslate2 + tokenizers), `numpy`, `trimesh`, `imageio`, `matplotlib`. **No torch**
(deliberately — Silero runs via onnxruntime, STT via CTranslate2).

## Build-time only (safe to remove after setup)

These were needed to *produce* the runtime artifacts and can be deleted to reclaim space (already
done on this machine):

- `~/gst-plugins-rs` build tree — **2.9 GB** (cargo target + source). To rebuild the plugins, re-clone
  and follow [camera-webrtc.md](camera-webrtc.md):
  ```bash
  git clone --depth 1 --branch 0.14.1 https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git ~/gst-plugins-rs
  cd ~/gst-plugins-rs
  cargo cinstall -p gst-plugin-webrtc --prefix=$HOME/.local/gst-plugins-rs --release
  cargo cinstall -p gst-plugin-rtp    --prefix=$HOME/.local/gst-plugins-rs --release
  ```
- gst-plugins-rs `.a` static libs (~487 MB) — runtime loads only the `.so`. (Removed; the install
  is now 198 MB.)
- Rust toolchain + `cargo-c`, and node/npm (for `gstwebrtc-api.js`) — only used during the build.

## GPU note

Verified: faster-whisper on the GPU works out of the box here — CTranslate2 found the CUDA/cuDNN
libs, no extra setup. If on another machine the GPU path errors on cuDNN/cuBLAS, install the
`nvidia-cudnn-cu12` / `nvidia-cublas-cu12` wheels or set `device="cpu"` in `transcribe.py`.
