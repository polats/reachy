"""Check and restart the reachy-mini daemon from the Gradio app.

The daemon is relaunched the same way we run it by hand: with the Placo kinematics
engine (needed for gravity-compensation / compliant hold) and with ``GST_PLUGIN_PATH``
pointing at the locally-built gst-plugins-rs so the robot camera (webrtcsink) keeps
working after a restart. See docs/camera-webrtc.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DAEMON_URL = "http://localhost:8000"
DAEMON_PORT = "8000"

_ROOT = Path(__file__).resolve().parents[2]
_DAEMON_BIN = Path(sys.executable).resolve().parent / "reachy-mini-daemon"
# locally-built gst-plugins-rs (webrtcsink/rtpgccbwe) — required for the camera
_GST_PLUGIN_PATH = str(Path.home() / ".local" / "gst-plugins-rs" / "lib" / "x86_64-linux-gnu")
_LOG = Path("/tmp/reachy-daemon.log")


def is_up(timeout: float = 1.5) -> bool:
    """True if the daemon's HTTP API answers on :8000."""
    try:
        with urllib.request.urlopen(DAEMON_URL + "/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _launch_cmd() -> list[str]:
    if _DAEMON_BIN.exists():
        return [str(_DAEMON_BIN), "--kinematics-engine", "Placo"]
    return ["uv", "run", "reachy-mini-daemon", "--kinematics-engine", "Placo"]  # fallback


def restart(wait: int = 45) -> bool:
    """Kill whatever holds :8000 and relaunch the daemon. Blocks until it's healthy (or times out).

    Returns True if the daemon comes back up within ``wait`` seconds.
    """
    # port-scoped kill only (never a broad pkill)
    subprocess.run(["fuser", "-k", f"{DAEMON_PORT}/tcp"], capture_output=True)
    time.sleep(2)

    env = dict(os.environ)
    gp = _GST_PLUGIN_PATH
    if env.get("GST_PLUGIN_PATH"):
        gp = gp + os.pathsep + env["GST_PLUGIN_PATH"]
    env["GST_PLUGIN_PATH"] = gp

    with open(_LOG, "ab") as log:
        subprocess.Popen(
            _launch_cmd(),
            cwd=str(_ROOT),
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,  # detach so it survives the app
        )

    for _ in range(wait):
        if is_up():
            return True
        time.sleep(1)
    return False
