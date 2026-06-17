"""Gradio dataset viewer for Reachy Mini moves — interactive 3D.

Browse a moves library (default: the 81-move emotions library) as an interactive 3D
robot (orbit / zoom / scrub timeline) driven directly from each move's head 4x4 +
antenna + body-yaw data, alongside its description and per-DOF channel plots.

    uv run python app.py            # http://127.0.0.1:7861
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reachy_motion import dataset as ds  # noqa: E402
from reachy_motion.robot_control import RobotController  # noqa: E402
from reachy_motion.web import CONTAINER_HTML, HEAD_HTML, move_trajectory  # noqa: E402

LIBRARY = ds.DEFAULT_LIBRARY
controller = RobotController()

# Vendored 8bitkick model bundle (URDF + per-part DRACO GLBs).
WEB_ROOT = Path(__file__).resolve().parent / "assets" / "reachy_web"
WEB_ASSETS = WEB_ROOT / "assets"
URDF_URL = f"/gradio_api/file={(WEB_ASSETS / 'reachy-mini.urdf').resolve()}"
MESH_BASE = f"/gradio_api/file={(WEB_ASSETS / 'meshes_optimized').resolve()}/"
KIN_URL = f"/gradio_api/file={(WEB_ROOT / 'src' / 'Kinematics.js').resolve()}"


def _plot(name: str):
    ch = ds.channels(name, LIBRARY)
    t = ch["t"]
    fig, axes = plt.subplots(3, 1, figsize=(6, 5.2), sharex=True)
    axes[0].plot(t, ch["head_roll"], label="roll")
    axes[0].plot(t, ch["head_pitch"], label="pitch")
    axes[0].plot(t, ch["head_yaw"], label="yaw")
    axes[0].set_ylabel("head °")
    axes[1].plot(t, ch["head_x"], label="x")
    axes[1].plot(t, ch["head_y"], label="y")
    axes[1].plot(t, ch["head_z"], label="z")
    axes[1].set_ylabel("head mm")
    axes[2].plot(t, ch["antenna_left"], label="ant L")
    axes[2].plot(t, ch["antenna_right"], label="ant R")
    axes[2].plot(t, ch["body_yaw"], label="body", linestyle="--")
    axes[2].set_ylabel("° ")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _info_md(name: str) -> str:
    info = ds.move_info(name, LIBRARY)
    sound = "🔊 with sound" if info.has_sound else "🔇 no sound"
    return (
        f"### {info.name}\n\n{info.description}\n\n"
        f"**{info.duration:.2f}s** · **{info.num_frames} frames** · {sound}"
    )


def select_move(name: str, connected: bool):
    if not name:
        return "", "", None
    info, plot = _info_md(name), _plot(name)
    if connected:
        # play the move on the physical robot (motion + sound); the viewer mirrors the
        # robot's live state, so no recorded trajectory is pushed.
        move = ds.get_library(LIBRARY).get(name)
        controller.play(move, move.sound_path)
        return "", info, plot
    data = move_trajectory(name, LIBRARY)
    wav = ds.get_library(LIBRARY).get(name).sound_path
    if wav is not None and Path(wav).exists():
        data["audio"] = f"/gradio_api/file={Path(wav).resolve()}"
    return json.dumps(data), info, plot


_CONNECTED_MSG = "🟢 **Connected** — selecting a move plays it on the robot"


def toggle_connect(connected: bool):
    """Single toggle: connect if disconnected, disconnect if connected."""
    off = (gr.update(value=False),)  # reset both guide checkboxes
    if connected:
        controller.disconnect()
        return (False, "not connected", gr.update(value="Connect robot", variant="primary"),
                gr.update(value=False), gr.update(value=False))
    try:
        controller.connect()
        return (True, _CONNECTED_MSG, gr.update(value="Disconnect robot", variant="secondary"),
                gr.update(value=False), gr.update(value=False))
    except Exception as e:  # noqa: BLE001
        return (False, f"🔴 Connect failed — is `reachy-mini-daemon` running on :8000?\n\n`{e}`",
                gr.update(value="Connect robot", variant="primary"),
                gr.update(value=False), gr.update(value=False))


_FREE_MSG = "🖐️ **Hand-guide · free** — easiest to move by hand; the viewer mirrors."
_COMPLIANT_MSG = "🖐️ **Hand-guide · compliant** — firmer, holds position. Move it by hand; the viewer mirrors."
_PLACO_HINT = "⚠️ **Compliant** needs the daemon on `--kinematics-engine Placo`."


def _apply(hand_guide: bool, compliant: bool):
    ok = controller.set_mode(hand_guide, compliant)
    msg = _CONNECTED_MSG if not hand_guide else (_COMPLIANT_MSG if compliant else _FREE_MSG)
    if not ok and compliant:
        msg = _PLACO_HINT
    return msg, gr.update(value=hand_guide), gr.update(value=compliant)


def on_hand_guide(hand_guide: bool, compliant: bool, connected: bool):
    if not connected:
        return "Connect to the robot first.", gr.update(value=False), gr.update(value=False)
    # turning hand-guide off clears compliant and returns to normal hold
    return _apply(hand_guide, compliant and hand_guide)


def on_compliant(hand_guide: bool, compliant: bool, connected: bool):
    if not connected:
        return "Connect to the robot first.", gr.update(value=False), gr.update(value=False)
    # compliant implies hand-guide; unchecking it falls back to free if still hand-guiding
    return _apply(hand_guide or compliant, compliant)


def build() -> gr.Blocks:
    names = ds.list_moves(LIBRARY)
    head = (
        f"<script>window.REACHY_URDF_URL='{URDF_URL}';"
        f"window.REACHY_MESH_BASE='{MESH_BASE}';"
        f"window.REACHY_KIN_URL='{KIN_URL}';</script>\n" + HEAD_HTML
    )
    with gr.Blocks(title="Reachy Mini — Move Dataset Viewer", head=head) as demo:
        gr.Markdown(
            f"# 🤖 Reachy Mini — Move Dataset Viewer\n"
            f"Browsing **{len(names)} moves** from `{LIBRARY}` as an interactive 3D robot, "
            f"driven directly from each move's recorded head/antenna/body data."
        )
        traj = gr.Textbox(visible=False)  # carries trajectory JSON to the JS viewer
        connected = gr.State(False)
        with gr.Row():
            with gr.Column(scale=1):
                connect_btn = gr.Button("Connect robot", variant="primary")
                hand_guide_chk = gr.Checkbox(label="Hand-guide (move by hand)", value=False)
                compliant_chk = gr.Checkbox(label=" ↳ Compliant (firmer; holds position)", value=False)
                status = gr.Markdown("not connected")
                picker = gr.Dropdown(choices=names, value=names[0], label="Move", filterable=True)
                info = gr.Markdown()
            with gr.Column(scale=2):
                gr.HTML(CONTAINER_HTML)
            with gr.Column(scale=2):
                plot = gr.Plot(label="Channels (head pose · antennas · body)")

        # single toggle: backend SDK client (to command the robot) + browser mirror WS
        connect_btn.click(
            toggle_connect, inputs=connected,
            outputs=[connected, status, connect_btn, hand_guide_chk, compliant_chk],
        ).then(None, None, None, js="() => window.ReachyViewer.toggleRobot()")
        # hand-guide / compliant (compliant implies hand-guide)
        guide_io = dict(inputs=[hand_guide_chk, compliant_chk, connected],
                        outputs=[status, hand_guide_chk, compliant_chk])
        hand_guide_chk.change(on_hand_guide, **guide_io)
        compliant_chk.change(on_compliant, **guide_io)

        # push trajectory to the three.js viewer whenever it changes
        traj.change(None, inputs=traj, outputs=None, js="(t) => window.ReachyViewer.playMove(t)")
        picker.change(select_move, inputs=[picker, connected], outputs=[traj, info, plot])

        # the viewer self-initializes (poller in HEAD_HTML); just load the first move
        demo.load(select_move, inputs=[picker, connected], outputs=[traj, info, plot])
    return demo


if __name__ == "__main__":
    import os

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    # allow serving the vendored model bundle (URDF, meshes, Kinematics.js) + WAVs
    allowed = [str(WEB_ROOT.resolve())]
    sample_wav = ds.get_library(LIBRARY).get(ds.list_moves(LIBRARY)[0]).sound_path
    if sample_wav is not None:
        allowed.append(str(Path(sample_wav).resolve().parent))
    build().launch(
        server_name="127.0.0.1", server_port=port,
        theme=gr.themes.Soft(), allowed_paths=allowed,
    )
