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

from reachy_motion import dataset as ds  # noqa: E402
from reachy_motion import daemon_control  # noqa: E402
from reachy_motion import poses  # noqa: E402
from reachy_motion.robot_control import RobotController  # noqa: E402
from reachy_motion.web import (  # noqa: E402
    CAMERA_HTML,
    CHART_HTML,
    CONTAINER_HTML,
    GAMEPAD_HTML,
    HEAD_HTML,
    move_trajectory,
    pose_to_goto,
    pose_to_render,
    ready_render,
)

LIBRARY = ds.DEFAULT_LIBRARY
controller = RobotController()

# Vendored 8bitkick model bundle (URDF + per-part DRACO GLBs).
WEB_ROOT = Path(__file__).resolve().parent / "assets" / "reachy_web"
WEB_ASSETS = WEB_ROOT / "assets"
URDF_URL = f"/gradio_api/file={(WEB_ASSETS / 'reachy-mini.urdf').resolve()}"
MESH_BASE = f"/gradio_api/file={(WEB_ASSETS / 'meshes_optimized').resolve()}/"
KIN_URL = f"/gradio_api/file={(WEB_ROOT / 'src' / 'Kinematics.js').resolve()}"
IK_WASM_URL = f"/gradio_api/file={(WEB_ROOT / 'kin' / 'reachy_mini_rust_kinematics.js').resolve()}"
IK_DATA_URL = f"/gradio_api/file={(WEB_ROOT / 'kin' / 'kinematics_data.json').resolve()}"
GSTWEBRTC_URL = f"/gradio_api/file={(WEB_ROOT / 'gstwebrtc-api.js').resolve()}"


def _info_md(name: str) -> str:
    info = ds.move_info(name, LIBRARY)
    sound = "🔊 with sound" if info.has_sound else "🔇 no sound"
    return (
        f"### {info.name}\n\n{info.description}\n\n"
        f"**{info.duration:.2f}s** · **{info.num_frames} frames** · {sound}"
    )


def select_move(name: str, connected: bool):
    if not name:
        return "", ""
    data = move_trajectory(name, LIBRARY)  # includes channels for the chart
    info = _info_md(name)
    if connected:
        # play the move on the physical robot (motion + sound); the viewer mirrors the
        # live robot, so don't push browser audio (robot plays the sound). Chart still drawn.
        move = ds.get_library(LIBRARY).get(name)
        controller.play(move, move.sound_path)
        return json.dumps(data), info
    wav = ds.get_library(LIBRARY).get(name).sound_path
    if wav is not None and Path(wav).exists():
        data["audio"] = f"/gradio_api/file={Path(wav).resolve()}"
    return json.dumps(data), info


_SIM_MSG = "🖥️ **Simulator** — off-robot preview."
_CONNECTED_MSG2 = "🔌 **Connected** — Animate plays moves on the robot · Control drives it with a gamepad."


def go_simulator():
    """Simulator tab: disconnect from the robot, reset hold toggles."""
    controller.disconnect()
    return (False, _SIM_MSG, gr.update(value=False), gr.update(value=False))  # connected, status, hg, comp


def go_connected():
    """Connected tab: connect the SDK client to the daemon, start at the ready pose."""
    off = (gr.update(value=False), gr.update(value=False))  # hg, comp
    try:
        controller.connect()
        controller.goto_ready_async()  # ease to ready in the background; don't block the UI
        return (True, _CONNECTED_MSG2, *off)
    except Exception as e:  # noqa: BLE001
        return (False, f"🔴 Connect failed — is `reachy-mini-daemon` running on :8000?\n\n`{e}`", *off)


def _daemon_status_md() -> str:
    return ("🟢 **Daemon online** (`:8000`)" if daemon_control.is_up()
            else "🔴 **Daemon offline** — click Restart")


def on_restart_daemon(connected: bool):
    """Restart the daemon (Placo + camera GST_PLUGIN_PATH); reconnect if we were connected."""
    controller.disconnect()  # drop the now-stale client
    ok = daemon_control.restart()
    if not ok:
        return "🔴 **Daemon failed to start** — check `/tmp/reachy-daemon.log`"
    if connected:
        try:
            controller.connect()
            controller.goto_ready()
        except Exception:  # noqa: BLE001
            pass
    return "🟢 **Daemon restarted** (`:8000`)"


def on_enter_activity(connected: bool):
    """Switching to Control/Animate while connected: ensure motors are on for streaming/playback."""
    if connected:
        controller.set_mode(False, False)
        return gr.update(value=False), gr.update(value=False)  # clear hand-guide/compliant
    return gr.update(), gr.update()


_FREE_MSG = "🖐️ **Hand-guide · free** — easiest to move by hand; the viewer mirrors."
_COMPLIANT_MSG = "🖐️ **Hand-guide · compliant** — firmer, holds position. Move it by hand; the viewer mirrors."
_PLACO_HINT = "⚠️ **Compliant** needs the daemon on `--kinematics-engine Placo`."


def _apply(hand_guide: bool, compliant: bool):
    ok = controller.set_mode(hand_guide, compliant)
    msg = _CONNECTED_MSG2 if not hand_guide else (_COMPLIANT_MSG if compliant else _FREE_MSG)
    if not ok and compliant:
        msg = _PLACO_HINT
    return msg, gr.update(value=hand_guide), gr.update(value=compliant)


def on_hand_guide(hand_guide: bool, compliant: bool, connected: bool):
    if not connected:
        return "Connect to the robot first.", gr.update(value=False), gr.update(value=False)
    return _apply(hand_guide, compliant and hand_guide)


def on_compliant(hand_guide: bool, compliant: bool, connected: bool):
    if not connected:
        return "Connect to the robot first.", gr.update(value=False), gr.update(value=False)
    return _apply(hand_guide or compliant, compliant)


_apply_nonce = 0


def on_save_pose(capture: str):
    if not capture:
        return gr.update(), gr.update()
    try:
        data = json.loads(capture.split("|")[0])
    except Exception:  # noqa: BLE001
        return gr.update(), gr.update()
    name = poses.add_pose(data)
    choices = poses.list_poses()
    return gr.update(choices=choices, value=name), gr.update(choices=choices)


def on_select_pose(name: str, connected: bool):
    """Recall a pose: 3D always (applyPose eases if joysticking); robot via goto when connected."""
    global _apply_nonce
    pose = poses.get_pose(name) if name else None
    if not pose:
        return ""
    if connected:
        head, antennas, body = pose_to_goto(pose)
        controller.goto_pose(head, antennas, body)
    _apply_nonce += 1
    return json.dumps(pose_to_render(pose)) + "|" + str(_apply_nonce)


def on_set_l3(name: str):
    """Set which saved pose L3 resets to (persisted); push it to the browser."""
    poses.set_default(name or None)
    pose = poses.get_pose(name) if name else None
    return json.dumps(pose) if pose else ""


def on_delete_pose(name: str):
    if name:
        poses.delete_pose(name)
    choices = poses.list_poses()
    return gr.update(choices=choices, value=None), gr.update(choices=choices, value=poses.get_default())


def _default_pose_json() -> str:
    name = poses.get_default()
    pose = poses.get_pose(name) if name else None
    return json.dumps(pose) if pose else "null"


def build() -> gr.Blocks:
    names = ds.list_moves(LIBRARY)
    head = (
        f"<script>window.REACHY_URDF_URL='{URDF_URL}';"
        f"window.REACHY_MESH_BASE='{MESH_BASE}';"
        f"window.REACHY_KIN_URL='{KIN_URL}';"
        f"window.REACHY_IK_WASM_URL='{IK_WASM_URL}';"
        f"window.REACHY_IK_DATA_URL='{IK_DATA_URL}';"
        f"window.REACHY_DEFAULT_POSE={_default_pose_json()};"
        f"window.REACHY_READY={json.dumps(ready_render())};</script>\n"
        f'<script src="{GSTWEBRTC_URL}"></script>\n' + HEAD_HTML  # robot camera (WebRTC consumer)
    )
    css = """
/* dark-orange theme when controlling the real robot */
body.reachy-connected .gradio-container { background:#190d03 !important; }
body.reachy-connected .block, body.reachy-connected .form { background:#211204 !important; border-color:#7c2d12 !important; }
body.reachy-connected .tab-nav button.selected { color:#fb923c !important; border-bottom-color:#fb923c !important; }
body.reachy-connected button.primary { background:#c2410c !important; border-color:#9a3412 !important; }
body.reachy-connected h1 { color:#fb923c !important; }
"""
    with gr.Blocks(title="Reachy Mini — Move Dataset Viewer", head=head, css=css) as demo:
        gr.Markdown(
            f"# 🤖 Reachy Mini — Move Dataset Viewer\n"
            f"Browsing **{len(names)} moves** from `{LIBRARY}` as an interactive 3D robot, "
            f"driven directly from each move's recorded head/antenna/body data."
        )
        traj = gr.Textbox(visible=False)  # carries trajectory JSON to the JS viewer
        # --- pose UI hidden carriers disabled for now (kept for reuse) ---
        # pose_capture = gr.Textbox(visible=False)  # input slot; value supplied by getCurrentPose() js
        # pose_apply = gr.Textbox(visible=False)  # backend -> JS applyPose (value passed to .change js)
        # default_json = gr.Textbox(visible=False)  # backend -> JS window.REACHY_DEFAULT_POSE
        connected = gr.State(False)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():   # mode: Simulator (off-robot) <-> Connected (live robot)
                    with gr.Tab("🖥️ Simulator") as sim_tab:
                        gr.Markdown("Off-robot preview.")
                    with gr.Tab("🔌 Connected") as conn_tab:
                        gr.Markdown("Live robot.")
                        with gr.Row():
                            daemon_md = gr.Markdown(_daemon_status_md())
                            daemon_restart_btn = gr.Button("🔄 Restart daemon", scale=0, size="sm")
                        daemon_timer = gr.Timer(8.0)  # live daemon health
                        hand_guide_chk = gr.Checkbox(label="Hand-guide (move by hand)", value=False)
                        compliant_chk = gr.Checkbox(label=" ↳ Compliant (firmer; holds position)", value=False)
                status = gr.Markdown("not connected")
                with gr.Tabs():   # activity: Control (gamepad) <-> Animate (moves)
                    with gr.Tab("🎮 Control") as control_tab:
                        gr.Markdown("Connect a gamepad — **L stick** look (pan/tilt) · "
                                    "**R stick** turn body + height · **L2/L1·R2/R1** antennas · "
                                    "**L3** ready pose. FPS-style (head turns relative to the body); "
                                    "movement is bounded to the robot's reachable range so it never "
                                    "goes out of the workspace.")
                        gr.HTML(GAMEPAD_HTML)
                    with gr.Tab("🎬 Animate") as animate_tab:
                        picker = gr.Dropdown(choices=names, value=None, label="Move",
                                             filterable=True, elem_id="move-pick")
                        info = gr.Markdown()
                # --- Poses UI disabled for now (kept for reuse) ---
                # with gr.Accordion("Poses", open=True):
                #     with gr.Row():
                #         pose_dd = gr.Dropdown(choices=poses.list_poses(), value=None,
                #                               label="Go to pose", filterable=True, scale=4)
                #         del_btn = gr.Button("🗑", scale=1, min_width=44)
                #     pose_save_btn = gr.Button("💾 Save current pose  ·  or Space / R3",
                #                               elem_id="pose_save_btn", size="sm")
                #     l3_dd = gr.Dropdown(choices=poses.list_poses(), value=poses.get_default(),
                #                         label="L3 reset pose", filterable=True)
            with gr.Column(scale=2):
                gr.HTML(CONTAINER_HTML)
            with gr.Column(scale=2):
                # camera (robot's live WebRTC feed); collapse to turn it off, expand to start
                with gr.Accordion("📷 Camera", open=False):
                    gr.HTML(CAMERA_HTML)
                gr.HTML(CHART_HTML)

        # mode tabs: Simulator (off-robot) <-> Connected (live robot + dark-orange theme)
        mode_out = [connected, status, hand_guide_chk, compliant_chk]
        sim_tab.select(go_simulator, None, mode_out).then(
            None, None, None, js="() => window.ReachyViewer.setMode(false)")
        conn_tab.select(go_connected, None, mode_out).then(
            None, None, None, js="() => window.ReachyViewer.setMode(true)")
        # daemon health: live status + one-click restart (relaunches with Placo + camera plugins)
        daemon_timer.tick(lambda: _daemon_status_md(), None, daemon_md, show_progress="hidden")
        daemon_restart_btn.click(on_restart_daemon, connected, daemon_md)
        # activity tabs: entering Control/Animate while connected enables motors (clears hand-guide)
        control_tab.select(on_enter_activity, connected, [hand_guide_chk, compliant_chk])
        animate_tab.select(on_enter_activity, connected, [hand_guide_chk, compliant_chk])
        # hand-guide / compliant hold modes (Connected only)
        guide_out = [status, hand_guide_chk, compliant_chk]
        hand_guide_chk.change(on_hand_guide, [hand_guide_chk, compliant_chk, connected], guide_out)
        compliant_chk.change(on_compliant, [hand_guide_chk, compliant_chk, connected], guide_out)

        # --- pose wiring disabled for now (kept for reuse) ---
        # pose_save_btn.click(on_save_pose, inputs=pose_capture, outputs=[pose_dd, l3_dd],
        #                     js="() => window.ReachyViewer.getCurrentPose()")
        # pose_dd.change(on_select_pose, inputs=[pose_dd, connected], outputs=pose_apply)
        # pose_apply.change(None, inputs=pose_apply, outputs=None,
        #                   js="(p) => window.ReachyViewer.applyPose(p)")
        # del_btn.click(on_delete_pose, inputs=pose_dd, outputs=[pose_dd, l3_dd])
        # l3_dd.change(on_set_l3, inputs=l3_dd, outputs=default_json)
        # default_json.change(None, inputs=default_json, outputs=None,
        #                     js="(d) => { window.REACHY_DEFAULT_POSE = (d && d.length) ? JSON.parse(d) : null; }")

        # push trajectory to the three.js viewer whenever it changes
        traj.change(None, inputs=traj, outputs=None, js="(t) => window.ReachyViewer.playMove(t)")
        picker.change(select_move, inputs=[picker, connected], outputs=[traj, info])
        # no auto-load: the dropdown starts empty and the viewer shows a static neutral pose
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
