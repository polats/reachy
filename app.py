"""Gradio dataset viewer for Reachy Mini moves — interactive 3D.

Browse a moves library (default: the 81-move emotions library) as an interactive 3D
robot (orbit / zoom / scrub timeline) driven directly from each move's head 4x4 +
antenna + body-yaw data, alongside its description and per-DOF channel plots.

    uv run python app.py            # http://127.0.0.1:7861
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from reachy_motion import anim  # noqa: E402
from reachy_motion import audio_monitor  # noqa: E402
from reachy_motion import behaviors_store  # noqa: E402
from reachy_motion import conversation as convo  # noqa: E402
from reachy_motion import dataset as ds  # noqa: E402
from reachy_motion import daemon_control  # noqa: E402
from reachy_motion import poses  # noqa: E402
from reachy_motion import transcribe  # noqa: E402
from reachy_motion import tts  # noqa: E402
from reachy_motion.robot_control import RobotController  # noqa: E402
from reachy_motion.web import (  # noqa: E402
    AUDIO_HTML,
    CAMERA_HTML,
    CHART_HTML,
    CONTAINER_HTML,
    GAMEPAD_HTML,
    HEAD_HTML,
    POSE_PANEL_HTML,
    TIMELINE_HTML,
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
VIEWER_JS_URL = f"/gradio_api/file={(WEB_ROOT / 'viewer.js').resolve()}"  # the Three.js viewer (was inline)


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
    _connected[0] = False
    return (False, _SIM_MSG, gr.update(value=False), gr.update(value=False))  # connected, status, hg, comp


def on_enter_author():
    """Authoring is simulator-only — drop any robot connection so it's never 'Connected'."""
    return go_simulator()


def go_connected():
    """Connected tab: connect the SDK client to the daemon, start at the ready pose."""
    off = (gr.update(value=False), gr.update(value=False))  # hg, comp
    try:
        controller.connect()
        controller.goto_ready_async()  # ease to ready in the background; don't block the UI
        _connected[0] = True
        return (True, _CONNECTED_MSG2, *off)
    except Exception as e:  # noqa: BLE001
        _connected[0] = False
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


_audio_ctr = itertools.count()


_connected = [False]  # set by go_connected/go_simulator (stable; not a flickery Timer input)
_voice_on = [False]   # set by the Voice accordion via on_set_voice


def on_set_voice(flag: str):
    """Frontend tells us whether the Voice accordion is expanded (loop on/off)."""
    _voice_on[0] = (str(flag) == "1")


def audio_tick():
    """Run the voice loop only while wanted: connected AND the Voice accordion expanded.

    Reads stable module flags (not Timer inputs — a gr.State read every 0.12s flickers here).
    """
    active = _connected[0] and _voice_on[0]
    if not active:
        if convo.conversation.running:
            convo.conversation.stop()
        if transcribe.transcriber.running:
            transcribe.transcriber.stop()
        if audio_monitor.monitor.running:
            audio_monitor.monitor.stop()
        return ""  # clears the indicator/waveform/transcript
    audio_monitor.monitor.start()
    transcribe.transcriber.start()
    convo.conversation.start()
    snap = audio_monitor.monitor.snapshot()
    snap.update(transcribe.transcriber.snapshot())   # interim, stt_ready
    snap.update(convo.conversation.snapshot())        # dialogue, speaking
    snap["n"] = next(_audio_ctr)  # vary the value so .change always fires
    return json.dumps(snap)


def on_enter_activity(connected: bool):
    """Switching to Control/Animate while connected: ensure motors are on for streaming/playback."""
    if connected:
        controller.set_mode(False, False)
        return gr.update(value=False), gr.update(value=False)  # clear hand-guide/compliant
    return gr.update(), gr.update()


# ===== Animation authoring (simulator-only: previews via the 3D viewer, never the robot) =====
# Editable columns of the segment table: duration + ease + the expressive channels.
# (x,y head-shift are omitted from the table — rarely used; kept 0 in the spec.)
EDIT_CH = ("z", "roll", "pitch", "yaw", "antL", "antR", "body")
SEG_COLS = ["dur", "ease"] + list(EDIT_CH)
SEG_TYPES = ["number", "str"] + ["number"] * len(EDIT_CH)
EASE_HINT = "ease: " + ", ".join(anim.EASES.keys())


def _rows(df):
    if df is None:
        return []
    if hasattr(df, "values"):
        return df.values.tolist()
    return list(df)


def _spec_from_ui(df, layers) -> dict:
    segs = []
    for r in _rows(df):
        if r is None or r[0] in (None, ""):
            continue
        try:
            dur = float(r[0])
        except (TypeError, ValueError):
            continue
        ease = str(r[1] or "smooth").strip()
        pose = {c: 0.0 for c in anim.CHANNELS}
        for i, c in enumerate(EDIT_CH):
            try:
                pose[c] = float(r[2 + i] or 0.0)
            except (TypeError, ValueError):
                pose[c] = 0.0
        segs.append({"dur": dur, "ease": ease, "pose": pose})
    return {"fps": anim.FPS, "segments": segs,
            "layers": [{"type": l} for l in (layers or [])]}


def _ui_from_spec(spec: dict):
    rows = []
    for seg in spec.get("segments", []):
        p = seg.get("pose", {})
        rows.append([seg.get("dur", 0.5), seg.get("ease", "smooth")]
                    + [round(float(p.get(c, 0.0)), 2) for c in EDIT_CH])
    layers = [l["type"] for l in spec.get("layers", []) if l.get("type") in anim.LAYERS]
    return rows, layers


def author_preview(df, layers):
    """Push the animation SPEC (not a baked trajectory) to the viewer.

    The browser bakes it client-side (pose interpolation + WASM IK, per frame on its render
    loop — the same path live gamepad control uses). This keeps the wire payload tiny
    (hundreds of bytes vs an 18-60KB baked trajectory) and moves the per-frame IK off the
    server, so editing/browsing no longer ships big payloads through Gradio's serial queue.
    Empty (no segments) -> "" so the viewer falls back to the static ready pose.
    """
    spec = _spec_from_ui(df, layers)
    return json.dumps(spec) if spec.get("segments") else ""


# --- Phase-1 authoring: the viewer owns the spec client-side; the server only loads/saves it ---
def author_load(name):
    """Load a behavior into the 3D editor: set the name + life-layer checkboxes, and push the
    spec JSON to the hidden carrier (whose .change calls viewer.loadSpec)."""
    spec = behaviors_store.get(name)
    if not spec:
        return gr.update(), gr.update(), gr.update()
    layers = [l["type"] for l in spec.get("layers", []) if l.get("type") in anim.LAYERS]
    return name, layers, json.dumps(spec)


def author_new_ui():
    """New behavior: clear the name + layers (the .then JS resets the editor to one keyframe)."""
    return "", []


def _parse_spec(spec_json: str) -> dict:
    try:
        spec = json.loads(spec_json) if spec_json else None
    except Exception:  # noqa: BLE001
        spec = None
    if not isinstance(spec, dict) or not spec.get("segments"):
        return {}
    return spec


def _name_and_spec(payload: str):
    """Save/Dup js sends one JSON blob {name, spec} (spec is itself a JSON string) — single input
    + single js return is the reliable Gradio pattern (multi-value js→inputs is flaky)."""
    try:
        d = json.loads(payload) if payload else {}
    except Exception:  # noqa: BLE001
        d = {}
    return (d.get("name") or "").strip(), _parse_spec(d.get("spec") or "")


def author_save(payload):
    name, spec = _name_and_spec(payload)
    if not spec:
        return gr.update(), "⚠️ add a keyframe before saving"
    name = name or behaviors_store.unique_name()
    behaviors_store.save(name, spec)
    return gr.update(choices=behaviors_store.list_behaviors(), value=name), f"💾 saved **{name}**"


def author_duplicate(payload):
    name, spec = _name_and_spec(payload)
    if not spec:
        return gr.update(), gr.update(), "⚠️ nothing to duplicate"
    new = behaviors_store.unique_name(f"{name or 'behavior'} copy")
    behaviors_store.save(new, spec)
    return gr.update(choices=behaviors_store.list_behaviors(), value=new), new, f"⧉ duplicated to **{new}**"


def author_delete(name):
    if name:
        behaviors_store.delete(name)
    return gr.update(choices=behaviors_store.list_behaviors(), value=None), "🗑 deleted"


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
        f'<script src="{GSTWEBRTC_URL}"></script>\n'        # robot camera (WebRTC consumer)
        + HEAD_HTML                                          # three.js importmap (must be inline)
        # cache-bust the module by file mtime so viewer.js edits always reload (it's a fixed URL)
        + f'\n<script type="module" src="{VIEWER_JS_URL}?_v={int((WEB_ROOT / "viewer.js").stat().st_mtime)}"></script>'
    )
    css = """
/* dark-orange theme when controlling the real robot */
body.reachy-connected .gradio-container { background:#190d03 !important; }
body.reachy-connected .block, body.reachy-connected .form { background:#211204 !important; border-color:#7c2d12 !important; }
body.reachy-connected .tab-nav button.selected { color:#fb923c !important; border-bottom-color:#fb923c !important; }
body.reachy-connected button.primary { background:#c2410c !important; border-color:#9a3412 !important; }
body.reachy-connected h1 { color:#fb923c !important; }
.reachy-hidden { display:none !important; }   /* in the DOM (JS-clickable) but not shown */
/* Author tab: taller preview so the character is easy to see while posing */
#author-viewer-slot #reachy3d { height: 480px !important; }
/* Top Play/Author panes: hide by collapsing height (NOT display:none) so the segments table
   keeps its width even when its pane is hidden — display:none gives it 0 width and the
   Dataframe's TanStack column logic recurses (stack overflow) when it's shown. */
.pane-hidden { height:0 !important; min-height:0 !important; overflow:hidden !important;
               margin:0 !important; padding:0 !important; opacity:0 !important; pointer-events:none !important; }
/* tab-bar buttons styled as tabs (elem_id lands on the <button>, which carries .maintab) */
#top-tabbar { gap:2px; border-bottom:1px solid rgba(255,255,255,0.12); margin-bottom:10px; }
button.maintab {
  background:transparent !important; border:none !important; border-bottom:2px solid transparent !important;
  border-radius:0 !important; box-shadow:none !important; color:rgba(255,255,255,0.55) !important;
  font-weight:600 !important; font-size:15px !important; }
button.maintab.tab-active { color:#fff !important; border-bottom-color:#3b82f6 !important; }
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
        # Top-level Play / Author panes. NOT gr.Tabs: an inactive gr.Tab is display:none, which
        # gives the segments Dataframe 0 width and trips a TanStack column-grouping recursion when
        # it's shown. These panes hide via height-collapse (.pane-hidden), so the table keeps its
        # width even when hidden (the same reason the old authoring Accordion never hit this).
        with gr.Row(elem_id="top-tabbar"):
            play_tab_btn = gr.Button("▶ Play", elem_id="tab-play", elem_classes=["maintab", "tab-active"])
            author_tab_btn = gr.Button("✏️ Author", elem_id="tab-author", elem_classes=["maintab"])

        with gr.Column(elem_id="play-pane"):
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
                with gr.Column(scale=2):
                    gr.HTML('<div id="play-viewer-home">' + CONTAINER_HTML + '</div>')
                with gr.Column(scale=2):
                    # camera (robot's live WebRTC feed); collapse to turn it off, expand to start
                    with gr.Accordion("📷 Camera", open=False):
                        gr.HTML(CAMERA_HTML)
                    # voice loop (mic/VAD/STT + repeat/TTS): runs only while this is expanded (like camera)
                    with gr.Accordion("🎙 Voice", open=False):
                        gr.HTML(AUDIO_HTML)
                        voice_dd = gr.Dropdown(tts.list_voices(), value=tts.current_voice(),
                                               label="Reachy's voice", filterable=True)
                    audio_json = gr.Textbox(visible=False)  # backend mic/dialogue snapshot -> JS pushAudio
                    audio_timer = gr.Timer(0.12)
                    voice_in = gr.Textbox(visible=False)    # carries voiceWanted() '1'/'0' to on_set_voice
                    # CSS-hidden (not visible=False) so it stays in the DOM and JS can click it
                    voice_btn = gr.Button("voice", elem_id="voice-set-btn", elem_classes=["reachy-hidden"])
                    gr.HTML('<div id="play-chart-home">' + CHART_HTML + '</div>')

        # Author pane (starts collapsed): controls left, smaller live preview right, channels below.
        # The single 3D viewer + channels chart relocate in here (JS) while it's open, then back.
        with gr.Column(elem_id="author-pane", elem_classes=["pane-hidden"]):
            gr.HTML('<div id="reachy-author"></div>')   # marker for the viewer's playback gate
            gr.Markdown("Pick a **Behavior** to edit (or **New**). **Pose** the robot in 3D and "
                        "press **＋ Keyframe** to build the animation — each keyframe eases from the "
                        "one before it. ▶ Preview plays it; **Save** keeps it.")
            author_spec_in = gr.Textbox(visible=False)   # server -> JS: load a spec into the editor
            author_save_in = gr.Textbox(visible=False)   # JS -> server: {name, spec} for Save/Dup
            with gr.Row():
                with gr.Column(scale=3):   # author controls (left): pick / pose / keyframe / actions
                    with gr.Row():
                        behavior_dd = gr.Dropdown(behaviors_store.list_behaviors(), value=None,
                                                  label="Behavior", filterable=True, scale=3)
                        name_tb = gr.Textbox(label="Name", scale=3, elem_id="author-name")
                        new_btn = gr.Button("➕ New", scale=1, min_width=64)
                        dup_btn = gr.Button("⧉ Dup", scale=1, min_width=64)
                        del_btn2 = gr.Button("🗑", scale=0, min_width=44)
                    gr.HTML(POSE_PANEL_HTML)   # pose sliders + ＋ Keyframe (driven by viewer.js)
                    with gr.Row():
                        layers_chk = gr.CheckboxGroup(["breath", "ear_idle"], value=[],
                                                      label="Life layers", scale=3)
                        preview_btn = gr.Button("▶ Preview", variant="primary", scale=1)
                        save_btn = gr.Button("💾 Save", scale=1)
                    author_msg = gr.Markdown()
                with gr.Column(scale=2):   # smaller live preview (the relocated 3D viewer)
                    gr.HTML('<div id="author-viewer-slot"></div>')
            gr.HTML(TIMELINE_HTML)   # keyframe chips (driven by viewer.js)
            gr.HTML('<div id="author-chart-slot"></div>')   # channels viewer at the bottom

        # mode tabs: Simulator (off-robot) <-> Connected (live robot + dark-orange theme)
        mode_out = [connected, status, hand_guide_chk, compliant_chk]
        sim_tab.select(go_simulator, None, mode_out).then(
            None, None, None, js="() => window.ReachyViewer.setMode(false)")
        conn_tab.select(go_connected, None, mode_out).then(
            None, None, None, js="() => window.ReachyViewer.setMode(true)")
        # daemon health: live status + one-click restart (relaunches with Placo + camera plugins)
        daemon_timer.tick(lambda: _daemon_status_md(), None, daemon_md, show_progress="hidden")
        daemon_restart_btn.click(on_restart_daemon, connected, daemon_md)
        # robot mic: poll levels while connected, render voice indicator + waveform
        audio_timer.tick(audio_tick, None, audio_json, show_progress="hidden")
        # the JS poll clicks voice_btn on accordion-toggle; its js reports voiceWanted() -> on_set_voice
        voice_btn.click(on_set_voice, voice_in, None,
                        js="() => (window.ReachyViewer.voiceWanted() ? '1' : '0')")
        voice_dd.change(lambda v: tts.set_voice(v), voice_dd, None)   # pick Reachy's TTS voice
        audio_json.change(None, audio_json, None,
                          js="(j) => window.ReachyViewer.pushAudio(j)")
        # top tabs (custom panes): show one, collapse the other, and relocate the viewer + chart
        play_tab_btn.click(None, None, None, js="""() => {
            document.getElementById('play-pane').classList.remove('pane-hidden');
            document.getElementById('author-pane').classList.add('pane-hidden');
            document.getElementById('tab-play').classList.add('tab-active');
            document.getElementById('tab-author').classList.remove('tab-active');
            window.ReachyViewer.relocate(false);
        }""")
        # Entering Author: authoring is simulator-only, so force Simulator mode (disconnect the
        # robot via the sub-tab) and the blue theme before relocating the viewer in.
        author_tab_btn.click(on_enter_author, None, [connected, status, hand_guide_chk, compliant_chk],
                             js="""() => {
            const sim = [...document.querySelectorAll('button[role=tab]')].find(b => /Simulator/.test(b.textContent));
            if (sim && !sim.classList.contains('selected')) sim.click();   // -> go_simulator (disconnect)
            document.getElementById('author-pane').classList.remove('pane-hidden');
            document.getElementById('play-pane').classList.add('pane-hidden');
            document.getElementById('tab-author').classList.add('tab-active');
            document.getElementById('tab-play').classList.remove('tab-active');
            window.ReachyViewer.relocate(true);
            window.ReachyViewer.setMode(false);   // blue theme + simulator viewer
        }""")
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

        # --- authoring (Phase 1): the viewer owns the spec; the server only loads/saves it ---
        # Editing (pose-drag, sliders, ＋Keyframe, timeline) is 100% client-side in viewer.js — no
        # round-trips while authoring. Gradio handles only: load a behavior into the editor, Save,
        # New, Dup, Delete, and mirroring the life-layer checkboxes into the client spec.
        # Load: server reads the stored spec -> hidden carrier -> viewer.loadSpec().
        behavior_dd.change(author_load, behavior_dd, [name_tb, layers_chk, author_spec_in],
                           show_progress="hidden")
        author_spec_in.change(None, author_spec_in, None,
                              js="(s) => window.ReachyViewer.loadSpec(s)")
        # life layers: toggle -> push into the client spec and re-bake (no server work)
        layers_chk.change(None, layers_chk, None, js="(l) => window.ReachyViewer.setLayers(l)")
        # New: clear name + layers, then reset the editor to a single fresh keyframe
        new_btn.click(author_new_ui, None, [name_tb, layers_chk]).then(
            None, None, None, js="() => window.ReachyViewer.newSpec()")
        preview_btn.click(None, None, None, js="() => window.ReachyViewer.replay()")
        # Save / Dup: js packs {name, spec} from the DOM + viewer into the single hidden carrier
        _pack_js = ("() => JSON.stringify({name: (document.querySelector('#author-name textarea,"
                    "#author-name input')||{}).value || '', spec: window.ReachyViewer.getEditSpec()})")
        save_btn.click(author_save, author_save_in, [behavior_dd, author_msg], js=_pack_js)
        dup_btn.click(author_duplicate, author_save_in, [behavior_dd, name_tb, author_msg], js=_pack_js)
        del_btn2.click(author_delete, behavior_dd, [behavior_dd, author_msg])
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
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=port,
        theme=gr.themes.Soft(), allowed_paths=allowed,
    )
