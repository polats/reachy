"""Server-side controller to play moves on the physical robot (via the daemon).

Used by the Gradio app's "Connect robot" mode: while connected, selecting a move plays
it on the real robot — motion through the SDK (``async_play_move``, non-blocking) and
sound through the ALSA path (:mod:`reachy_motion.audio`), since the SDK's own audio needs
the GStreamer stack we don't have. The browser viewer separately mirrors the robot's live
state over the daemon state WebSocket.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from . import audio


class RobotController:
    def __init__(self) -> None:
        self._mini = None
        self._snd = None  # background aplay process
        self._thread = None  # background play_move thread
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._mini is not None

    def connect(self) -> None:
        """Connect an SDK client to the running daemon (raises if unavailable).

        Only the SDK handshake (needed before any control command) blocks; the camera
        media re-acquire runs in the background so switching to Connected stays snappy.
        """
        from reachy_mini import ReachyMini

        with self._lock:
            if self._mini is None:
                self._mini = ReachyMini(media_backend="no_media")
        # no_media makes the daemon RELEASE its camera/mic (shutting down the WebRTC media
        # server the browser consumes). Re-acquire it so the daemon keeps streaming the
        # camera — but in the background (it's ~1s and only the camera needs it, not control).
        mini = self._mini

        def _reacquire():
            try:
                mini.acquire_media()
            except Exception as e:  # noqa: BLE001
                print(f"[robot_control] acquire_media error: {e}")

        threading.Thread(target=_reacquire, daemon=True).start()

    def disconnect(self) -> None:
        with self._lock:
            self._stop_sound()
            mini, self._mini = self._mini, None
        if mini is not None:
            # restore normal hold so the robot isn't left limp/compliant
            for fn in ("disable_gravity_compensation", "enable_motors", "cancel_move"):
                try:
                    getattr(mini, fn)()
                except Exception:
                    pass

    def set_mode(self, hand_guide: bool, compliant: bool) -> bool:
        """Set the robot's hold mode for hand-guiding.

        - hand_guide=False           → normal position hold (``enable_motors``).
        - hand_guide=True, compliant=False → motors off / free: easiest to move by hand.
        - hand_guide=True, compliant=True  → gravity compensation: firmer, holds position
          (needs the daemon on the Placo kinematics engine).
        """
        with self._lock:
            if self._mini is None:
                return False
            try:
                if not hand_guide:
                    try:
                        self._mini.disable_gravity_compensation()
                    except Exception:
                        pass
                    self._mini.enable_motors()
                elif compliant:
                    self._mini.enable_motors()
                    self._mini.enable_gravity_compensation()
                else:
                    try:
                        self._mini.disable_gravity_compensation()
                    except Exception:
                        pass
                    self._mini.disable_motors()
                return True
            except Exception as e:  # noqa: BLE001
                print(f"[robot_control] set_mode error: {e}")
                return False

    def _stop_sound(self) -> None:
        if self._snd is not None:
            try:
                self._snd.terminate()
            except Exception:
                pass
            self._snd = None

    def goto_pose(self, head, antennas, body_yaw: float, duration: float = 1.0) -> bool:
        """Smoothly move the robot to a saved pose (head 4x4, antennas, body). Non-jerky."""
        with self._lock:
            if self._mini is None:
                return False
            try:
                self._mini.cancel_move()
            except Exception:
                pass
            self._stop_sound()
            try:
                self._mini.goto_target(head=head, antennas=antennas, body_yaw=body_yaw,
                                       duration=duration)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"[robot_control] goto_pose error: {e}")
                return False

    def goto_ready(self, duration: float = 1.0) -> bool:
        """Move the robot to the canonical ready pose (SDK INIT head + ±10° antennas).

        The blocking move runs WITHOUT holding the controller lock, so toggling
        hand-guide/compliant stays responsive while the robot eases to ready.
        """
        import numpy as np
        from reachy_mini.utils import create_head_pose

        with self._lock:
            mini = self._mini
        if mini is None:
            return False
        try:
            mini.cancel_move()
        except Exception:
            pass
        try:
            mini.enable_motors()
            mini.goto_target(head=create_head_pose(),
                             antennas=np.array([-0.1745, 0.1745]),
                             body_yaw=0.0, duration=duration)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[robot_control] goto_ready error: {e}")
            return False

    def goto_ready_async(self, duration: float = 1.0) -> None:
        """Fire-and-forget goto_ready so the UI doesn't block on the move."""
        threading.Thread(target=self.goto_ready, args=(duration,), daemon=True).start()

    def play(self, move, wav: Optional[str | Path] = None) -> None:
        """Play a move on the robot now (motion + optional sound). Non-blocking.

        Cancels any move/sound already in progress so changing the selection interrupts
        cleanly. ``play_move`` is blocking, so it runs in a background thread; cancellation
        is via the SDK's ``cancel_move()``.
        """
        with self._lock:
            mini = self._mini
            if mini is None:
                return
            try:
                mini.cancel_move()  # signal any in-flight play_move to stop
            except Exception:
                pass
            self._stop_sound()
        prev = self._thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=1.5)

        def _run():
            try:
                mini.play_move(move, sound=False, initial_goto_duration=1.0)
            except Exception as e:  # noqa: BLE001
                print(f"[robot_control] play_move error: {e}")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if wav and Path(wav).exists():
            try:
                with self._lock:
                    self._snd = audio.play_wav(wav, blocking=False)
            except Exception as e:  # noqa: BLE001
                print(f"[robot_control] sound error: {e}")
