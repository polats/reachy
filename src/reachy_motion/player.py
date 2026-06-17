"""Connect to a Reachy Mini daemon (sim or real) and play moves.

Usage assumes a daemon is already running, e.g. for simulation::

    reachy-mini-daemon --sim            # opens a MuJoCo viewer window
    reachy-mini-daemon --sim --headless # no window (for testing)

``ReachyMini()`` auto-detects the local daemon on ``localhost:8000``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move

from .schema import MoveData


def play(
    move: Union[Move, MoveData],
    *,
    mini: Optional[ReachyMini] = None,
    play_frequency: float = 100.0,
    initial_goto_duration: float = 1.0,
    wav: Optional[str | Path] = None,
    media_backend: str = "no_media",
) -> None:
    """Play a move on a connected (or freshly connected) Reachy Mini, with optional sound.

    Accepts either a ``reachy_mini`` :class:`Move` (e.g. a ``SymbolicMove`` or
    ``RecordedMove``) or our :class:`~reachy_motion.schema.MoveData` (auto-adapted).

    Sound: if ``wav`` is given (or a ``MoveData`` carries a ``sound_path``), it is played
    on the robot speaker via :mod:`reachy_motion.audio` (direct ALSA), started in sync
    with the move's motion — independent of the SDK media stack (which needs the
    GStreamer webrtc plugin and isn't required for motion). ``media_backend`` stays
    ``"no_media"`` so the motion path never touches GStreamer.
    """
    if wav is None and isinstance(move, MoveData) and move.sound_path is not None:
        wav = move.sound_path

    if isinstance(move, MoveData):
        move = move.to_recorded_move()

    def _do(m: ReachyMini) -> None:
        # Pre-position to the move's start pose so sound and motion can start together
        # (play_move's own initial_goto would otherwise delay motion vs. sound).
        if wav is not None:
            from . import audio

            head0, ant0, by0 = move.evaluate(0.0)
            m.goto_target(head=head0, antennas=ant0, body_yaw=by0 or 0.0,
                          duration=initial_goto_duration)
            audio.play_wav(wav, blocking=False)
            m.play_move(move, play_frequency=play_frequency, initial_goto_duration=0.0,
                        sound=False)
        else:
            m.play_move(move, play_frequency=play_frequency,
                        initial_goto_duration=initial_goto_duration, sound=False)

    if mini is not None:
        _do(mini)
    else:
        # ReachyMini is a context manager (no close() method); use it for cleanup.
        with ReachyMini(media_backend=media_backend) as owned:
            _do(owned)


def play_move_file(path: str | Path, **kwargs) -> None:
    """Load a canonical move JSON (+ optional sibling .wav) and play it."""
    play(MoveData.load(path), **kwargs)
