"""Watch a move in a standalone MuJoCo window — no daemon required.

This is the reliable way to *see* motion on this machine (the official
``reachy-mini-daemon --sim`` viewer crashes here due to a GStreamer/GLFW clash;
this clean process does not). It's a visualizer only — it can't drive hardware.

    # a symbolic spec (generates + views)
    uv run python scripts/view_move.py --spec examples/curious.json
    uv run python scripts/view_move.py --spec examples/curious.json --loop

    # a baked move JSON in the canonical schema
    uv run python scripts/view_move.py --move out/curious.json

    # a move from the HF emotions library (downloads on first use)
    uv run python scripts/view_move.py --emotion happy

For motion on the real robot (or the official sim), use scripts/play_symbolic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reachy_motion.schema import MoveData  # noqa: E402
from reachy_motion.symbolic import SymbolicMove  # noqa: E402
from reachy_motion.viewer import view  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="symbolic spec JSON to generate and view")
    g.add_argument("--move", help="baked move JSON (canonical schema) to view")
    g.add_argument("--emotion", help="move name from --library to view")
    ap.add_argument("--library", default="pollen-robotics/reachy-mini-emotions-library")
    ap.add_argument("--scene", default="scene", help="MJCF scene name (scene/empty/minimal)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-realtime", action="store_true", help="play as fast as possible")
    args = ap.parse_args()

    if args.spec:
        move = SymbolicMove(json.loads(Path(args.spec).read_text()))
        label = move.description
    elif args.move:
        move = MoveData.load(args.move)
        label = move.description
    else:
        from reachy_mini.motion.recorded_move import RecordedMoves

        print(f"loading library {args.library}...")
        move = RecordedMoves(args.library).get(args.emotion)
        label = args.emotion

    print(f"viewing '{label}' ({move.duration:.2f}s) — close the window to exit")
    view(move, scene=args.scene, loop=args.loop, realtime=not args.no_realtime)
    print("done.")
    # The passive MuJoCo/GLFW viewer can hit a TLS assertion during its atexit
    # teardown on some Wayland setups. All work is finished here, so exit hard to
    # skip that buggy cleanup and guarantee a clean exit code.
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
