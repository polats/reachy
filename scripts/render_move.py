"""Render a move to a preview video (MP4/GIF) — the reliable way to *see* a move here.

    uv run python scripts/render_move.py --spec examples/curious.json -o out/curious.mp4
    uv run python scripts/render_move.py --move out/curious.json -o out/curious.gif
    uv run python scripts/render_move.py --emotion happy -o out/happy.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reachy_motion.render import render  # noqa: E402
from reachy_motion.schema import MoveData  # noqa: E402
from reachy_motion.symbolic import SymbolicMove  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="symbolic spec JSON")
    g.add_argument("--move", help="baked move JSON (canonical schema)")
    g.add_argument("--emotion", help="move name from --library")
    ap.add_argument("-o", "--out", required=True, help="output .mp4 or .gif")
    ap.add_argument("--library", default="pollen-robotics/reachy-mini-emotions-library")
    ap.add_argument("--scene", default="scene")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if args.spec:
        move = SymbolicMove(json.loads(Path(args.spec).read_text()))
    elif args.move:
        move = MoveData.load(args.move)
    else:
        from reachy_mini.motion.recorded_move import RecordedMoves

        print(f"loading library {args.library}...")
        move = RecordedMoves(args.library).get(args.emotion)

    print(f"rendering ({move.duration:.2f}s) -> {args.out} ...")
    out = render(
        move, args.out, scene=args.scene, fps=args.fps, width=args.width, height=args.height
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
