"""Build a symbolic move from a spec JSON, bake it, and play it in the sim.

    reachy-mini-daemon --sim          # in one terminal (opens the viewer)
    uv run python scripts/play_symbolic.py examples/curious.json

With --save, also writes the baked move to the canonical recorded-move schema
(out/<name>.json) so it can be played by RecordedMove or pushed to an HF dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reachy_motion.player import play  # noqa: E402
from reachy_motion.symbolic import SymbolicMove  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="path to a symbolic spec JSON (see examples/)")
    ap.add_argument("--fps", type=float, default=100.0, help="bake sampling rate")
    ap.add_argument("--no-play", action="store_true", help="bake only, do not connect")
    ap.add_argument("--save", metavar="PATH", help="also save baked move JSON here")
    ap.add_argument("--wav", metavar="PATH", help="WAV to play in sync on the robot speaker")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    move = SymbolicMove(spec)
    baked = move.bake(fps=args.fps)
    print(f"baked '{baked.description}' — {baked.num_frames} frames, {baked.duration:.2f}s")

    if args.save:
        out = baked.save(args.save)
        print(f"saved -> {out}")

    if not args.no_play:
        print("connecting to daemon and playing..." + (f" (sound: {args.wav})" if args.wav else ""))
        play(move, wav=args.wav)
        print("done.")


if __name__ == "__main__":
    main()
