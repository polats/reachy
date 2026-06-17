"""Play a move from a Hugging Face moves library (downloads on first use).

    reachy-mini-daemon --sim       # in one terminal
    uv run python scripts/play_emotion.py --list
    uv run python scripts/play_emotion.py happy

Default library is pollen-robotics/reachy-mini-emotions-library (81 moves, JSON+WAV).
This is the seed corpus for the Phase-3 learned model.
"""

from __future__ import annotations

import argparse

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

DEFAULT_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="move name to play")
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--list", action="store_true", help="list available moves and exit")
    ap.add_argument("--no-sound", action="store_true")
    args = ap.parse_args()

    print(f"loading library {args.library} (downloads to HF cache on first use)...")
    moves = RecordedMoves(args.library)
    names = sorted(moves.list_moves())

    if args.list or not args.name:
        print(f"{len(names)} moves:")
        for n in names:
            print(" ", n)
        return

    if args.name not in names:
        raise SystemExit(f"'{args.name}' not found. Use --list to see options.")

    move = moves.get(args.name)
    print(f"playing '{args.name}' ({move.duration:.2f}s)...")
    mini = ReachyMini()
    mini.play_move(move, initial_goto_duration=1.0, sound=not args.no_sound)
    print("done.")


if __name__ == "__main__":
    main()
