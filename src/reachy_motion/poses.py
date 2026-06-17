"""A small on-disk 'poses' dataset — named single poses you can recall on the robot.

A pose is the command-frame vector the joystick / set_target use::

    {x, y, z, roll, pitch, yaw, antL, antR, body}   # metres / radians

Stored in poses.json at the project root::

    {"default": <name or null>, "poses": {name: pose, ...}}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

_FILE = Path(__file__).resolve().parent.parent.parent / "poses.json"
_lock = threading.Lock()

KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "antL", "antR", "body")


def _load() -> dict:
    if _FILE.exists():
        try:
            d = json.loads(_FILE.read_text())
            d.setdefault("default", None)
            d.setdefault("poses", {})
            return d
        except Exception:
            pass
    return {"default": None, "poses": {}}


def _write(d: dict) -> None:
    _FILE.write_text(json.dumps(d, indent=2))


def list_poses() -> list[str]:
    return list(_load()["poses"].keys())


def get_pose(name: str) -> Optional[dict]:
    return _load()["poses"].get(name)


def get_default() -> Optional[str]:
    return _load().get("default")


def add_pose(pose: dict, name: Optional[str] = None) -> str:
    """Add a pose (auto-named 'pose N' if no name). Returns the name used."""
    clean = {k: float(pose.get(k, 0.0)) for k in KEYS}
    with _lock:
        d = _load()
        if not name:
            i = 1
            while f"pose {i}" in d["poses"]:
                i += 1
            name = f"pose {i}"
        d["poses"][name] = clean
        _write(d)
    return name


def set_default(name: Optional[str]) -> None:
    with _lock:
        d = _load()
        d["default"] = name if (name and name in d["poses"]) else None
        _write(d)


def delete_pose(name: str) -> None:
    with _lock:
        d = _load()
        d["poses"].pop(name, None)
        if d.get("default") == name:
            d["default"] = None
        _write(d)
