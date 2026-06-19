"""On-disk store of authored animation specs (behaviors.json).

Seeded from the code-authored builtins (anim.BUILTINS) on first use; edits/new behaviors
from the authoring UI are saved here. Each value is a spec (see anim.py): segments + layers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import anim

_FILE = Path(__file__).resolve().parents[2] / "behaviors.json"
_lock = threading.Lock()


def _load() -> dict:
    if _FILE.exists():
        try:
            d = json.loads(_FILE.read_text())
            if isinstance(d, dict) and d:
                return d
        except Exception:
            pass
    d = anim.builtin_specs()      # seed from code on first run
    _write(d)
    return d


def _write(d: dict) -> None:
    _FILE.write_text(json.dumps(d, indent=1))


def list_behaviors() -> list[str]:
    return list(_load().keys())


def get(name: str) -> dict | None:
    return _load().get(name)


def save(name: str, spec: dict) -> None:
    with _lock:
        d = _load()
        d[name] = spec
        _write(d)


def delete(name: str) -> None:
    with _lock:
        d = _load()
        d.pop(name, None)
        _write(d)


def unique_name(base: str = "behavior") -> str:
    d = _load()
    i = 1
    while f"{base} {i}" in d:
        i += 1
    return f"{base} {i}"
