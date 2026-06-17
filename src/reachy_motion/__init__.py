"""reachy-motion — text-to-motion + sound behaviors for Reachy Mini.

Phase 1: symbolic (procedural) motion emitted as the canonical recorded-move schema,
playable in the MuJoCo sim or on real hardware. Later phases swap a learned model in
behind the same schema-emitting interface.
"""

from .schema import MoveData
from .symbolic import CHANNELS, SymbolicMove

__all__ = ["MoveData", "SymbolicMove", "CHANNELS"]
