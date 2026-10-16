"""Shared Pydantic models used across the EVE Search Agent project.

This module defines the *public* observation interface: everything WALL-E's
agent can ever see flows through :class:`Observation` and
:class:`ActionResult`. Ground truth (EVE's true coordinates, hidden clue
placements, hidden status) lives only inside the environment and is never
exposed through these models.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field

EVE_STATUSES: tuple[str, ...] = ("dormant", "standby", "active", "damaged")


class Orientation(str, Enum):
    """WALL-E's heading on the compass rose. North is decreasing y."""

    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    @property
    def delta(self) -> tuple[int, int]:
        return _ORIENTATION_DELTA[self.value]

    def turn_left(self) -> "Orientation":
        return Orientation(CARDINAL_ORDER[(CARDINAL_ORDER.index(self.value) - 1) % 4])

    def turn_right(self) -> "Orientation":
        return Orientation(CARDINAL_ORDER[(CARDINAL_ORDER.index(self.value) + 1) % 4])


_ORIENTATION_DELTA: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

CARDINAL_ORDER: tuple[str, ...] = ("N", "E", "S", "W")


class CellType(str, Enum):
    """Terrain/cell categories revealed by local scans."""

    UNKNOWN = "unknown"
    OPEN = "open"
    OBSTACLE = "obstacle"
    HAZARD = "hazard"
    RECHARGE_STATION = "recharge_station"
    CLUE_MARKER = "clue_marker"


class ActionKind(str, Enum):
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SCAN = "scan"
    SIGNAL_PING = "signal_ping"
    WAIT = "wait"
    RECHARGE = "recharge"


class ObservationKind(str, Enum):
    TERRAIN_SCAN = "terrain_scan"          # cells revealed within scan radius
    SIGNAL_MEASUREMENT = "signal_measurement"  # noisy strength (+bearing)
    CLUE_DISCOVERED = "clue_discovered"    # an evidence packet picked up
    BLOCKED = "blocked"                    # move/rotate attempt failed
    RECHARGE_REPORT = "recharge_report"    # battery altered on a station
    STATUS = "status"                      # internal bookkeeping note


class Observation(BaseModel):
    """An atomic piece of information delivered to the agent.

    Observations NEVER contain EVE's true coordinates or hidden status.
    """

    kind: ObservationKind
    message: str
    position: tuple[int, int]  # WALL-E's position where the observation was made
    orientation: Orientation
    battery: float
    remaining_time: float
    data: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Result of a single action applied to the environment."""

    step: int
    action: ActionKind
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool
    observations: list[Observation] = Field(default_factory=list)
    position: tuple[int, int]
    orientation: Orientation
    battery: float
    remaining_time: float
    reason: str = ""


class ClueKind(str, Enum):
    REGION = "region"      # qualitative area hint ("north", "centre", ...)
    RING = "ring"          # max distance from the clue pickup point
    BEARING = "bearing"    # noisy compass bearing from the pickup point
    STATUS = "status"      # hint about EVE's status


class Clue(BaseModel):
    """An evidence packet WALL-E collects in the field.

    Payload keys depend on ``kind``. The environment constructs clue payloads
    truthfully from hidden ground truth, but the *clue itself* -- once inside
    an Observation -- is in-domain knowledge the agent may combine freely.
    """

    id: str
    kind: ClueKind
    text: str
    pickup: Optional[tuple[int, int]] = None  # set when collected
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceEntry(BaseModel):
    """One row of the JSONL action trace."""

    step: int
    sim_time: float
    position: tuple[int, int]
    orientation: Orientation
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool
    observations: list[str]  # short human-readable summaries
    battery: float
    remaining_time: float
    belief_summary: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    next_objective: str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

REGION_NAMES: tuple[str, ...] = (
    "north-west", "north", "north-east",
    "west", "centre", "east",
    "south-west", "south", "south-east",
)


def region_name(x: int, y: int, width: int, height: int) -> str:
    """Qualitative 3x3 compass region of a map position (public vocabulary)."""
    col = 0 if x < width / 3 else (1 if x < 2 * width / 3 else 2)
    row = 0 if y < height / 3 else (1 if y < 2 * height / 3 else 2)
    return REGION_NAMES[row * 3 + col]

def grid_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two grid points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Chebyshev (king-move) distance, useful for sensor radii."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bearing_between(from_xy: Sequence[float], to_xy: Sequence[float]) -> float:
    """Compass bearing in degrees from ``from_xy`` to ``to_xy`` (0 = North)."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]  # dy negative = northwards
    deg = math.degrees(math.atan2(dx, -dy))
    return deg % 360.0


def angular_distance_deg(a: float, b: float) -> float:
    """Smallest unsigned angle between two bearings, in degrees."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def wrap_to_box(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


try:  # pragma: no cover - exercised in unit tests
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None
