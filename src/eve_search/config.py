"""Validated mission configuration (YAML -> Pydantic) for EVE Search Agent.

The configuration describes the *simulation* (hidden ground truth included).
Only the environment is allowed to read the hidden sections (``evidence.eve``,
clue placements...); the agent consumes the mission spec for sensor/cost
parameters that WALL-E would plausibly know, and nothing else.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .models import EVE_STATUSES, Orientation

Coordinate = tuple[int, int]


class StrictModel(BaseModel):
    """Base model: reject unknown fields so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class MissionMeta(StrictModel):
    id: str = Field(min_length=1)
    description: str = ""
    confidence_threshold: float = Field(default=0.35, gt=0.0, le=1.0)


class WorldConfig(StrictModel):
    width: int = Field(gt=0, le=200)
    height: int = Field(gt=0, le=200)
    seed: int = 42
    obstacle_density: float = Field(default=0.10, ge=0.0, le=1.0)
    hazard_density: float = Field(default=0.0, ge=0.0, le=1.0)
    obstacles: Optional[list[Coordinate]] = None
    hazards: Optional[list[Coordinate]] = None
    recharge_stations: Optional[list[Coordinate]] = None

    def _check(self, coord: Sequence[int], field: str) -> None:
        x, y = coord
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{field} coordinate {(x, y)} outside {self.width}x{self.height} map")


class WalleConfig(StrictModel):
    start: Coordinate
    orientation: Orientation = Orientation.EAST

    @model_validator(mode="after")
    def _valid(self) -> "WalleConfig":
        x, y = self.start
        if x < 0 or y < 0:
            raise ValueError("WALL-E start coordinates must be non-negative")
        return self


class InitialSignalConfig(StrictModel):
    """The distress-signal fragment WALL-E starts the mission with.

    ``coordinates`` may be omitted: then the agent only has a bearing (and,
    when available, a strength reading) to triangulate from. If ALL fields
    are omitted the environment generates a truthful noisy fragment from its
    (hidden) ground truth at mission start.
    """

    strength: Optional[float] = Field(default=None, ge=0.0)
    coordinates: Optional[Coordinate] = None
    bearing_deg: Optional[float] = Field(default=None, ge=0.0, lt=360.0)
    timestamp: Optional[str] = None

    @model_validator(mode="after")
    def _needs_something(self) -> "InitialSignalConfig":
        if (
            self.coordinates is None
            and self.bearing_deg is None
            and self.strength is None
            and self.timestamp is None
        ):
            raise ValueError(
                "initial signal must include coordinates, a bearing/strength "
                "estimate, or a timestamp"
            )
        return self


class SignalPropagation(StrictModel):
    """Propagation model parameters (public knowledge, not hidden state)."""

    reference_strength: float = Field(default=100.0, gt=0.0)
    decay_length: float = Field(default=9.0, gt=0.0)
    noise_sigma: float = Field(default=6.0, gt=0.0)
    bearing_noise_deg: float = Field(default=10.0, gt=0.0)


class SignalConfig(StrictModel):
    initial: InitialSignalConfig
    propagation: SignalPropagation = SignalPropagation()


class ResourcesConfig(StrictModel):
    battery_capacity: float = Field(gt=0.0)
    initial_battery: float = Field(gt=0.0)
    mission_time: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _battery_within_capacity(self) -> "ResourcesConfig":
        if self.initial_battery > self.battery_capacity:
            raise ValueError("initial_battery cannot exceed battery_capacity")
        return self


class CostsConfig(StrictModel):
    move_battery: float = Field(default=1.2, ge=0.0)
    move_time: float = Field(default=1.5, gt=0.0)
    rotate_battery: float = Field(default=0.1, ge=0.0)
    rotate_time: float = Field(default=0.5, gt=0.0)
    scan_battery: float = Field(default=3.0, ge=0.0)
    scan_time: float = Field(default=4.0, gt=0.0)
    signal_ping_battery: float = Field(default=0.5, ge=0.0)
    signal_ping_time: float = Field(default=0.5, gt=0.0)
    wait_battery: float = Field(default=0.0, ge=0.0)
    wait_time: float = Field(default=1.0, gt=0.0)
    recharge_battery_gain: float = Field(default=18.0, gt=0.0)
    recharge_time: float = Field(default=6.0, gt=0.0)
    blocked_move_penalty_time: float = Field(default=0.5, ge=0.0)


class EvidenceConfig(StrictModel):
    """Hidden-world generation parameters for clues and EVE herself."""

    clue_count: int = Field(default=5, ge=1, le=50)
    clue_spread: float = Field(default=9.0, gt=0.0)
    scan_radius: int = Field(default=3, ge=1, le=20)
    seed: Optional[int] = None  # defaults to world.seed
    scan_collection_radius: int = Field(default=1, ge=0, le=5)


class EveConfig(StrictModel):
    """HIDDEN ground truth. Never handed to the agent."""

    status: str
    position: Optional[Coordinate] = None  # derived from seed when None

    @field_validator("status")
    @classmethod
    def _known_status(cls, v: str) -> str:
        v = v.lower()
        if v not in EVE_STATUSES:
            raise ValueError(f"unknown EVE status {v!r}; expected one of {EVE_STATUSES}")
        return v


class MissionSpec(StrictModel):
    """Root of the validated mission configuration."""

    mission: MissionMeta
    world: WorldConfig
    walle: WalleConfig
    signal: SignalConfig
    resources: ResourcesConfig
    costs: CostsConfig = CostsConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    eve: EveConfig

    @model_validator(mode="after")
    def _validate_layout(self) -> "MissionSpec":
        world = self.world
        start_x, start_y = self.walle.start
        if not (0 <= start_x < world.width and 0 <= start_y < world.height):
            raise ValueError("WALL-E start position outside map bounds")

        named_lists = (
            ("obstacles", world.obstacles),
            ("hazards", world.hazards),
            ("recharge_stations", world.recharge_stations),
        )
        for field, coords in named_lists:
            if coords is None:
                continue
            for c in coords:
                self.world._check(c, field)

        obstructed = set(map(tuple, world.obstacles or []))
        for c in world.hazards or []:
            if tuple(c) in obstructed:
                raise ValueError(f"hazard {tuple(c)} sits on an obstacle cell")
        stations = [tuple(c) for c in world.recharge_stations or []]
        for c in stations:
            if c in obstructed:
                raise ValueError(f"recharge station {c} sits on an obstacle cell")
            if tuple(c) == tuple(self.walle.start):
                raise ValueError(f"recharge station {c} coincides with WALL-E start")

        if self.eve.position is not None:
            self.world._check(self.eve.position, "eve.position")
            if tuple(self.eve.position) == tuple(self.walle.start):
                raise ValueError("EVE cannot start exactly on WALL-E's tile")
        if stations and len(stations) != len(set(stations)):
            raise ValueError("duplicate recharge stations")
        return self

    @property
    def rng_seed(self) -> int:
        return self.world.seed


def load_mission(path: str | Path) -> MissionSpec:
    """Load and validate a YAML mission configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mission config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"mission config {path} must be a YAML mapping")
    return MissionSpec.model_validate(raw)


def derive_eve_position(spec: MissionSpec) -> Coordinate:
    """Deterministic EVE placement when the config omits explicit coordinates."""
    if spec.eve.position is not None:
        return tuple(spec.eve.position)
    rng = random.Random(spec.rng_seed * 7919 + 13)
    w, h = spec.world.width, spec.world.height
    sx, sy = spec.walle.start
    for _ in range(500):
        x = rng.randrange(w)
        y = rng.randrange(h)
        # keep EVE away from WALL-E's tile and map corners for an interesting search
        if (x, y) != (sx, sy) and (x, y) not in {(0, 0), (w - 1, h - 1), (w - 1, 0), (0, h - 1)}:
            return (x, y)
    return (w // 2, h // 2)
