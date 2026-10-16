"""Simulated environment: grid world, hidden ground truth, public observations.

Key design rule: **ground truth is private to this module.** EVE's true
coordinates, the clue placements and her status are stored in underscore
attributes and are never returned by any public method. Agents see the world
only through :class:`~eve_search.models.Observation` objects attached to an
:class:`~eve_search.models.ActionResult`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from .config import MissionSpec, derive_eve_position
from .models import (
    REGION_NAMES,
    ActionKind,
    ActionResult,
    CellType,
    Clue,
    ClueKind,
    Observation,
    ObservationKind,
    Orientation,
    angular_distance_deg,
    bearing_between,
    grid_distance,
    region_name,
    wrap_to_box,
)


REGION_NAMES = (
    "north-west", "north", "north-east",
    "west", "centre", "east",
    "south-west", "south", "south-east",
)


@dataclass
class _HiddenState:
    """Ground truth. The agent must never touch this."""

    eve_position: tuple[int, int]
    eve_status: str
    clue_placements: list[tuple[tuple[int, int], Clue]] = dc_field(default_factory=list)
    obstacles: set[tuple[int, int]] = dc_field(default_factory=set)
    hazards: set[tuple[int, int]] = dc_field(default_factory=set)
    stations: list[tuple[int, int]] = dc_field(default_factory=list)

    def clue_marker_positions(self) -> set[tuple[int, int]]:
        return {pos for pos, _ in self.clue_placements}


class Environment:
    """Deterministic simulated world driven by a validated :class:`MissionSpec`."""

    def __init__(self, spec: MissionSpec, seed: Optional[int] = None):
        self.spec = spec
        self.width = spec.world.width
        self.height = spec.world.height
        self.oracle_seed = seed if seed is not None else spec.rng_seed
        self._rng = np.random.default_rng(self.oracle_seed)

        self._hidden = self._build_world()

        self._walle_pos: tuple[int, int] = tuple(spec.walle.start)
        self._orientation: Orientation = spec.walle.orientation
        self._battery: float = float(spec.resources.initial_battery)
        self._time_left: float = float(spec.resources.mission_time)
        self._step_count: int = 0
        self._finished: bool = False
        self._build_initial_briefing()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def _build_world(self) -> _HiddenState:
        spec = self.spec
        rng = np.random.default_rng(self.oracle_seed)
        hidden = _HiddenState(
            eve_position=tuple(derive_eve_position(spec)),
            eve_status=spec.eve.status,
        )

        # Obstacles ------------------------------------------------------
        if spec.world.obstacles is not None:
            obstacles = {tuple(c) for c in spec.world.obstacles}
        else:
            obstacles = set()
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) != self._walle_start and rng.random() < spec.world.obstacle_density:
                        obstacles.add((x, y))
        hidden.obstacles = obstacles

        # Hazards --------------------------------------------------------
        if spec.world.hazards is not None:
            hazards = {tuple(c) for c in spec.world.hazards}
        else:
            hazards = set()
            for x in range(self.width):
                for y in range(self.height):
                    if (
                        (x, y) not in obstacles
                        and (x, y) != self._walle_start
                        and rng.random() < spec.world.hazard_density
                    ):
                        hazards.add((x, y))
        hidden.hazards = hazards

        # Recharge stations ---------------------------------------------
        if spec.world.recharge_stations is not None:
            stations = [tuple(c) for c in spec.world.recharge_stations]
        else:
            stations = self._generate_stations(rng, obstacles, hazards)
        hidden.stations = stations

        # EVE must not be stuck inside an obstacle ------------------------
        evp = hidden.eve_position
        if evp in obstacles:
            for dx in range(1, self.width):
                cand = ((evp[0] + dx) % self.width, evp[1])
                if cand not in obstacles:
                    hidden.eve_position = cand
                    break

        # Clues: truthful evidence packets scattered near EVE --------------
        self._place_clues(hidden)
        return hidden

    def _build_initial_briefing(self) -> None:
        """Construct the mission-start signal fragment.

        A config-supplied bearing/strength fragment is used as-is; any missing
        measurement is generated truthfully from (hidden) ground truth        with the configured noise, so the briefing is always physically coherent.
        The fragment is *data WALL-E receives*, never raw hidden state.
        """
        init = self.spec.signal.initial
        prop = self.spec.signal.propagation
        briefing_rng = np.random.default_rng(self.oracle_seed * 31 + 5)

        coordinates = init.coordinates
        bearing = init.bearing_deg
        strength = init.strength

        if strength is None:
            d = grid_distance(self._walle_start, self._hidden.eve_position)
            mu = self.expected_strength(d)
            strength = float(np.clip(mu + briefing_rng.normal(0, prop.noise_sigma), 0.0, prop.reference_strength))
        if bearing is None and coordinates is None:
            true_b = bearing_between(self._walle_start, self._hidden.eve_position)
            bearing = (true_b + float(briefing_rng.normal(0, prop.bearing_noise_deg))) % 360.0

        self._briefing: dict = {
            "coordinates": coordinates,
            "bearing_deg": bearing,
            "strength": strength,
            "timestamp": init.timestamp,
        }

    def mission_briefing(self) -> dict:
        """The distress-signal fragment WALL-E starts with (observations only)."""
        return dict(self._briefing)

    @property
    def _walle_start(self) -> tuple[int, int]:
        return tuple(self.spec.walle.start)

    def _generate_stations(
        self, rng: np.random.Generator, obstacles: set, hazards: set
    ) -> list[tuple[int, int]]:
        sx, sy = self._walle_start
        candidates = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in obstacles and (x, y) not in hazards and (x, y) != self._walle_start
        ]
        if not candidates:
            candidates = [(self.width // 2, self.height // 2)]
        rng.shuffle(candidates)
        stations: list[tuple[int, int]] = []
        for cand in candidates:
            if all(grid_distance(cand, s) >= 4 for s in stations):
                stations.append(cand)
            if len(stations) >= 3:
                break
        return stations

    def _place_clues(self, hidden: _HiddenState) -> None:
        spec = self.spec
        ev = spec.evidence
        seed = ev.seed if ev.seed is not None else self.oracle_seed
        rng = random.Random(seed)
        evp = hidden.eve_position

        pool = [
            ClueKind.RING,
            ClueKind.BEARING,
            ClueKind.REGION,
            ClueKind.BEARING,
            ClueKind.RING,
            ClueKind.STATUS,
            ClueKind.REGION,
        ]
        kinds = [pool[i % len(pool)] for i in range(ev.clue_count)]
        rng.shuffle(kinds)

        w, h = self.width, self.height
        placed: list[tuple[tuple[int, int], Clue]] = []
        used_stations = set(hidden.stations)
        for idx, kind in enumerate(kinds):
            for _ in range(300):
                ang = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(1.5, ev.clue_spread)
                cx = int(round(evp[0] + dist * math.cos(ang)))
                cy = int(round(evp[1] + dist * math.sin(ang)))
                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))
                if (
                    (cx, cy) in hidden.obstacles
                    or (cx, cy) in used_stations
                    or (cx, cy) == self._walle_start
                ):
                    continue
                if any(pos == (cx, cy) for pos, _ in placed):
                    continue
                clue = self._build_clue(kind, idx, (cx, cy), evp, hidden, rng)
                placed.append(((cx, cy), clue))
                break
        hidden.clue_placements = placed

    def _build_clue(
        self,
        kind: ClueKind,
        idx: int,
        pickup: tuple[int, int],
        eve: tuple[int, int],
        hidden: _HiddenState,
        rng: random.Random,
    ) -> Clue:
        cid = f"clue-{idx + 1:02d}"
        if kind is ClueKind.RING:
            true_d = grid_distance(pickup, eve)
            slack = rng.uniform(0.75, 1.6)
            radius = round(max(1.0, true_d * slack), 1)
            return Clue(
                id=cid,
                kind=kind,
                text=f"Thermal residue max ~{radius:g} units from here.",
                payload={"radius": radius},
            )
        if kind is ClueKind.BEARING:
            bearing = bearing_between(pickup, eve)
            sigma = self.spec.signal.propagation.bearing_noise_deg
            noisy = (bearing + rng.gauss(0, sigma)) % 360.0
            return Clue(
                id=cid,
                kind=kind,
                text=f"Vector coupling bearing ~{noisy:.0f} deg (+/-{sigma:g} deg).",
                payload={"bearing_deg": round(noisy, 1), "sigma_deg": round(sigma, 1)},
            )
        if kind is ClueKind.REGION:
            region = self._region_of(eve)
            return Clue(
                id=cid,
                kind=kind,
                text=f"Boot logs hive-cluster around the {region} sector.",
                payload={"region": region},
            )
        if kind is ClueKind.STATUS:
            return Clue(
                id=cid,
                kind=kind,
                text=f"Signature waveform suggests status: {hidden.eve_status}.",
                payload={"status": hidden.eve_status},
            )
        raise AssertionError(f"unhandled clue kind {kind}")  # pragma: no cover

    def _region_of(self, pos: tuple[int, int]) -> str:
        return region_name(pos[0], pos[1], self.width, self.height)

    # ------------------------------------------------------------------
    # Public read-only accessors the agent may rely on
    # ------------------------------------------------------------------
    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def current_position(self) -> tuple[int, int]:
        return self._walle_pos

    @property
    def current_orientation(self) -> Orientation:
        return self._orientation

    @property
    def battery(self) -> float:
        return self._battery

    @property
    def remaining_time(self) -> float:
        return self._time_left

    def cell_is_station(self, pos: tuple[int, int]) -> bool:
        """True only for the tile WALL-E currently occupies (near-by sensing)."""
        return pos == self._walle_pos and pos in self._hidden.stations

    def signal_physics(self) -> dict[str, float]:
        """Propagation parameters WALL-E's engineering team knows (not hidden state)."""
        p = self.spec.signal.propagation
        return {
            "reference_strength": p.reference_strength,
            "decay_length": p.decay_length,
            "noise_sigma": p.noise_sigma,
            "bearing_noise_deg": p.bearing_noise_deg,
        }

    def evaluation_only_map(self) -> dict:
        """EVALUATION-ONLY ground-truth dump for the labelled visualization.

        Never consumed by the agent, never part of observations or traces.
        """
        return {
            "eve_position": list(self._hidden.eve_position),
            "eve_status": self._hidden.eve_status,
            "obstacles": [list(c) for c in self._hidden.obstacles],
            "hazards": [list(c) for c in self._hidden.hazards],
            "stations": [list(c) for c in self._hidden.stations],
            "clue_positions": [list(c) for c in self._hidden.clue_marker_positions()],
        }

    # ------------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------------
    def apply_action(self, action: ActionKind, params: Optional[dict] = None) -> ActionResult:
        params = params or {}
        if self._finished:
            raise RuntimeError("mission already finished; no further actions")
        self._step_count += 1
        step = self._step_count
        observations: list[Observation] = []
        success = True
        reason = ""

        if action is ActionKind.MOVE_FORWARD:
            success, reason, observations = self._do_move()
        elif action in (ActionKind.TURN_LEFT, ActionKind.TURN_RIGHT):
            success, reason, observations = self._do_rotate(action)
        elif action is ActionKind.SCAN:
            observations = self._do_scan()
        elif action is ActionKind.SIGNAL_PING:
            observations = self._do_signal_ping()
        elif action is ActionKind.WAIT:
            self._time_left -= self.spec.costs.wait_time
            observations.append(
                self._obs(ObservationKind.STATUS, "Standing by to conserve systems.")
            )
        elif action is ActionKind.RECHARGE:
            success, reason, observations = self._do_recharge()
        else:  # pragma: no cover
            raise ValueError(f"unsupported action {action}")

        if self._time_left <= 0:
            self._time_left = max(self._time_left, 0.0)
            self._finished = True
        if self._battery <= 0:
            self._battery = max(self._battery, 0.0)
            self._finished = True

        return ActionResult(
            step=step,
            action=action,
            params=params,
            success=success,
            observations=observations,
            position=self._walle_pos,
            orientation=self._orientation,
            battery=round(self._battery, 3),
            remaining_time=round(self._time_left, 3),
            reason=reason,
        )

    # -- individual actions ----------------------------------------------
    def _obs(
        self,
        kind: ObservationKind,
        message: str,
        data: Optional[dict] = None,
        pos: Optional[tuple[int, int]] = None,
    ) -> Observation:
        return Observation(
            kind=kind,
            message=message,
            position=pos or self._walle_pos,
            orientation=self._orientation,
            battery=round(self._battery, 3),
            remaining_time=round(self._time_left, 3),
            data=data or {},
        )

    def _do_move(self):
        costs = self.spec.costs
        self._time_left -= costs.move_time
        dx, dy = self._orientation.delta
        target = (self._walle_pos[0] + dx, self._walle_pos[1] + dy)
        if not wrap_to_box(target[0], target[1], self.width, self.height):
            msg = f"Boundary hit heading {self._orientation.value}."
            self._time_left -= costs.blocked_move_penalty_time
            obs = [self._obs(ObservationKind.BLOCKED, msg, {"target": list(target)})]
            return False, msg, obs
        if target in self._hidden.obstacles:
            msg = "Obstacle blocks the move."
            self._time_left -= costs.blocked_move_penalty_time
            obs = [self._obs(ObservationKind.BLOCKED, msg, {"target": list(target)})]
            return False, msg, obs

        self._walle_pos = target
        self._battery -= costs.move_battery
        notes = [
            self._obs(
                ObservationKind.STATUS,
                f"Moved to {target} facing {self._orientation.value}.",
                {"moved_to": list(target)},
            )
        ]
        if target in self._hidden.hazards:
            drain = self.spec.resources.battery_capacity * 0.10
            self._battery -= drain
            notes.append(
                self._obs(
                    ObservationKind.STATUS,
                    f"Hazard damaged power cell; drained {drain:.1f}% battery.",
                    {"hazard": True, "drain": round(drain, 2)},
                )
            )
        return True, "", notes

    def _do_rotate(self, action: ActionKind):
        costs = self.spec.costs
        self._time_left -= costs.rotate_time
        self._battery -= costs.rotate_battery
        self._orientation = (
            self._orientation.turn_left() if action is ActionKind.TURN_LEFT
            else self._orientation.turn_right()
        )
        obs = [self._obs(ObservationKind.STATUS, f"Now facing {self._orientation.value}.")]
        return True, "", obs

    def _do_scan(self) -> list[Observation]:
        spec = self.spec
        ev = spec.evidence
        self._time_left -= spec.costs.scan_time
        self._battery -= spec.costs.scan_battery
        radius = ev.scan_radius
        pos = self._walle_pos

        revealed: dict[str, str] = {}
        for x in range(max(0, pos[0] - radius), min(self.width, pos[0] + radius + 1)):
            for y in range(max(0, pos[1] - radius), min(self.height, pos[1] + radius + 1)):
                revealed[f"{x},{y}"] = self._surface_type((x, y)).value

        notes = [
            self._obs(
                ObservationKind.TERRAIN_SCAN,
                f"Local scan complete r={radius}: {len(revealed)} cells mapped.",
                {"cells": revealed, "scan_radius": radius},
            )
        ]

        coll = ev.scan_collection_radius
        for cpos, clue in list(self._hidden.clue_placements):
            cheb = max(abs(cpos[0] - pos[0]), abs(cpos[1] - pos[1]))
            if cheb <= coll:
                notes.append(
                    self._obs(
                        ObservationKind.CLUE_DISCOVERED,
                        f"Collected evidence {clue.id}: {clue.text}",
                        {"clue": clue.model_dump()},
                        pos=cpos,
                    )
                )
                self._hidden.clue_placements.remove((cpos, clue))
        return notes

    def _do_signal_ping(self) -> list[Observation]:
        prop = self.spec.signal.propagation
        self._time_left -= self.spec.costs.signal_ping_time
        self._battery -= self.spec.costs.signal_ping_battery

        d = grid_distance(self._walle_pos, self._hidden.eve_position)
        mu = self.expected_strength(d)
        strength = float(
            np.clip(mu + self._rng.normal(0.0, prop.noise_sigma), 0.0, prop.reference_strength)
        )
        true_bearing = bearing_between(self._walle_pos, self._hidden.eve_position)
        bearing = (true_bearing + float(self._rng.normal(0.0, prop.bearing_noise_deg))) % 360.0

        return [
            self._obs(
                ObservationKind.SIGNAL_MEASUREMENT,
                f"Signal ping: strength {strength:.1f}, bearing {bearing:.0f} deg.",
                {"strength": round(strength, 2), "bearing_deg": round(bearing, 1)},
            )
        ]

    def _do_recharge(self):
        costs = self.spec.costs
        if not self.cell_is_station(self._walle_pos):
            msg = "No recharge station at this tile."
            return False, msg, [self._obs(ObservationKind.BLOCKED, msg)]
        self._time_left -= costs.recharge_time
        before = self._battery
        self._battery = min(
            self.spec.resources.battery_capacity,
            self._battery + costs.recharge_battery_gain,
        )
        gained = self._battery - before
        obs = [
            self._obs(
                ObservationKind.RECHARGE_REPORT,
                f"Recharged +{gained:.1f}% battery.",
                {"gained": round(gained, 2)},
            )
        ]
        return True, "", obs

    # ------------------------------------------------------------------
    # Signal physics (public knowledge, reusable by the agent)
    # ------------------------------------------------------------------
    def expected_strength(self, distance: float) -> float:
        """Mean received strength at ``distance`` under the configured model."""
        prop = self.spec.signal.propagation
        return prop.reference_strength * math.exp(-distance / prop.decay_length)

    def invert_strength(self, strength: float) -> float:
        """Distance implied by a (noiseless) strength reading."""
        prop = self.spec.signal.propagation
        s = min(max(strength, 1e-6), prop.reference_strength)
        return -prop.decay_length * math.log(s / prop.reference_strength)

    # ------------------------------------------------------------------
    # Per-cell surface typing (used by scans; not a bulk ground-truth dump)
    # ------------------------------------------------------------------
    def _surface_type(self, pos: tuple[int, int]) -> CellType:
        if pos in self._hidden.obstacles:
            return CellType.OBSTACLE
        if pos in self._hidden.hazards:
            return CellType.HAZARD
        if pos in self._hidden.clue_marker_positions():
            return CellType.CLUE_MARKER
        if pos in self._hidden.stations:
            return CellType.RECHARGE_STATION
        return CellType.OPEN
