"""Autonomous WALL-E search agent.

The agent is a genuine closed-loop decision maker:
* it maintains a Bayesian belief over EVE's location/status,
* it selects actions to maximise expected information per unit battery/time,
* it plans paths with A* over its *known* map (never ground truth),
* it keeps a battery reserve for reaching a recharge station or safety,
* it replans when observations contradict expectations, and
* it records an honest JSONL decision trace while it acts (no post-hoc
  rationalisation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

from .belief import BeliefState
from .config import MissionSpec
from .environment import Environment
from .models import (
    ActionKind,
    ActionResult,
    Clue,
    ObservationKind,
    Orientation,
    TraceEntry,
    grid_distance,
)
from .planner import PathPlan, astar, _orient_from_delta


@dataclass
class Decision:
    action: ActionKind
    params: dict = dc_field(default_factory=dict)
    rationale: str = ""
    next_objective: str = ""


def _rotations_between(current: Orientation, required: Orientation) -> list[ActionKind]:
    """Minimal rotation actions to reach ``required`` heading."""
    order = [o.value for o in Orientation]
    ci = order.index(current.value)
    ri = order.index(required.value)
    diff = (ri - ci) % 4
    if diff == 0:
        return []
    if diff <= 2:
        return [ActionKind.TURN_RIGHT] * diff
    return [ActionKind.TURN_LEFT] * (4 - diff)


class SearchAgent:
    """Resource-aware autonomous search agent."""

    def __init__(self, env: Environment, spec: MissionSpec, max_steps: int = 4000):
        self.env = env
        self.spec = spec
        self.max_steps = max_steps

        # --- agent-side knowledge (built ONLY from observations) ----------
        self.known_cells: set[tuple[int, int]] = {env.current_position}
        self.known_obstacles: set[tuple[int, int]] = set()
        self.known_hazards: set[tuple[int, int]] = set()
        self.known_stations: set[tuple[int, int]] = set()
        self.pending_clue_markers: set[tuple[int, int]] = set()

        self.belief = BeliefState(spec.world.width, spec.world.height)
        physics = env.signal_physics()
        briefing = env.mission_briefing()
        self.belief.apply_initial_signal(
            coordinates=briefing.get("coordinates"),
            bearing_deg=briefing.get("bearing_deg"),
            strength=briefing.get("strength"),
            from_pos=env.current_position,
            physics=physics,
        )

        self.collected_clues: list[Clue] = []
        self.ping_positions: list[tuple[int, int]] = []
        self.scan_positions: list[tuple[int, int]] = []
        self.path_history: list[tuple[int, int]] = [env.current_position]
        self.visit_counts: dict[tuple[int, int], int] = {env.current_position: 1}
        self.recent_positions: list[tuple[int, int]] = [env.current_position]
        self._current_target: Optional[tuple[int, int]] = None
        self.consecutive_rotations: int = 0
        self.consecutive_scans: int = 0
        self.stuck_counter: int = 0
        self.trace: list[TraceEntry] = []

        self.steps_since_ping = 0
        self.moves_since_ping = 0
        self.moves_since_scan = 0
        self.distance_traveled = 0
        self.successful_moves = 0
        self.blocked_moves = 0
        self.scans_done = 0
        self.pings_done = 0
        self.recharges_done = 0
        self.mode = "explore"
        self.outcome: Optional[str] = None
        self._pending_post_clue_ping = False
        self._confirmed = False

    # ------------------------------------------------------------------
    # Observation ingestion (the ONLY channel into agent knowledge)
    # ------------------------------------------------------------------
    def _observe(self, result: ActionResult) -> None:
        for obs in result.observations:
            if obs.kind is ObservationKind.TERRAIN_SCAN:
                for key, val in obs.data.get("cells", {}).items():
                    x, y = (int(v) for v in key.split(","))
                    self.known_cells.add((x, y))
                    if val == "obstacle":
                        self.known_obstacles.add((x, y))
                    elif val == "hazard":
                        self.known_hazards.add((x, y))
                    elif val == "recharge_station":
                        self.known_stations.add((x, y))
                        self.pending_clue_markers.discard((x, y))
                    elif val == "clue_marker":
                        self.pending_clue_markers.add((x, y))
                    else:
                        # open terrain / obstacle / hazard: definitely not a marker
                        self.pending_clue_markers.discard((x, y))
            elif obs.kind is ObservationKind.SIGNAL_MEASUREMENT:
                strength = obs.data.get("strength")
                bearing = obs.data.get("bearing_deg")
                if strength is not None:
                    self.belief.apply_signal(
                        float(strength),
                        float(bearing) if bearing is not None else None,
                        obs.position,
                        self.env.signal_physics(),
                    )
                self.pings_done += 1
                self._pending_post_clue_ping = False
            elif obs.kind is ObservationKind.CLUE_DISCOVERED:
                clue = Clue.model_validate(obs.data["clue"])
                self.collected_clues.append(clue)
                if clue.pickup:
                    self.pending_clue_markers.discard(tuple(clue.pickup))
                self.belief.apply_clue(clue)
                self._pending_post_clue_ping = True
            elif obs.kind is ObservationKind.BLOCKED:
                target = obs.data.get("target")
                if target:
                    t = (int(target[0]), int(target[1]))
                    if 0 <= t[0] < self.env.width and 0 <= t[1] < self.env.height:
                        self.known_obstacles.add(t)

    # ------------------------------------------------------------------
    # Planning helpers (agent's own knowledge only)
    # ------------------------------------------------------------------
    def _plan_to(
        self,
        target: tuple[int, int],
        from_pos: Optional[tuple[int, int]] = None,
        penalize_revisit: bool = True,
        prefer_unexplored: bool = False,
    ) -> PathPlan:
        frm = from_pos or self.env.current_position
        v_counts = self.visit_counts if penalize_revisit else None
        r_cells = self.recent_positions if penalize_revisit else None
        return astar(
            frm,
            target,
            self.env.width,
            self.env.height,
            known_obstacles=self.known_obstacles,
            known_hazards=self.known_hazards,
            known_cells=self.known_cells,
            visit_counts=v_counts,
            recent_cells=r_cells,
            prefer_unexplored=prefer_unexplored,
        )

    def _plan_leg_time(self, plan: PathPlan) -> float:
        c = self.spec.costs
        return len(plan.path) * c.move_time + plan.turn_count * c.rotate_time

    def _plan_leg_battery(self, plan: PathPlan) -> float:
        c = self.spec.costs
        return len(plan.path) * c.move_battery + plan.turn_count * c.rotate_battery

    def _nearest_station_plan(self, pos: Optional[tuple[int, int]] = None) -> Optional[PathPlan]:
        """Cheapest feasible A* plan to a known recharge station."""
        if pos is None:
            pos = self.env.current_position
        best: Optional[PathPlan] = None
        for st in self.known_stations:
            plan = self._plan_to(st, pos, penalize_revisit=False)
            if plan.feasible and (best is None or plan.path_cost < best.path_cost):
                best = plan
        return best

    def _reserve_battery(self) -> float:
        """Battery kept aside to guarantee a safe stop (station or start)."""
        cap = self.spec.resources.battery_capacity
        plan = self._nearest_station_plan()
        if plan is not None and plan.feasible:
            return self._plan_leg_battery(plan) * 1.25 + 2.0
        return max(0.25 * cap, 25.0)

    def _reserve_time(self) -> float:
        plan = self._nearest_station_plan()
        c = self.spec.costs
        if plan is not None and plan.feasible:
            return self._plan_leg_time(plan) + c.recharge_time
        return 15.0

    def _exploration_target(self) -> Optional[tuple[int, int]]:
        """Pick the frontier cell maximising (coverage + belief mass) minus cost."""
        pos = self.env.current_position
        radius = self.spec.evidence.scan_radius
        frontier: set[tuple[int, int]] = set()
        for (x, y) in self.known_cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if (
                    nxt not in self.known_cells
                    and nxt not in self.known_obstacles
                    and 0 <= nxt[0] < self.env.width
                    and 0 <= nxt[1] < self.env.height
                ):
                    frontier.add(nxt)
        if not frontier:
            return None

        # Stick to current target while moving toward it to avoid orientation flutter
        if (
            self._current_target is not None
            and self._current_target in frontier
            and self.stuck_counter == 0
            and self.consecutive_rotations < 2
        ):
            current_plan = self._plan_to(self._current_target, prefer_unexplored=True)
            if current_plan.feasible and len(current_plan.path) >= 2:
                return self._current_target

        est = self.belief.map_estimate()
        belief_map = self.belief.posterior()
        best: Optional[tuple[float, tuple[int, int]]] = None

        candidates = sorted(frontier, key=lambda c: grid_distance(c, pos))[:100]

        for cand in candidates:
            plan = self._plan_to(cand, prefer_unexplored=True)
            if not plan.feasible:
                continue
            new_cells = 0
            for cx in range(cand[0] - radius, cand[0] + radius + 1):
                for cy in range(cand[1] - radius, cand[1] + radius + 1):
                    if (
                        (cx, cy) not in self.known_cells
                        and 0 <= cx < self.env.width
                        and 0 <= cy < self.env.height
                    ):
                        new_cells += 1

            path_revisit_penalty = 0.0
            for p in plan.path[:-1]:
                vc = self.visit_counts.get(p, 0)
                if vc > 1:
                    path_revisit_penalty += (vc - 1) * 0.4
                if p in self.recent_positions[-8:]:
                    path_revisit_penalty += 0.5

            bmass = float(belief_map[cand[1], cand[0]])
            score = (
                1.5 * new_cells
                + 500.0 * bmass
                - 0.05 * grid_distance(cand, est)
                - 0.25 * (plan.path_cost + 0.5 * plan.turn_count)
                - 0.4 * path_revisit_penalty
            )
            if best is None or score > best[0]:
                best = (score, cand)

        chosen = best[1] if best else None
        self._current_target = chosen
        return chosen

    # ------------------------------------------------------------------
    # The decision function (one action at a time)
    # ------------------------------------------------------------------
    def _choose_action(self) -> Optional[Decision]:
        if self.env.is_finished:
            return None
        pos = self.env.current_position
        ori = self.env.current_orientation
        costs = self.spec.costs
        bat = self.env.battery
        tl = self.env.remaining_time
        cap = self.spec.resources.battery_capacity

        reserve_b = self._reserve_battery()
        reserve_t = self._reserve_time()
        at_station = self.env.cell_is_station(pos)

        # -- 1) Rescue mode: preserve safe recovery capability ----------
        if not at_station and bat <= reserve_b + max(4.0, 0.05 * cap):
            plan = self._nearest_station_plan()
            if plan is not None and plan.feasible and self._leg_affordable(plan, reserve_b, reserve_t):
                nxt = self._follow_plan_step(plan, pos, ori)
                if nxt is not None:
                    self.mode = "rescue"
                    nxt.rationale = (
                        f"battery {bat:.1f}% is at/below reserve {reserve_b:.1f}%; "
                        f"diverting to known recharge station (path cost {plan.path_cost:.1f})"
                    )
                    nxt.next_objective = f"recharge at station {plan.path[-1]}"
                    return nxt
            if (plan is None or not plan.feasible) and bat <= 0.30 * cap:
                plan_home = self._plan_to(self.spec.walle.start, penalize_revisit=False)
                if (
                    plan_home.feasible
                    and plan_home.path[-1] != pos
                    and self._leg_affordable(plan_home, 5.0, reserve_t)
                ):
                    nxt = self._follow_plan_step(plan_home, pos, ori)
                    if nxt is not None:
                        self.mode = "rescue"
                        nxt.rationale = (
                            f"no station charted and battery {bat:.1f}%; regrouping at mission start"
                        )
                        nxt.next_objective = "return to start and scan for stations"
                        return nxt

        if at_station and bat < 0.94 * cap and tl > reserve_t + costs.recharge_time and bat < 0.98 * cap:
            self.mode = "rescue"
            self.recharges_done += 1
            return Decision(
                ActionKind.RECHARGE,
                params={"repeat": len([e for e in self.trace if e.action == "recharge"])},
                rationale=(
                    f"on station; battery {bat:.1f}/{cap:.0f}%; topping up while time admits "
                    f"(reserve {reserve_b:.1f}%)"
                ),
                next_objective="resume search with restored battery",
            )

        # -- 2) Convergence: verify the MAP estimate -------------------
        threshold = self.spec.mission.confidence_threshold
        if self.belief.max_posterior() >= threshold:
            self.mode = "verify"
            est = self.belief.map_estimate()
            if grid_distance(pos, est) <= 2.0:
                if self._confirmed:
                    self.outcome = "located"
                    return None
                self._confirmed = True
                return Decision(
                    ActionKind.SIGNAL_PING,
                    rationale=(
                        f"posterior {self.belief.max_posterior():.3f} >= threshold "
                        f"{threshold}: confirming proximity at MAP estimate {est}"
                    ),
                    next_objective="final confirmation ping, then mission report",
                )
            plan = self._plan_to(est, penalize_revisit=False)
            leg_b = self._plan_leg_battery(plan)
            leg_t = self._plan_leg_time(plan)
            if plan.feasible and bat - reserve_b - leg_b > 0 and tl - reserve_t - leg_t > 0:
                nxt = self._follow_plan_step(plan, pos, ori)
                nxt.rationale = (
                    f"posterior {self.belief.max_posterior():.3f} >= threshold "
                    f"{threshold}: closing on MAP estimate {est} (leg battery {leg_b:.1f}%)"
                )
                nxt.next_objective = f"reach {est} to verify"
                return nxt
            self.outcome = "located_unverified"
            return None

        # -- 2b) Pending clue markers outrank generic frontiers ------------
        if self.pending_clue_markers:
            coll = self.spec.evidence.scan_collection_radius
            near = [
                m for m in self.pending_clue_markers
                if max(abs(m[0] - pos[0]), abs(m[1] - pos[1])) <= coll
                and tl > costs.scan_time
                and bat > costs.scan_battery
            ]
            if near:
                mark = min(near, key=lambda m: (m[0] - pos[0]) ** 2 + (m[1] - pos[1]) ** 2)
                return Decision(
                    ActionKind.SCAN,
                    rationale=(
                        f"standing within reach of clue marker at {mark}; scanning "
                        "now to lift the evidence packet"
                    ),
                    next_objective=f"collect evidence at {mark}",
                )
            best: Optional[tuple[float, tuple[int, int]]] = None
            for mark in sorted(self.pending_clue_markers):
                plan = self._plan_to(mark, penalize_revisit=False)
                if (
                    plan.feasible
                    and len(plan.path) >= 2
                    and self._leg_affordable(plan, reserve_b, reserve_t)
                ):
                    if best is None or plan.path_cost < best[0]:
                        best = (plan.path_cost, mark)
            if best is not None:
                mark = best[1]
                plan = self._plan_to(mark, penalize_revisit=False)
                nxt = self._follow_plan_step(plan, pos, ori)
                if nxt is not None:
                    nxt.rationale = (
                        f"clue marker sighted at {mark}: walking to the cell and "
                        "scanning to lift the evidence packet"
                    )
                    nxt.next_objective = f"collect evidence at {mark}"
                    return nxt

        # -- 2c) Terrain scan cadence keeps the known map growing ---------
        if (
            self.moves_since_scan >= 6
            and tl > costs.scan_time + reserve_t
            and bat > reserve_b + costs.scan_battery
        ):
            return Decision(
                ActionKind.SCAN,
                rationale=(
                    f"{self.moves_since_scan} cells traversed since last scan; "
                    "refreshing known terrain / hunting for clue markers"
                ),
                next_objective="extend known map, look for evidence markers",
            )

        # -- 3) Cheap information: signal pings ---------------------------
        if self._pending_post_clue_ping and tl > costs.signal_ping_time + reserve_t:
            return Decision(
                ActionKind.SIGNAL_PING,
                params={"trigger": "post-clue"},
                rationale=(
                    "clue just folded into belief; triangulating immediately "
                    "before committing battery to travel"
                ),
                next_objective="combine clue with fresh signal reading",
            )
        if self.moves_since_ping >= 4 and tl > costs.signal_ping_time + reserve_t:
            return Decision(
                ActionKind.SIGNAL_PING,
                params={"trigger": "cadence"},
                rationale=(
                    f"4+ cells traversed since last reading; belief entropy "
                    f"{self.belief.entropy():.2f} nats warrants a cheap update"
                ),
                next_objective="sharpen location belief",
            )

        # -- 4) Deliberate exploration -----------------------------------
        target = self._exploration_target()
        if target is not None:
            plan = self._plan_to(target, prefer_unexplored=True)
            if plan.feasible and self._leg_affordable(plan, reserve_b, reserve_t):
                nxt = self._follow_plan_step(plan, pos, ori)
                if nxt is not None:
                    nxt.rationale = (
                        f"frontier target {target}: maximises (coverage + belief mass - travel cost) "
                        f"under reserve {reserve_b:.1f}%"
                    )
                    nxt.next_objective = f"move toward {target}, scan on arrival"
                    return nxt

        # -- 4b) Backtracking fallback when blocked or local frontiers exhausted --
        if len(self.path_history) > 1 and (self.stuck_counter > 0 or pos in self.scan_positions):
            for past_pos in reversed(self.path_history[:-1]):
                if past_pos != pos:
                    backtrack_plan = self._plan_to(past_pos, penalize_revisit=False)
                    if backtrack_plan.feasible and len(backtrack_plan.path) >= 2:
                        nxt = self._follow_plan_step(backtrack_plan, pos, ori)
                        if nxt is not None:
                            self._current_target = None
                            nxt.rationale = f"backtracking from dead-end or blocked cell {pos} toward {past_pos}"
                            nxt.next_objective = f"retreat to branch point {past_pos}"
                            return nxt

        # -- 4c) Break rotation spin if turning repeatedly without progress --
        if self.consecutive_rotations >= 3:
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                neighbor = (pos[0] + dx, pos[1] + dy)
                if (
                    neighbor not in self.known_obstacles
                    and 0 <= neighbor[0] < self.env.width
                    and 0 <= neighbor[1] < self.env.height
                    and neighbor != pos
                ):
                    req = _orient_from_delta((dx, dy))
                    if req == ori:
                        return Decision(ActionKind.MOVE_FORWARD, rationale="breaking rotation loop by moving to open neighbor")
                    rot = _rotations_between(ori, req)
                    if rot:
                        return Decision(rot[0], rationale="breaking rotation loop toward open neighbor")

        # -- 5) Fallbacks: scan in place (only once per tile!), else conserve ---
        if (
            pos not in self.scan_positions
            and self.consecutive_scans == 0
            and tl > costs.scan_time + reserve_t
            and bat > reserve_b + costs.scan_battery
        ):
            return Decision(
                ActionKind.SCAN,
                rationale="no affordable frontier leg; scanning in place for clue markers/coverage",
                next_objective="collect nearby evidence before resource cutoff",
            )
        return Decision(
            ActionKind.WAIT,
            rationale="resources too tight to act; conserving toward the end-of-mission report",
            next_objective="end-of-mission report",
        )

    def _leg_affordable(self, plan: PathPlan, reserve_b: float, reserve_t: float) -> bool:
        return (
            self.env.battery >= self._plan_leg_battery(plan) + reserve_b
            and self.env.remaining_time >= self._plan_leg_time(plan) + reserve_t
        )

    def _follow_plan_step(self, plan: PathPlan, pos: tuple[int, int], ori: Orientation) -> Optional[Decision]:
        next_cell: Optional[tuple[int, int]] = None
        for i, cell in enumerate(plan.path):
            if cell == pos and i + 1 < len(plan.path):
                next_cell = plan.path[i + 1]
                break
        if next_cell is None:
            next_cell = plan.path[1] if len(plan.path) > 1 else pos
        if next_cell == pos:
            # Already standing on the target: caller should fall through to a
            # different action (scan/wait) rather than stall the loop.
            return None
        required = _orient_from_delta((next_cell[0] - pos[0], next_cell[1] - pos[1]))
        rot = _rotations_between(ori, required)
        if rot:
            return Decision(
                rot[0],
                rationale=f"rotating {ori.value} -> {required.value} to follow the planned path",
                next_objective=f"advance toward {next_cell}",
            )
        return Decision(
            ActionKind.MOVE_FORWARD,
            rationale=f"advancing to {next_cell} along the planned path",
            next_objective=f"continue leg toward {plan.path[-1]}",
        )

    # ------------------------------------------------------------------
    # Single stepwise execution — each call does ONE action, no outcome
    # classification.  run() handles termination and final classification.
    # ------------------------------------------------------------------
    def step(self, action: ActionKind | str | None = None, params: Optional[dict] = None) -> dict:
        """Execute exactly one agent action (autonomous or commanded).

        Returns a dict suitable for the UI / API:
        - outcome: str or None (set by ``_choose_action`` in terminal paths,
          or by classification when finished)
        - action: the ActionKind taken, or None
        - position: agent's position after the action
        - battery: battery level after the action
        - remaining_time: time remaining after the action
        - trace_entry: the TraceEntry appended, or None
        - finished: whether the mission has finished

        Safe to call repeatedly; once terminated, further calls
        are no-ops returning the final state.
        """
        # Mission already finished — no-op
        if self.outcome is not None:
            return {
                "outcome": self.outcome,
                "action": None,
                "position": self.env.current_position,
                "battery": self.env.battery,
                "remaining_time": self.env.remaining_time,
                "trace_entry": None,
                "finished": True,
            }

        step_idx = len(self.trace)

        if step_idx >= self.max_steps:
            self.outcome = "max_steps_reached"
            return {
                "outcome": self.outcome,
                "action": None,
                "position": self.env.current_position,
                "battery": self.env.battery,
                "remaining_time": self.env.remaining_time,
                "trace_entry": None,
                "finished": True,
            }

        if action is not None:
            if isinstance(action, str):
                alias_map = {
                    "rotate_left": ActionKind.TURN_LEFT,
                    "rotate_right": ActionKind.TURN_RIGHT,
                    "ping": ActionKind.SIGNAL_PING,
                }
                action = alias_map.get(action.lower(), action)
                action = ActionKind(action)
            decision = Decision(
                action=action,
                params=params or {},
                rationale="Manual operator command",
                next_objective="Manual control",
            )
        else:
            decision = self._choose_action()
            if decision is None:
                # _choose_action may set self.outcome (confirmed locate etc.) or
                # return None when already terminal; do NOT classify here.
                if self.env.is_finished and self.outcome is None:
                    self.outcome = self._classify_outcome()
                return {
                    "outcome": self.outcome,
                    "action": None,
                    "position": self.env.current_position,
                    "battery": self.env.battery,
                    "remaining_time": self.env.remaining_time,
                    "trace_entry": None,
                    "finished": self.env.is_finished or self.outcome is not None,
                }

        pre = self.belief.summary()
        result = self.env.apply_action(decision.action, decision.params)
        self._observe(result)

        # --- per-action bookkeeping ---
        self.steps_since_ping += 1
        self.moves_since_scan += 1
        if decision.action is ActionKind.MOVE_FORWARD:
            if result.success:
                self.moves_since_ping += 1
                self.successful_moves += 1
                self.distance_traveled += 1
                self.path_history.append(result.position)
                self.known_cells.add(result.position)
                self.pending_clue_markers.discard(result.position)
                self.visit_counts[result.position] = self.visit_counts.get(result.position, 0) + 1
                self.recent_positions.append(result.position)
                if len(self.recent_positions) > 16:
                    self.recent_positions.pop(0)
                self.consecutive_rotations = 0
                self.consecutive_scans = 0
                self.stuck_counter = 0
                if self._current_target == result.position:
                    self._current_target = None
            else:
                self.blocked_moves += 1
                self.stuck_counter += 1
                self._current_target = None
        elif decision.action in (ActionKind.TURN_LEFT, ActionKind.TURN_RIGHT):
            self.consecutive_rotations += 1
            self.consecutive_scans = 0
        elif decision.action is ActionKind.SCAN:
            self.scan_positions.append(result.position)
            self.scans_done += 1
            self.moves_since_scan = 0
            self.moves_since_ping = 0
            self.consecutive_scans += 1
            self.consecutive_rotations = 0
        elif decision.action is ActionKind.SIGNAL_PING:
            self.steps_since_ping = 0
            if result.success:
                self.moves_since_ping = 0
            self.consecutive_rotations = 0
            self.consecutive_scans = 0
        elif decision.action is ActionKind.WAIT:
            self.consecutive_rotations = 0
            self.consecutive_scans = 0
        elif decision.action is ActionKind.RECHARGE:
            if result.success:
                self.recharges_done += 1
            self.consecutive_rotations = 0
            self.consecutive_scans = 0

        post = self.belief.summary()
        self.trace.append(
            TraceEntry(
                step=result.step,
                sim_time=round(
                    self.spec.resources.mission_time - result.remaining_time, 3
                ),
                position=result.position,
                orientation=result.orientation,
                action=decision.action.value,
                params=decision.params,
                success=result.success,
                observations=[o.message for o in result.observations],
                battery=round(result.battery, 3),
                remaining_time=round(result.remaining_time, 3),
                belief_summary={
                    "before_max_posterior": pre["max_posterior"],
                    "after_map_estimate": post["map_estimate"],
                    "after_max_posterior": post["max_posterior"],
                    "after_entropy_nats": post["entropy_nats"],
                },
                rationale=decision.rationale,
                next_objective=decision.next_objective,
            )
        )

        step_idx += 1
        finished = self.env.is_finished or self.outcome is not None
        if finished and self.outcome is None:
            self.outcome = self._classify_outcome()
        return {
            "outcome": self.outcome,
            "action": decision.action.value,
            "position": result.position,
            "battery": round(result.battery, 3),
            "remaining_time": round(result.remaining_time, 3),
            "trace_entry": self.trace[-1] if self.trace else None,
            "finished": finished,
        }

    def run(self) -> dict:
        """Run the mission to completion.

        Calls ``step()`` repeatedly until the mission terminates (outcome set
        or ``env.is_finished``), then classifies the outcome and returns the
        final summary.
        """
        while not self._is_finished():
            result = self.step()
            if result.get("finished"):
                break
        # Classify the outcome if not already set by a terminal _choose_action
        # path (e.g. ``self.outcome = "located"`` in the convergence branch).
        if self.outcome is None:
            self.outcome = self._classify_outcome()
        return self._final_summary()

    def _is_finished(self) -> bool:
        """Return True if the mission has reached a terminal outcome."""
        return self.outcome is not None or self.env.is_finished

    def _classify_outcome(self) -> str:
        """Classify the final outcome based on the environment state."""
        if self.env.is_finished and self.env.battery <= 0:
            return "battery_exhausted"
        if self.env.is_finished and self.env.remaining_time <= 0:
            return "time_exhausted"
        if self.belief.max_posterior() >= self.spec.mission.confidence_threshold:
            return "located"
        if self.env.cell_is_station(self.env.current_position):
            return "uncertain_returned_to_station"
        return "uncertain"

    def _final_summary(self) -> dict:
        est = self.belief.map_estimate()
        return {
            "outcome": self.outcome,
            "steps": len(self.trace),
            "map_estimate": list(est),
            "max_posterior": self.belief.max_posterior(),
            "entropy": self.belief.entropy(),
            "status_posterior": self.belief.status_posterior(),
            "clues": [c.model_dump() for c in self.collected_clues],
            "battery": self.env.battery,
            "remaining_time": self.env.remaining_time,
            "distance_traveled": self.distance_traveled,
            "trace": self.trace,
        }