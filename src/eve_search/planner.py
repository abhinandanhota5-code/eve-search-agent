"""Grid pathfinding and resource-aware cost helpers (A* over 4-connected cells).

The planner works purely on *known* information: obstacle knowledge the agent
has scanned, hazard knowledge, and station knowledge. Unknown cells are
assumed traversable (spike cost) because the world is only ever partially
observed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .models import CARDINAL_ORDER, Orientation


@dataclass
class PathPlan:
    path: list[tuple[int, int]]          # intermediate cells incl. start & goal
    path_cost: float                     # estimated step cost (battery-neutral)
    turn_count: int                      # rotations needed when following path
    feasible: bool                       # False when unreachable


def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    known_obstacles: set[tuple[int, int]] = frozenset(),
    known_hazards: set[tuple[int, int]] = frozenset(),
    known_cells: set[tuple[int, int]] = frozenset(),
    allow_unknown: bool = True,
    hazard_penalty: float = 6.0,
    unknown_penalty: float = 0.35,
    visit_counts: Optional[dict[tuple[int, int], int]] = None,
    recent_cells: Optional[Sequence[tuple[int, int]]] = None,
    revisit_penalty: float = 0.3,
    recent_penalty: float = 0.5,
    prefer_unexplored: bool = False,
) -> PathPlan:
    """A* over a 4-connected grid with soft hazard/unknown penalties and visit costs."""
    if start == goal:
        return PathPlan([start], 0.0, 0, True)

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    recent_list = list(recent_cells) if recent_cells else []
    recent_set = set(recent_list)

    while open_heap:
        f, _, node = heapq.heappop(open_heap)
        if node == goal:
            path = [goal]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            path.reverse()
            return PathPlan(path, g_score[goal], _turns_along(path), True)

        x, y = node
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = (nx, ny)
            if nxt in known_obstacles:
                continue
            step_cost = 1.0
            if nxt in known_hazards:
                step_cost += hazard_penalty
            elif nxt not in known_cells and allow_unknown:
                step_cost += 0.0 if prefer_unexplored else unknown_penalty

            # Penalize frequently visited cells when exploring
            if visit_counts and nxt in visit_counts and nxt != goal:
                step_cost += revisit_penalty * min(visit_counts[nxt], 5)

            # Penalize recently visited cells to encourage alternative routes
            if recent_set and nxt in recent_set and nxt != goal:
                recency_idx = len(recent_list) - 1 - recent_list.index(nxt)
                recency_factor = max(0.2, 1.0 - (recency_idx / max(1, len(recent_list))))
                step_cost += recent_penalty * recency_factor

            tentative = g_score[node] + step_cost
            if tentative < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative
                came[nxt] = node
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(nxt, goal), counter, nxt))
    return PathPlan([start], math.inf, 0, False)


def _turns_along(path: Sequence[tuple[int, int]]) -> int:
    if len(path) < 2:
        return 0
    head = None
    turns = 0
    for i in range(1, len(path)):
        d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if head is not None and d != head:
            turns += 1
        head = d
    return turns


def turns_needed(current: Orientation, path: Sequence[tuple[int, int]]) -> int:
    """Minimum L/R rotations to walk ``path`` starting from ``current`` heading."""
    if len(path) < 2:
        return 0
    head = None
    turns = 0
    facing = current
    for i in range(1, len(path)):
        d = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if head is None:
            head = d
            facing = _orient_from_delta(d)
            turns += _rotation_steps(current, facing)
        elif d != head:
            new = _orient_from_delta(d)
            turns += _rotation_steps(HEAD_TO_ORIENT[head], new)
            head = d
    return turns


HEAD_TO_ORIENT = {
    (1, 0): Orientation.EAST,
    (-1, 0): Orientation.WEST,
    (0, 1): Orientation.SOUTH,
    (0, -1): Orientation.NORTH,
}


def _orient_from_delta(d):
    return HEAD_TO_ORIENT[d]


def _rotation_steps(a: Orientation, b: Orientation) -> int:
    ai = CARDINAL_ORDER.index(a.value)
    bi = CARDINAL_ORDER.index(b.value)
    diff = (bi - ai) % 4
    return min(diff, 4 - diff)
