"""Pathfinder tests: A* correctness with obstacles, cost estimates, battery logic."""

from __future__ import annotations

import pytest

from eve_search.planner import PathPlan, astar, turns_needed
from eve_search.models import Orientation


def full_known(w: int, h: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(w) for y in range(h)}


def test_straight_path_on_empty_grid():
    plan = astar((2, 2), (6, 2), 12, 12, known_cells=full_known(12, 12))
    assert plan.feasible
    assert plan.path_cost == 4
    assert len(plan.path) == 5


def test_path_originates_at_start_and_ends_at_goal():
    plan = astar((0, 0), (5, 5), 12, 12, known_cells=full_known(12, 12))
    assert plan.path[0] == (0, 0)
    assert plan.path[-1] == (5, 5)


def test_obstacle_forces_detour():
    # Wall across x=5 except a gap at y=9 and y=0
    obstacles = {(5, y) for y in range(0, 10)}
    plan = astar((0, 5), (8, 5), 12, 12, known_obstacles=obstacles)
    assert plan.feasible
    assert plan.path_cost > 8  # detour cost exceeds manhattan-ish 8
    for cell in plan.path:
        assert cell not in obstacles


def test_fully_walled_goal_is_infeasible():
    obstacles = {(5, y) for y in range(0, 12)}  # wall spans whole height
    plan = astar((0, 5), (8, 5), 12, 12, known_obstacles=obstacles)
    assert not plan.feasible


def test_hazard_penalty_chooses_longer_safe_route():
    hazards = {(3, 5)}
    known = full_known(12, 12)
    straight = astar((0, 5), (8, 5), 12, 12, known_hazards=hazards, known_cells=known)
    detoured = astar((0, 5), (8, 5), 12, 12, known_cells=known)
    assert straight.feasible and detoured.feasible
    assert straight.path_cost >= detoured.path_cost  # hazard route never cheaper


def test_unknown_penalty_makes_unscanned_cells_costlier():
    plan_unknown = astar((0, 0), (4, 0), 8, 8)  # nothing scanned yet
    plan_known = astar((0, 0), (4, 0), 8, 8, known_cells=full_known(8, 8))
    assert plan_unknown.feasible and plan_known.feasible
    assert plan_unknown.path_cost > plan_known.path_cost


def test_turns_needed_counts_rotations():
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]  # east then south
    assert turns_needed(Orientation.EAST, path) == 1  # start aligned, 1 turn
    assert turns_needed(Orientation.NORTH, path) == 2  # align + 1 turn
    assert turns_needed(Orientation.EAST, [(1, 0), (2, 0)]) == 0


def test_early_goal_check_frontier_start_case():
    plan = astar((0, 0), (1, 0), 5, 5, known_cells=full_known(5, 5))
    assert plan.path == [(0, 0), (1, 0)]
    assert plan.path_cost == 1


def test_revisit_penalty_chooses_unvisited_alternative():
    """A* should prefer an unvisited path over a heavily visited path of identical length."""
    known = full_known(5, 5)
    # Path A goes through (1, 0); Path B goes through (0, 1). Both reach (1, 1) in 2 steps.
    visit_counts = {(1, 0): 4}
    plan = astar((0, 0), (1, 1), 5, 5, known_cells=known, visit_counts=visit_counts)
    assert plan.feasible
    assert (1, 0) not in plan.path, "Planner should avoid the heavily revisited cell when an alternative exists"
    assert (0, 1) in plan.path

