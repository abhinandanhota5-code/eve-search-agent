"""Belief-state tests: Bayesian updates concentrate on truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eve_search.belief import BeliefState
from eve_search.models import Clue, ClueKind

PHYS = {
    "reference_strength": 100.0,
    "decay_length": 9.0,
    "noise_sigma": 5.0,
    "bearing_noise_deg": 10.0,
}


def test_uniform_prior_entropy_is_max(todo_flag=None):
    b = BeliefState(10, 10)
    assert abs(b.entropy() - math.log(100)) < 1e-6
    assert b.max_posterior() == pytest.approx(0.01)


def test_signal_update_concentrates_on_true_cell():
    w, h = 20, 16
    true_cell = (14, 3)
    b = BeliefState(w, h)
    # measurement as if WALL-E stands at (2, 12) with a clean reading
    dist = math.hypot(true_cell[0] - 2, true_cell[1] - 12)
    strength = 100.0 * math.exp(-dist / PHYS["decay_length"])
    from eve_search.models import bearing_between
    bearing = bearing_between((2, 12), true_cell)
    b.apply_signal(strength, bearing, (2, 12), PHYS)
    est = b.map_estimate()
    d = math.hypot(est[0] - true_cell[0], est[1] - true_cell[1])
    assert d <= 1.5, f"MAP {est} should be adjacent to truth {true_cell}"
    # posterior peak should be clearly above the uniform baseline 1/(w*h)
    assert b.max_posterior() > 3 / (w * h)


def test_noisy_signals_still_sharpen_belief():
    w, h = 24, 20
    true_cell = (18, 5)
    b = BeliefState(w, h)
    rng = np.random.default_rng(3)
    obs_from = (3, 3)
    for _ in range(6):
        dist = math.hypot(true_cell[0] - obs_from[0], true_cell[1] - obs_from[1])
        s = 100.0 * math.exp(-dist / PHYS["decay_length"]) + rng.normal(0, PHYS["noise_sigma"])
        from eve_search.models import bearing_between
        br = (bearing_between(obs_from, true_cell) + rng.normal(0, PHYS["bearing_noise_deg"])) % 360
        b.apply_signal(s, br, obs_from, PHYS)
    est = b.map_estimate()
    assert math.hypot(est[0] - true_cell[0], est[1] - true_cell[1]) <= 2.5
    pre = math.log(w * h)
    assert b.entropy() < pre


def test_half_plane_disagreement_wins_for_consistent_source():
    w, h = 24, 20
    b = BeliefState(w, h)
    # bearing east from (5,10): only eastern half should gain mass
    b.apply_signal(100.0 * math.exp(-4.0 / 9.0), 90.0, (5, 10), PHYS)
    p = b.posterior()
    east_mass = p[:, 6:].sum()
    west_mass = p[:, :6].sum()
    assert east_mass > west_mass


def test_ring_clue_restricts_to_radius():
    w, h = 24, 20
    b = BeliefState(w, h)
    clue = Clue(id="c1", kind=ClueKind.RING, text="ring",
                pickup=(10, 10), payload={"radius": 4.0})
    b.apply_clue(clue)
    p = b.posterior()
    # outside roughly 2.5x the radius mass must be negligible
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    far = np.hypot(xs - 10, ys - 10) > 12
    assert p[far].sum() < 0.01


def test_region_clue_upweights_matching_regions():
    w, h = 24, 20
    b = BeliefState(w, h)
    clue = Clue(id="c2", kind=ClueKind.REGION, text="region",
                pickup=(5, 5), payload={"region": "north-east"})
    b.apply_clue(clue)
    est = b.map_estimate()
    assert est[0] >= w * 2 / 3
    assert est[1] < h / 3


def test_status_clue_updates_status_posterior():
    b = BeliefState(12, 10)
    assert b.status_posterior()["active"] == pytest.approx(0.25)
    clue = Clue(id="c3", kind=ClueKind.STATUS, text="status",
                pickup=(3, 3), payload={"status": "active"})
    b.apply_clue(clue)
    sp = b.status_posterior()
    assert sp["active"] > 0.55
    assert sp == pytest.approx({k: (sp["active"] if k == "active" else (1 - sp["active"]) / 3)
                                for k in sp})


def test_clue_without_pickup_is_ignored():
    b = BeliefState(10, 10)
    clue = Clue(id="c4", kind=ClueKind.RING, text="?", payload={"radius": 3})
    b.apply_clue(clue)
    assert b.max_posterior() == pytest.approx(0.01)


def test_belief_mass_within_radius_grows_when_truth_consistent():
    b = BeliefState(24, 20)
    truth = (12, 8)
    m0 = b.mass_within(truth, 3.0)
    b.apply_clue(Clue(id="r", kind=ClueKind.RING, text="t", pickup=(13, 7), payload={"radius": 3.0}))
    m1 = b.mass_within(truth, 3.0)
    assert m1 > m0
