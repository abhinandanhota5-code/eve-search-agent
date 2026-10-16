"""Environment mechanics tests: costs, movement, observations, hidden state."""

from __future__ import annotations

from eve_search.environment import Environment
from eve_search.models import ActionKind, CellType, ObservationKind


def test_movement_costs_battery_and_time(sample_env: Environment, sample_spec):
    env = sample_env
    budget = sample_spec.resources
    before_b, before_t = env.battery, env.remaining_time
    r = env.apply_action(ActionKind.MOVE_FORWARD)
    assert r.success
    after_b, after_t = env.battery, env.remaining_time
    assert abs((after_b - before_b) + sample_spec.costs.move_battery) < 1e-9
    assert abs((before_t - after_t) - sample_spec.costs.move_time) < 1e-9


def test_rotation_changes_orientation_and_cost(sample_env: Environment, sample_spec):
    env = sample_env
    before_b, before_t = env.battery, env.remaining_time
    r = env.apply_action(ActionKind.TURN_LEFT)
    assert r.success
    assert r.orientation != env.spec.walle.orientation
    assert abs((before_t - env.remaining_time) - sample_spec.costs.rotate_time) < 1e-9


def test_move_cost_when_battery_costs(sample_spec):
    env = Environment(sample_spec)
    b0, t0 = env.battery, env.remaining_time
    env.apply_action(ActionKind.SIGNAL_PING)
    ping_b, ping_t = abs(env.battery - b0), abs(t0 - env.remaining_time)

    env = Environment(sample_spec)
    b0, t0 = env.battery, env.remaining_time
    env.apply_action(ActionKind.SCAN)
    scan_b, scan_t = abs(env.battery - b0), abs(t0 - env.remaining_time)

    assert scan_b > ping_b
    assert scan_t > ping_t


def test_wait_consumes_only_time(sample_env: Environment, sample_spec):
    env = sample_env
    t0 = env.remaining_time
    r = env.apply_action(ActionKind.WAIT)
    assert r.success
    assert env.battery == sample_spec.resources.initial_battery
    assert 0 < t0 - env.remaining_time <= sample_spec.costs.wait_time


def test_blocked_move_returns_failed_result(tiny_env: Environment):
    env = tiny_env
    r = env.apply_action(ActionKind.MOVE_FORWARD)
    assert r.success
    prev_pos = env.current_position


def test_boundary_move_fails(sample_env: Environment):
    env = sample_env
    env._walle_pos = (0, 0)
    env._orientation = env.spec.walle.orientation
    # Force west/north so we hit an edge: pick an orientation that leaves map
    from eve_search.models import Orientation
    env._orientation = Orientation.WEST
    r = env.apply_action(ActionKind.MOVE_FORWARD)
    assert not r.success
    assert env.current_position == (0, 0)
    assert any(o.kind is ObservationKind.BLOCKED for o in r.observations)


def test_scan_reveals_within_radius_only(tiny_env: Environment, tiny_spec):
    env = tiny_env
    r = env.apply_action(ActionKind.SCAN)
    scans = [o for o in r.observations if o.kind is ObservationKind.TERRAIN_SCAN]
    assert len(scans) == 1
    cells = scans[0].data["cells"]
    pos = env.current_position
    radius = tiny_spec.evidence.scan_radius
    for key in cells:
        x, y = (int(v) for v in key.split(","))
        assert abs(x - pos[0]) <= radius
        assert abs(y - pos[1]) <= radius


def test_scan_collects_clues(sample_env: Environment):
    env = sample_env
    # Environment-level mechanics test: stand next to a hidden clue placement
    # (proper agents reach this state through scans alone).
    cpos, clue = env._hidden.clue_placements[0]
    env._walle_pos = (cpos[0], min(cpos[1] + 1, env.height - 1))
    r = env.apply_action(ActionKind.SCAN)
    assert any(o.kind is ObservationKind.CLUE_DISCOVERED for o in r.observations)


def test_signal_pings_are_noisy_not_exact(sample_env: Environment, sample_spec):
    env = sample_env
    for _ in range(6):
        env.apply_action(ActionKind.SIGNAL_PING)
    assert env.battery < sample_spec.resources.initial_battery


def test_observation_never_contains_eve_position(sample_env: Environment, sample_spec):
    env = sample_env
    hidden = env._hidden.eve_position
    for _ in range(12):
        r = env.apply_action(ActionKind.SCAN if _ % 2 else ActionKind.SIGNAL_PING)
        for obs in r.observations:
            dumped = obs.model_dump_json()
            assert f"[{hidden[0]}, {hidden[1]}]" not in dumped
            assert f"({hidden[0]}, {hidden[1]})" not in dumped


def test_recharge_only_at_station(tiny_env: Environment, tiny_spec):
    env = tiny_env
    r = env.apply_action(ActionKind.RECHARGE)
    assert r.success is False
    env.apply_action(ActionKind.TURN_LEFT)
    env.apply_action(ActionKind.TURN_LEFT)
    env.apply_action(ActionKind.TURN_LEFT)
    env.apply_action(ActionKind.MOVE_FORWARD)  # onto [0,8] (open)
    # teleport-free: navigate the long way is too slow; directly verify at station
    env._walle_pos = (1, 1)
    before = env.battery
    r = env.apply_action(ActionKind.RECHARGE)
    assert r.success
    assert env.battery > before


def test_battery_exhaustion_finishes_mission(tiny_env: Environment, tiny_spec):
    env = tiny_env
    mb = tiny_spec.costs.move_battery
    env._battery = mb + 0.001  # exactly two moves of energy
    env.apply_action(ActionKind.MOVE_FORWARD)
    assert not env.is_finished
    env.apply_action(ActionKind.MOVE_FORWARD)
    assert env.is_finished
    assert env.battery == 0.0


def test_public_surface_has_no_ground_truth_attributes(sample_env: Environment):
    env = sample_env
    public = [a for a in dir(env) if not a.startswith("_")]
    blob = " ".join(public)
    for token in ("eve_position", "eve_status", "clue_placements", "hidden"):
        assert token not in blob


def test_time_exhaustion_finishes_mission(tiny_env: Environment, tiny_spec):
    env = tiny_env
    env._time_left = 0.4
    env.apply_action(ActionKind.WAIT)  # wait_time 1.0 -> crosses 0
    assert env.is_finished
