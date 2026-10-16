"""Agent-level tests: hidden-state isolation, decisions, trace, determinism."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from eve_search.agent import SearchAgent
from eve_search.environment import Environment
from eve_search.models import ActionKind, ObservationKind


def run_agent(spec, seed=None, max_steps=4000):
    env = Environment(spec, seed=seed)
    agent = SearchAgent(env, spec, max_steps=max_steps)
    summary = agent.run()
    return env, agent, summary


def test_agent_never_accesses_hidden_state_via_public_interface(sample_spec):
    """The public observation interface must never carry EVE's ground truth."""
    env = Environment(sample_spec)
    agent = SearchAgent(env, sample_spec)
    agent.run()

    eve_true = env._hidden.eve_position
    eve_status = env._hidden.eve_status

    # 1) Observation payloads must not contain any ground-truth field.
    forbidden_keys = {"eve_position", "true_position", "ground_truth", "oracle"}
    for entry in agent.trace:
        for msg in entry.observations:
            assert "eve_position" not in msg.lower()
        # rationale/next_objective are the agent's own inference; allowed.

    # Re-derive raw observations the same way the agent saw them to check `data`
    # fields: rerun a scripted scan/ping (deterministic) and inspect payloads.
    env2 = Environment(sample_spec)
    r = env2.apply_action(ActionKind.SCAN)
    for obs in r.observations:
        leaked = forbidden_keys & set(obs.data.keys())
        assert not leaked, f"observation leaked hidden keys {leaked}"
        assert "status" not in obs.data or obs.data.get("status") != eve_status

    for obs in env2.apply_action(ActionKind.SIGNAL_PING).observations:
        assert not (forbidden_keys & set(obs.data.keys()))
        assert "distance" not in obs.data  # a direct distance would be ground truth

    # 2) The agent's known map must not secretly equal the hidden map
    hidden_obstacles = env._hidden.obstacles
    assert not hidden_obstacles.issubset(agent.known_obstacles) or len(agent.known_obstacles) < len(hidden_obstacles)

    # 3) Agent must not hold any attribute referencing hidden EVE state
    truth_markers = [a for a in dir(agent) if "eve" in a.lower() and a != "belief"]
    assert not truth_markers

    # 4) env public surface lacks hidden state
    public = [a for a in dir(env) if not a.startswith("_")]
    assert not any("eve" in p.lower() for p in public)

    # 5) JSON dump of the observation models has no hidden coordinate pair
    import json as _json
    env3 = Environment(sample_spec)
    for action in (
        ActionKind.SCAN,
        ActionKind.SIGNAL_PING,
        ActionKind.MOVE_FORWARD,
        ActionKind.RECHARGE,
    ):
        res = env3.apply_action(action)
        for obs in res.observations:
            payload = _json.dumps(obs.model_dump(), default=str)
            assert f"'position': [{eve_true[0]}, {eve_true[1]}]" not in payload
            assert f"['{eve_true[0]}', '{eve_true[1]}']" not in payload


def test_agent_belief_respects_observation_barrier(sample_spec):
    """Even after many scans, agent knowledge stays sparse vs ground truth."""
    env = Environment(sample_spec, seed=5)
    agent = SearchAgent(env, sample_spec, max_steps=150)
    agent.run()
    total_cells = env.width * env.height
    assert len(agent.known_cells) < total_cells * 0.8


def test_full_mission_completes_and_locates(sample_spec):
    env, agent, summary = run_agent(sample_spec, seed=None)
    assert summary["outcome"] in (
        "located", "located_unverified", "uncertain", "uncertain_returned_to_station",
        "battery_exhausted", "time_exhausted",
    )
    # Belief on the sample config should converge strongly on truth (seed 42)
    eve_true = env._hidden.eve_position
    est = summary["map_estimate"]
    assert math.hypot(est[0] - eve_true[0], est[1] - eve_true[1]) <= 4.0, (
        f"estimate {est} too far from truth {eve_true}"
    )
    assert summary["max_posterior"] > 0.05


def test_every_action_recorded_in_trace(sample_spec):
    env, agent, summary = run_agent(sample_spec)
    steps = [e.step for e in agent.trace]
    assert steps == sorted(steps)
    assert len(agent.trace) == summary["steps"]
    assert len(agent.trace) > 0
    for entry in agent.trace:
        assert entry.rationale, "each entry needs a rationale"
        assert entry.next_objective
        assert entry.battery >= 0
        assert entry.observations is not None


def test_belief_changes_with_new_evidence(sample_spec):
    env, agent, summary = run_agent(sample_spec)
    # posterior should tighten from the uniform prior as evidence arrives
    pre = [e.belief_summary["before_max_posterior"] for e in agent.trace]
    post = [e.belief_summary["after_max_posterior"] for e in agent.trace]
    assert max(post) > pre[0]
    assert summary["entropy"] < math.log(sample_spec.world.width * sample_spec.world.height) - 2.0


def test_deterministic_reruns_produce_identical_traces(sample_spec):
    _, a1, s1 = run_agent(sample_spec, seed=11)
    _, a2, s2 = run_agent(sample_spec, seed=11)
    assert [e.model_dump() for e in a1.trace] == [e.model_dump() for e in a2.trace]
    assert s1["map_estimate"] == s2["map_estimate"]
    assert s1["outcome"] == s2["outcome"]


def test_agent_adapts_to_different_map_and_seed(sample_spec):
    """Same spec, different seed: agent must still complete without crash."""
    for seed in (3, 77, 2028):
        env, agent, summary = run_agent(sample_spec, seed=seed)
        assert summary["steps"] > 0
        assert env.battery <= sample_spec.resources.battery_capacity


def test_tiny_mission_agent_works(tiny_spec):
    env, agent, summary = run_agent(tiny_spec)
    assert summary["outcome"]
    assert len(agent.trace) == summary["steps"]


def test_agent_pings_and_scans_occur(sample_spec):
    env, agent, summary = run_agent(sample_spec)
    actions = {e.action for e in agent.trace}
    assert ActionKind.SIGNAL_PING.value in actions
    assert ActionKind.SCAN.value in actions
    assert ActionKind.MOVE_FORWARD.value in actions


def test_blocked_moves_are_replanned_not_repeated(sample_spec):
    env, agent, summary = run_agent(sample_spec)
    # walk the trace: no more than 2 consecutive failures on the same target
    fails = 0
    for e in agent.trace:
        if not e.success and e.action == ActionKind.MOVE_FORWARD.value:
            fails += 1
        else:
            fails = 0
        assert fails <= 2, "agent should learn from blocked cells"


def test_agent_stops_on_resource_exhaustion(sample_spec):
    spec = sample_spec.model_copy(deep=True)
    spec.resources.mission_time = 40.0
    env, agent, summary = run_agent(spec)
    assert summary["outcome"] in ("time_exhausted", "uncertain", "uncertain_returned_to_station")
    assert env.remaining_time >= 0
    assert not env.is_finished or env.remaining_time == 0


def test_agent_never_uses_more_battery_than_capacity(sample_spec):
    env, agent, summary = run_agent(sample_spec)
    assert env.battery >= 0
    assert env.battery <= sample_spec.resources.battery_capacity + 1e-9


def test_rescue_mode_keeps_reserve(tiny_spec):
    """Short battery forces rescue behaviour well before empty."""
    spec = tiny_spec.model_copy(deep=True)
    spec.resources.initial_battery = 18.0
    env, agent, summary = run_agent(spec)
    # battery must never drop below 0
    assert env.battery >= 0
    modes = set()  # trace rationales should mention reserves at some point
    for e in agent.trace:
        if "reserve" in e.rationale or "recharge" in e.rationale:
            modes.add("rescue")
    assert modes == {"rescue"} or summary["outcome"] != "battery_exhausted"


def test_agent_leaves_repeatedly_visited_area_for_alternative_frontier(sample_spec):
    """The agent must penalize recently/frequently visited areas and select an alternative frontier."""
    env = Environment(sample_spec)
    agent = SearchAgent(env, sample_spec)

    pos = env.current_position
    # Simulate a scenario where a small local pocket has been heavily traversed
    local_cluster = {pos, (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1)}
    for c in local_cluster:
        agent.known_cells.add(c)
        agent.visit_counts[c] = 5
    agent.recent_positions = list(local_cluster) * 3

    # Ensure agent knows an outward cell opening an alternative frontier
    distant_cell = (pos[0] + 3, pos[1] + 3)
    agent.known_cells.add(distant_cell)

    # When picking an exploration target, it should target an unvisited frontier
    target = agent._exploration_target()
    assert target is not None
    assert agent.visit_counts.get(target, 0) == 0

    # Step the agent and verify it explores outside the local pocket
    for _ in range(15):
        if agent._is_finished():
            break
        agent.step()

    new_cells = [p for p in agent.path_history if p not in local_cluster]
    assert len(new_cells) > 0, "Agent must leave repeatedly visited pocket and explore new cells"

