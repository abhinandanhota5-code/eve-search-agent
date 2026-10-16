"""Reporting/artifact tests: trace JSONL, report JSON, report markdown."""

from __future__ import annotations

import json

from eve_search.agent import SearchAgent
from eve_search.environment import Environment
from eve_search.reporting import build_report, save_report, save_trace


def _run(sample_spec, seed=42):
    env = Environment(sample_spec, seed=seed)
    agent = SearchAgent(env, sample_spec)
    summary = agent.run()
    return env, agent, summary


def test_trace_jsonl_written_and_parses(sample_spec, tmp_path):
    env, agent, summary = _run(sample_spec)
    path = save_trace(agent, tmp_path)
    assert path.name == "action_trace.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(agent.trace)
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["step"] == 1
    assert all("rationale" in p and "battery" in p for p in parsed)
    assert all(p["action"] for p in parsed)


def test_report_json_and_markdown_written(sample_spec, tmp_path):
    env, agent, summary = _run(sample_spec)
    report = build_report(sample_spec, env, agent, summary)
    json_path, md_path = save_report(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["mission"]["id"] == sample_spec.mission.id
    assert data["outcome"] == report["outcome"]
    txt = md_path.read_text()
    assert "# EVE Search Agent - Mission Report" in txt
    assert "Inference" in txt
    assert "Recommended next action" in txt


def test_report_separates_observed_from_inferred(sample_spec, tmp_path):
    env, agent, summary = _run(sample_spec)
    report = build_report(sample_spec, env, agent, summary)
    assert "evidence_observed" in report
    assert "inference" in report
    inference_text = json.dumps(report["inference"])
    assert "NOT" in inference_text or "never observed" in inference_text


def test_report_full_data_serialization(sample_spec, tmp_path):
    env, agent, summary = _run(sample_spec)
    report = build_report(sample_spec, env, agent, summary)
    blob = json.dumps(report)  # must be JSON serializable
    assert report["activity"]["distance_explored"] == agent.distance_traveled
    assert "recommended_next_action" in report and report["recommended_next_action"]
