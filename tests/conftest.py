"""Shared fixtures: the sample mission spec and a tiny deterministic spec."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from eve_search.config import MissionSpec, load_mission
from eve_search.environment import Environment

SAMPLE_CONFIG = PROJECT_ROOT / "configs" / "sample_mission.yaml"


@pytest.fixture()
def sample_spec() -> MissionSpec:
    return load_mission(SAMPLE_CONFIG)


@pytest.fixture()
def sample_env(sample_spec: MissionSpec) -> Environment:
    return Environment(sample_spec)


@pytest.fixture()
def tiny_spec() -> MissionSpec:
    """Small, fast, fully-explicit mission used by planner/agent unit tests."""
    raw = {
        "mission": {"id": "tiny", "description": "unit test", "confidence_threshold": 0.25},
        "world": {
            "width": 12,
            "height": 10,
            "seed": 7,
            "obstacles": [[5, 0], [5, 1], [5, 2], [5, 3]],
            "hazards": [[3, 4]],
            "recharge_stations": [[1, 1], [10, 8]],
        },
        "walle": {"start": [1, 8], "orientation": "N"},
        "signal": {
            "initial": {"bearing_deg": 20.0, "strength": 55.0, "timestamp": "2805-01-01T00:00:00Z"},
            "propagation": {
                "reference_strength": 100.0,
                "decay_length": 6.0,
                "noise_sigma": 5.0,
                "bearing_noise_deg": 10.0,
            },
        },
        "resources": {"battery_capacity": 100.0, "initial_battery": 100.0, "mission_time": 120.0},
        "costs": {},
        "evidence": {"clue_count": 3, "clue_spread": 4.5, "scan_radius": 2, "seed": 99},
        "eve": {"status": "active", "position": [9, 2]},
    }
    return MissionSpec.model_validate(raw)


@pytest.fixture()
def tiny_env(tiny_spec: MissionSpec) -> Environment:
    return Environment(tiny_spec)
