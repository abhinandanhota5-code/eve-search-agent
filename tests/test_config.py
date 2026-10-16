"""Configuration validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eve_search.config import MissionSpec, derive_eve_position, load_mission


def test_sample_config_loads_and_validates(sample_spec: MissionSpec):
    assert sample_spec.mission.id == "sample-long-range-rescue"
    assert sample_spec.world.width == 26
    assert sample_spec.world.height == 20
    assert sample_spec.resources.initial_battery <= sample_spec.resources.battery_capacity
    assert sample_spec.eve.status in ("dormant", "standby", "active", "damaged")


def test_completely_empty_initial_signal_rejected():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "active"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_timestamp_only_initial_signal_is_valid():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"timestamp": "2805-01-01T00:00:00Z"}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "active"},
    }
    spec = MissionSpec.model_validate(raw)
    assert spec.signal.initial.timestamp is not None


def test_out_of_bounds_recharge_station_rejected():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1, "recharge_stations": [[20, 3]]},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"bearing_deg": 20.0}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "active"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_battery_over_capacity_rejected():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"bearing_deg": 20.0}},
        "resources": {"battery_capacity": 100, "initial_battery": 120, "mission_time": 50},
        "eve": {"status": "active"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_hazard_on_obstacle_rejected():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1,
                  "obstacles": [[4, 4]], "hazards": [[4, 4]]},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"bearing_deg": 20.0}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "active"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_unknown_status_rejected():
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"bearing_deg": 20.0}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "hyper-thrusted"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_unknown_fields_rejected(sample_spec: MissionSpec):
    raw = {
        "mission": {"id": "x"},
        "world": {"width": 10, "height": 10, "seed": 1, "gravitaz": 3},
        "walle": {"start": [1, 1], "orientation": "N"},
        "signal": {"initial": {"bearing_deg": 20.0}},
        "resources": {"battery_capacity": 100, "initial_battery": 100, "mission_time": 50},
        "eve": {"status": "active"},
    }
    with pytest.raises(ValidationError):
        MissionSpec.model_validate(raw)


def test_derived_eve_position_is_deterministic(sample_spec: MissionSpec):
    a = derive_eve_position(sample_spec)
    b = derive_eve_position(sample_spec)
    assert a == b
    w, h = sample_spec.world.width, sample_spec.world.height
    assert 0 <= a[0] < w and 0 <= a[1] < h


def test_load_mission_missing_file():
    with pytest.raises(FileNotFoundError):
        load_mission("configs/does_not_exist.yaml")
