"""Flask web application for the EVE Search Agent.

Provides a live, interactive three-column UI for running and monitoring
WALL-E missions to locate EVE. Connects to the real simulator — every
value (position, belief heatmap, resources, action trace) comes from the
actual agent/environment pair.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from flask import Flask, jsonify, render_template, request

from .agent import SearchAgent
from .config import MissionSpec, load_mission
from .environment import Environment
from .models import ActionKind, chebyshev_distance

app = Flask(__name__)

# Global active mission session
_session: Optional[MissionSession] = None


class MissionSession:
    """Manages an active simulation session with step-by-step and autoplay controls."""

    def __init__(self, spec: MissionSpec, seed: Optional[int] = None):
        self.spec = spec
        self.seed = seed if seed is not None else spec.rng_seed
        self.env = Environment(spec, seed=self.seed)
        self.agent = SearchAgent(self.env, spec)
        self.status = "ready"
        self._lock = threading.Lock()
        self._playing = False
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.speed = 0.35  # seconds per autoplay step
        self.last_step_result: Optional[dict] = None

    def step(self, action: ActionKind | str | None = None, params: Optional[dict] = None) -> dict:
        with self._lock:
            if self.agent.outcome is not None or self.env.is_finished:
                self.status = "completed"
                return self.to_dict()
            self.last_step_result = self.agent.step(action=action, params=params)
            if self.agent.outcome is not None or self.env.is_finished:
                if self.agent.outcome is None:
                    self.agent.outcome = self.agent._classify_outcome()
                self.status = "completed"
                self._playing = False
            elif self.status != "running":
                self.status = "idle"
            return self.to_dict()

    def play(self, speed: Optional[float] = None) -> None:
        with self._lock:
            if speed is not None:
                self.speed = max(0.05, min(2.0, float(speed)))
            if self.agent.outcome is not None or self.env.is_finished:
                self.status = "completed"
                return
            if self._playing:
                return
            self._playing = True
            self.status = "running"
            self._stop_event.clear()
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

    def pause(self) -> None:
        with self._lock:
            self._playing = False
            self._stop_event.set()
            if self.status == "running":
                self.status = "paused"

    def _play_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if not self._playing or self.agent.outcome is not None or self.env.is_finished:
                    self._playing = False
                    if self.agent.outcome is not None or self.env.is_finished:
                        if self.agent.outcome is None:
                            self.agent.outcome = self.agent._classify_outcome()
                        self.status = "completed"
                    break
                self.last_step_result = self.agent.step()
                if self.agent.outcome is not None or self.env.is_finished:
                    if self.agent.outcome is None:
                        self.agent.outcome = self.agent._classify_outcome()
                    self.status = "completed"
                    self._playing = False
                    break
            time.sleep(self.speed)

    def to_dict(self, include_eval: bool = False) -> dict[str, Any]:
        pos = self.env.current_position
        ori = self.env.current_orientation.value
        belief_est = self.agent.belief.map_estimate()
        max_p = self.agent.belief.max_posterior()
        entropy = self.agent.belief.entropy()

        stations = list(self.agent.known_stations)
        if stations:
            dist_safety = min(chebyshev_distance(pos, s) for s in stations)
        else:
            dist_safety = chebyshev_distance(pos, tuple(self.spec.walle.start))

        posterior = self.agent.belief.posterior().tolist()

        data: dict[str, Any] = {
            "status": self.status,
            "outcome": self.agent.outcome,
            "step": len(self.agent.trace),
            "pos": list(pos),
            "orientation": ori,
            "battery": round(self.env.battery, 2),
            "battery_capacity": self.spec.resources.battery_capacity,
            "battery_pct": round((self.env.battery / self.spec.resources.battery_capacity) * 100, 1),
            "time_left": round(self.env.remaining_time, 1),
            "total_time": self.spec.resources.mission_time,
            "agent_mode": self.agent.mode if self.status != "completed" else (self.agent.outcome or "completed"),
            "target_tile": list(belief_est),
            "dist_safety": dist_safety,
            "recharge_count": self.agent.recharges_done,
            "distance_traveled": self.agent.distance_traveled,
            "scans_done": self.agent.scans_done,
            "pings_done": self.agent.pings_done,
            "clues": [c.model_dump() for c in self.agent.collected_clues],
            "pending_clue_markers": [list(c) for c in self.agent.pending_clue_markers],
            "map_estimate": list(belief_est),
            "confidence": round(max_p, 4),
            "entropy": round(entropy, 4),
            "cells_known": len(self.agent.known_cells),
            "known_cells": [f"{x},{y}" for x, y in self.agent.known_cells],
            "known_obstacles": [f"{x},{y}" for x, y in self.agent.known_obstacles],
            "known_hazards": [f"{x},{y}" for x, y in self.agent.known_hazards],
            "known_stations": [f"{x},{y}" for x, y in self.agent.known_stations],
            "path": [list(c) for c in self.agent.path_history],
            "belief_map": posterior,
            "world": {
                "width": self.spec.world.width,
                "height": self.spec.world.height,
                "seed": self.seed,
            },
            "latest_trace": self.agent.trace[-1].model_dump() if self.agent.trace else None,
            "trace": [t.model_dump() for t in self.agent.trace[-50:]],
            "is_playing": self._playing,
        }

        if include_eval:
            data["evaluation"] = self.env.evaluation_only_map()

        return data


def parse_mission_spec(data: Any) -> MissionSpec:
    """Parse mission spec from file path, raw YAML/JSON string, or dict."""
    if isinstance(data, (str, Path)):
        p = Path(str(data))
        if p.exists() and p.is_file():
            return load_mission(p)
        raw = yaml.safe_load(str(data))
        return MissionSpec.model_validate(raw)
    if isinstance(data, dict):
        return MissionSpec.model_validate(data)
    # Default fallback
    return load_mission(Path("configs/sample_mission.yaml"))


def reset_state() -> None:
    """Stop any active background run and clear active session."""
    global _session
    if _session is not None:
        _session.pause()
    _session = None


@app.route("/", methods=["GET"])
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/config/default", methods=["GET"])
def get_default_config():
    """Return default mission configuration."""
    cfg_path = Path("configs/sample_mission.yaml")
    if not cfg_path.exists():
        return jsonify({"status": "error", "message": "sample_mission.yaml not found"}), 404
    text = cfg_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    return jsonify({
        "status": "ok",
        "yaml": text,
        "json": parsed,
    })


@app.route("/init", methods=["POST"])
@app.route("/reset", methods=["POST"])
def init_mission():
    """Initialize or reset a mission session with specified config and seed."""
    global _session
    reset_state()

    data = request.get_json(silent=True) or {}
    config_input = data.get("config") or data.get("config_path") or "configs/sample_mission.yaml"
    seed = data.get("seed")
    include_eval = bool(data.get("include_eval", False))

    try:
        spec = parse_mission_spec(config_input)
        if seed is not None:
            seed = int(seed)
        _session = MissionSession(spec, seed=seed)
        return jsonify({
            "status": "ok",
            "message": "Mission initialized",
            "state": _session.to_dict(include_eval=include_eval),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/run", methods=["POST"])
def run():
    """Start or resume mission autoplay."""
    global _session

    data = request.get_json(silent=True) or {}
    speed = data.get("speed")

    if _session is None:
        config_input = data.get("config") or data.get("config_path") or "configs/sample_mission.yaml"
        seed = data.get("seed")
        try:
            spec = parse_mission_spec(config_input)
            if seed is not None:
                seed = int(seed)
            _session = MissionSession(spec, seed=seed)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    _session.play(speed=speed)
    return jsonify({
        "status": "started",
        "message": "Mission started in background",
        "state": _session.to_dict(include_eval=bool(data.get("include_eval", False))),
    })


@app.route("/pause", methods=["POST"])
def pause():
    """Pause mission autoplay."""
    global _session
    if _session is None:
        return jsonify({"status": "error", "message": "No active mission"}), 400
    _session.pause()
    return jsonify({
        "status": "paused",
        "message": "Mission paused",
        "state": _session.to_dict(),
    })


@app.route("/step", methods=["POST"])
def step():
    """Execute exactly one simulation step."""
    global _session
    data = request.get_json(silent=True) or {}

    if _session is None:
        config_input = data.get("config") or data.get("config_path") or "configs/sample_mission.yaml"
        seed = data.get("seed")
        try:
            spec = parse_mission_spec(config_input)
            if seed is not None:
                seed = int(seed)
            _session = MissionSession(spec, seed=seed)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    action = data.get("action")
    include_eval = bool(data.get("include_eval", False))
    state_data = _session.step(action=action)
    if include_eval:
        state_data["evaluation"] = _session.env.evaluation_only_map()
    return jsonify({
        "status": "ok",
        "state": state_data,
        "last_action": _session.last_step_result.get("action") if _session.last_step_result else None,
    })


@app.route("/manual", methods=["POST"])
def manual():
    """Execute a manual operator action."""
    global _session
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    if not action:
        return jsonify({"status": "error", "message": "Missing 'action' field"}), 400

    if _session is None:
        try:
            spec = parse_mission_spec("configs/sample_mission.yaml")
            _session = MissionSession(spec)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    include_eval = bool(data.get("include_eval", False))
    try:
        state_data = _session.step(action=action)
        if include_eval:
            state_data["evaluation"] = _session.env.evaluation_only_map()
        return jsonify({
            "status": "ok",
            "action": action,
            "state": state_data,
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400


@app.route("/state", methods=["GET"])
def state():
    """Return current mission state for polling."""
    global _session
    if _session is None:
        return jsonify({"status": "no_mission"})

    include_eval = request.args.get("eval", "false").lower() in ("true", "1")
    return jsonify(_session.to_dict(include_eval=include_eval))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)