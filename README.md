# EVE Search Agent

An autonomous exploration and decision-making system: WALL-E is dropped into
an unknown 26x20 grid world and must locate EVE from her fading distress
signal, using only what its sensors can legally see.

```
        +------------------------------------+
        |  belief heatmap      path  pings   |
        |  ~~~~~~~~~~~~~~      ====   x     |
        |              WALL-E ==>  clue ♦    |
        |          posterior: P(EVE here)    |
        |  station ■   obstacle ▓  hazard ░  |
        +------------------------------------+
```

The agent never reads hidden ground truth. It maintains a Bayesian belief map
over EVE's position, collects noisy signal measurements and field clues,
plans battery-aware A* paths to the most informative frontier cells, keeps a
safety reserve for recharge stations, and prints an honest report with its
posterior — even when that means *"I could not find her."*

## Project objective & architecture

```
eve-search-agent/
├── README.md
├── pyproject.toml
├── configs/
│   └── sample_mission.yaml       # validated mission configuration
├── src/eve_search/
│   ├── models.py                 # shared enums & observation/trace models
│   ├── config.py                 # YAML -> Pydantic validation
│   ├── environment.py            # simulator; hidden ground truth lives here ONLY
│   ├── belief.py                 # Bayesian posterior over EVE cell + status
│   ├── planner.py                # A* over the *known* map with cost penalties
│   ├── agent.py                  # closed-loop decision maker + action trace
│   ├── reporting.py              # JSONL trace, JSON + Markdown reports
│   ├── visualization.py          # operational map (+ labelled evaluation view)
│   ├── sample.py                 # bundled sample mission YAML
│   └── cli.py                    # python -m eve_search.cli
├── tests/                        # pytest suite (58 tests)
└── outputs/                      # generated mission artifacts
```

| Module | Responsibility |
|---|---|
| `environment.py` | Grid world, movement/rotation/scan/ping/recharge actions, clue generation, signal propagation. Ground truth (EVE cell, status, clue placements) is private; everything leaves the module as `Observation` objects. |
| `belief.py` | Discrete posterior `P(EVE at cell)` + categorical `P(status)`. Updates from noisy strength/bearing physics and clue payloads (ring / bearing / region / status likelihoods). |
| `planner.py` | A* (4-connected) with obstacle blocking, hazard cost penalties, unknown-cell soft penalties, turn-count estimates. |
| `agent.py` | The decision loop: rescue → verify → pings → clue markers → frontier exploration, with battery reserve and mission-time budgeting. Records every action *while acting*. |
| `reporting.py` | `action_trace.jsonl`, `mission_report.json`, `mission_report.md`. |
| `visualization.py` | `mission_map.png` (agent knowledge only) and an optional explicitly-labelled evaluation-only view. |

## Environment & observation design

* **Grid world** with open terrain, obstacles (block movement), hazards (drain
  10% battery on entry, refundable via nothing — avoid them), and recharge
  stations (visible only when scanned).
* **Ground truth is hidden.** The environment holds `EVE position` and
  `EVE status` in private state; no public method returns it. What the agent
  sees:
  * `signal_measurement`: noisy strength + bearing pings, following
    `s(d) = s0 * exp(-d / λ)` with Gaussian noise.
  * `terrain_scan`: cell categories inside a Chebyshev radius. Unrevealed
    clue markers appear as `clue_marker` cells; the *evidence packet* itself
    is only collected when WALL-E stands within the collection radius.
  * `clue_discovered`: truthful but only-informative packets
    (`ring`: max distance from pickup, `bearing`: noisy compass hint,
    `region`: 3x3 sector, `status`: EVE state hint).
  * `blocked` / `recharge_report` / `status` bookkeeping.
* Deterministic seeds everywhere (`world.seed`, optional `evidence.seed`)
  make missions repeatable.

## YAML configuration schema

```yaml
mission:
  id: sample-long-range-rescue
  description: one-line mission story
  confidence_threshold: 0.12      # posterior mass to declare "probable contact"

world:
  width: 26
  height: 20
  seed: 42                        # deterministic world generation
  obstacle_density: 0.10          # used when explicit lists are omitted
  hazard_density: 0.04
  # explicit placements may replace generation:
  # obstacles: [[8, 3], [9, 3]]
  # hazards: [[12, 6]]
  # recharge_stations: [[1, 1], [13, 9], [23, 17]]

walle:
  start: [2, 2]                   # coordinates, 0-indexed
  orientation: E                  # N | E | S | W

signal:
  initial:                        # the distress-signal fragment
    bearing_deg: 38.0             # optional (noisy compass fragment)
    strength: 21.5                # optional
    coordinates: [14, 5]          # optional transponder fix
    timestamp: "2805-11-02T09:14:00Z"
  propagation:                    # public physics, not hidden state
    reference_strength: 100.0
    decay_length: 9.0
    noise_sigma: 7.0
    bearing_noise_deg: 12.0

resources:
  battery_capacity: 100.0
  initial_battery: 100.0
  mission_time: 340.0

costs:                            # battery + time per action
  move_battery: 1.2
  move_time: 1.5
  rotate_battery: 0.1
  rotate_time: 0.5
  scan_battery: 3.0
  scan_time: 4.0
  signal_ping_battery: 0.5
  signal_ping_time: 0.5
  wait_time: 1.0
  recharge_battery_gain: 22.0     # per recharge action
  recharge_time: 6.0
  blocked_move_penalty_time: 0.5

evidence:                         # hidden-world generation parameters
  clue_count: 6
  clue_spread: 10.0               # clues scattered within this distance of EVE
  scan_radius: 3
  scan_collection_radius: 1
  seed: 1337

eve:                              # HIDDEN ground truth (simulator-only)
  status: standby                 # dormant | standby | active | damaged
  position: [7, 16]               # optional; derived from seed when omitted
```

Validation (Pydantic) rejects unknown keys, out-of-bounds coordinates,
`initial_battery > capacity`, hazards on obstacles, duplicate stations,
unknown statuses, etc.

## Agent decision-making & belief updates

* **Belief**: a position posterior (battery of per-cell likelihoods, combined
  in log space) plus a status posterior. Each signal ping multiplies the map
  by `P(strength | d(cell, WALL-E)) * P(bearing | angle(cell, WALL-E))`; each
  clue applies a matching likelihood kernel (ring falloff, von-Mises-ish
  bearing cone, region membership, status emphasis).
* **Action selection**, in priority order, every step:
  1. **Rescue** — if battery is at/below the reserve (cost to the nearest
     *known* station * 1.25 + margin), divert and recharge; with no charted
     station, regroup at the mission start while still affordable.
  2. **Verify** — when `max posterior >= confidence_threshold`, close on the
     MAP estimate, ping once to confirm, then stop with outcome `located`
     (or `located_unverified` if the final approach is unaffordable).
  3. **Clue markers** approach any revealed `clue_marker` and scan in reach
     to lift the evidence packet.
  4. **Cadence pings** — every 4+ cells traversed, a cheap refresh.
  5. **Frontier exploration** — choose the reachable unknown cell that
     maximises `(cells revealed + belief mass at target) − (A* travel cost)`,
     re-planning each step, then scan on a 6-cell terrain cadence.
  6. **Fallbacks** — scan in place for markers/coverage, or wait while
     conserving (resources too tight to risk stranding systems).
* **Replanning** happens implicitly every step, using only the agent's own
  known map; blocked moves are logged once and never retried blindly.
* **Stopping**: sufficient evidence, exhausted battery/time, or an unsafe
  continuation. Uncertainty is reported honestly — no invented discoveries.

## Resource-management strategy

| Mechanism | Implementation |
|---|---|
| Movement budget | Full A* path + turn-count checked against battery/time *before* starting a leg |
| Safety reserve | `battery_needed_to_reach_nearest_station * 1.25 + 2` (25% capacity fallback before any station is charted) |
| Mission-time budgeting | Every leg must also fit inside `remaining_time − reserve_time` |
| Recharging | Only on station tiles; tops up to ~94% while time admits |
| Worth-it check | Frontier targets are scored against coverage + posterior mass − travel cost; unaffordable legs trigger cheaper sensing instead |

## Installation (macOS)

Requires Python 3.11+:

```bash
cd eve-search-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'          # core + matplotlib + pytest
```

## Run

```bash
# full sample mission
python -m eve_search.cli --config configs/sample_mission.yaml --output outputs/sample

# different reproducible world
python -m eve_search.cli --config configs/sample_mission.yaml --output outputs/seed7 --seed 7

# quiet mode / skip the ground-truth-labelled view
python -m eve_search.cli --config configs/sample_mission.yaml -o outputs/q --no-eval-view

# also installed as a console script
eve-search --config configs/sample_mission.yaml --output outputs/sample
```

## Interactive Web Dashboard

Launch the WALL-E mission control dashboard on port 8080:

```bash
python -m eve_search.web_app
# Open http://127.0.0.1:8080 in your browser
```

Features:
* Live interactive exploration grid with Bayesian belief heatmap overlay.
* Step-by-step single action execution or continuous autoplay.
* Manual operator control (move, turn, scan, ping, wait, recharge).
* Evaluation-only Ground Truth reveal mode strictly separated from agent beliefs.

## Tests

```bash
python -m pytest                 # 66 passing tests
```

Coverage includes configuration validation, action costs, hidden-state
isolation (an observation payload can never carry hidden fields — see
`test_agent.py::test_agent_never_accesses_hidden_state_via_public_interface`),
Bayesian updates, pathfinding around obstacles, battery/time limits,
recharge/return-to-safety behaviour, deterministic replay, full mission
execution, and trace/report generation.

## Trace & report formats

**`action_trace.jsonl`** — one JSON object per action:

```json
{
  "step": 42, "sim_time": 118.5, "position": [7, 13], "orientation": "S",
  "action": "signal_ping", "params": {"trigger": "cadence"}, "success": true,
  "observations": ["Signal ping: strength 43.7, bearing 168 deg."],
  "battery": 43.9, "remaining_time": 187.0,
  "belief_summary": {"before_max_posterior": 0.056, "after_map_estimate": [7, 15],
                     "after_max_posterior": 0.164, "after_entropy_nats": 3.11},
  "rationale": "4+ cells traversed since last reading; belief entropy 3.38 nats warrants a cheap update",
  "next_objective": "sharpen location belief"
}
```

**`mission_report.{json,md}`** — outcome, activity metrics, evidence observed
(raw sensor readings), inference (posterior estimate, credible region, status
probabilities, entropy — clearly marked as inference), resource/safety
assessment, and a recommended next action.

**`mission_map.png`** — scanned terrain, WALL-E path (blue), pings (x), scans
(▪), stations, collected clues (♦), belief heatmap and the posterior star.
`mission_map_evaluation_only.png` (skippable via `--no-eval-view`) adds
EVE's true position in an explicitly-magenta-labelled evaluation-only panel;
it is never part of observations, traces, or the agent's decisions.

## Sample mission results

`python -m eve_search.cli --config configs/sample_mission.yaml --output outputs/sample`
(seed 42) finishes in **54 actions** and reports:

```
outcome        located
inference      EVE @ [8, 16]  (posterior p = 0.272, 80% credible box x 7..9, y 16..17)
truth          EVE  @ [7, 16]   <- 1 cell away; evaluation-only, post-hoc
evidence       2 bearing clues + 1 proximity confirmation ping
resources      battery 39.4% / 100%, mission time 257/340 left
blocked moves  0
```

Alternative seeds complete without crashes and change the search story
(`test_agent.py::test_agent_adapts_to_different_map_and_seed`).

## Current limitations & future improvements

* Scan/ping budgeting is cadence-based rather than a fully learned policy;
  expected-information-gain is approximated from the posterior mean:
* Grid obstacles have no smooth occupancy model; unknown cells are assumed
  traversable and bad surprises are only recovered after the fact.
* EVE motion (she could wander, or discharge; status changes) is outside the
  current static-ground-truth model.
* No partial observability learning of propagation error ("noise_sigma" is
  truthfully known); a self-calibrating filter is a natural next step.
* The status posterior currently only gains mass from `status` clues; direct
  physical confirmation dialog is future work.
* Visualization is a static PNG; an animated replay of the trace would help
  debugging missions.
