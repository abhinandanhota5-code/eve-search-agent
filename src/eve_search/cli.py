"""CLI: run a full EVE Search Agent mission.

Usage::

    python -m eve_search.cli --config configs/sample_mission.yaml --output outputs/sample
    python -m eve_search.cli --config configs/sample_mission.yaml --seed 7 --no-eval-view

The CLI loads the validated mission YAML, runs the autonomous agent to
completion, prints a concise progress summary, and writes:

* ``action_trace.jsonl`` - every action + observation + rationale
* ``mission_report.json`` - machine-readable final report
* ``mission_report.md`` - human-readable final report
* ``mission_map.png`` - operational visualization (when matplotlib is present)
* ``mission_map_evaluation_only.png`` - optional, explicitly labelled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import MissionSpec, load_mission
from .environment import Environment
from .agent import SearchAgent
from .reporting import build_report, save_report, save_trace
from .visualization import save_mission_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="eve-search",
        description="Run an autonomous WALL-E mission to locate EVE.",
    )
    ap.add_argument("--config", "-c", required=True, help="Path to the mission YAML file")
    ap.add_argument("--output", "-o", default="outputs/run", help="Output directory")
    ap.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Override the world random seed for reproducible alternate runs",
    )
    ap.add_argument(
        "--max-steps", type=int, default=4000,
        help="Hard cap on simulation actions (safety net)",
    )
    ap.add_argument(
        "--no-eval-view", action="store_true",
        help="Skip the explicitly-labelled evaluation-only map",
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Print no progress summary")
    ap.add_argument(
        "--animate", action="store_true",
        help="Also render mission_animation.gif replaying WALL-E's run",
    )
    ap.add_argument(
        "--sample-config", action="store_true",
        help="Write the bundled sample mission YAML to the given --config path and exit",
    )
    return ap.parse_args(argv)


def run_mission(config_path: Path, output_dir: Path, seed: int | None = None,
                max_steps: int = 4000, include_eval: bool = True, quiet: bool = False,
                animate: bool = False) -> dict:
    spec: MissionSpec = load_mission(config_path)
    env = Environment(spec, seed=seed)
    agent = SearchAgent(env, spec, max_steps=max_steps)

    summary = agent.run()

    trace_path = save_trace(agent, output_dir)
    report = build_report(spec, env, agent, summary)
    json_path, md_path = save_report(report, output_dir)
    map_path = save_mission_map(env, agent, output_dir, include_evaluation_view=include_eval)
    gif_path = None
    if animate:
        from .animation import save_mission_gif
        gif_path = save_mission_gif(spec, agent.trace, output_dir, seed=seed)

    if not quiet:
        print(f"[eve-search] mission {spec.mission.id} finished: outcome={report['outcome']}")
        print(f"  steps={report['activity']['steps']}  distance={report['activity']['distance_explored']}"
              f"  scans={report['activity']['scans']}  pings={report['activity']['signal_pings']}"
              f"  clues={len(agent.collected_clues)}")
        est = report["inference"]["inferred_location_estimate"]
        print(f"  inference: EVE @ {est} (p={report['inference']['confidence']:.3f}, "
              f"region={report['inference']['region_of_estimate']})")
        print(f"  resources: battery={report['resources']['battery_remaining_pct']}% "
              f"time_left={report['resources']['mission_time_remaining']}")
        print(f"  artifacts: {trace_path.name}, {json_path.name}, {md_path.name}"
              + (f", {map_path.name}" if map_path else ", (matplotlib not installed: no map)")
              + (f", {gif_path.name}" if gif_path else "")
              + (", mission_map_evaluation_only.png" if include_eval and map_path else ""))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_config:
        from .sample import write_sample_mission

        write_sample_mission(Path(args.config))
        print(f"sample mission written to {args.config}")
        return 0
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2
    try:
        run_mission(
            config_path,
            Path(args.output),
            seed=args.seed,
            max_steps=args.max_steps,
            include_eval=not args.no_eval_view,
            quiet=args.quiet,
            animate=args.animate,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface validation/agent errors clearly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
