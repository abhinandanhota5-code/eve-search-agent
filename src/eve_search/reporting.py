"""Mission artifacts: action trace (JSONL) and final report (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .agent import SearchAgent
from .belief import BeliefState
from .environment import Environment
from .models import REGION_NAMES, TraceEntry, region_name


def save_trace(agent: SearchAgent, out_dir: Path) -> Path:
    """Write ``action_trace.jsonl`` - one JSON object per recorded action."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "action_trace.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in agent.trace:
            fh.write(entry.model_dump_json() + "\n")
    return path


def build_report(
    spec,
    env: Environment,
    agent: SearchAgent,
    summary: dict,
    extra_trace: Optional[list[TraceEntry]] = None,
) -> dict:
    """Assemble the final mission report dict (JSON-ready).

    Observed evidence and model inferences are kept in clearly separated
    sections.
    """
    belief = agent.belief
    est = belief.map_estimate()
    p = belief.posterior()
    conf = belief.max_posterior()
    threshold = spec.mission.confidence_threshold
    outcome = summary["outcome"]

    # 80% credible region: smallest set of cells holding 80% of the mass
    flat = sorted(
        ((float(p[y, x]), int(x), int(y)) for y in range(p.shape[0]) for x in range(p.shape[1])),
        key=lambda t: -t[0],
    )
    cred_cells: list[list[int]] = []
    acc = 0.0
    for m, x, y in flat:
        if acc >= 0.80:
            break
        cred_cells.append([x, y])
        acc += m
    if cred_cells:
        xs = [c[0] for c in cred_cells]
        ys = [c[1] for c in cred_cells]
        region = {
            "kind": "bounding_box",
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "cells": len(cred_cells),
            "posterior_mass": round(acc, 4),
        }
    else:  # pragma: no cover
        region = {"kind": "empty"}

    status_post = belief.status_posterior()
    region_post = belief.region_posterior()
    best_region = max(region_post, key=lambda k: region_post[k])

    evidence_observed = []
    for clue in agent.collected_clues:
        evidence_observed.append({
            "source": "collected_clue",
            "id": clue.id,
            "kind": clue.kind.value,
            "text": clue.text,
            "pickup": list(clue.pickup) if clue.pickup else None,
        })
    for pos in agent.ping_positions[-5:]:
        evidence_observed.append({
            "source": "signal_ping",
            "text": f"noisy strength/bearing ping recorded at {list(pos)}",
        })

    recharge_ok = bool(agent.known_stations)
    safety_ok = agent.env.battery > spec.resources.battery_capacity * 0.15
    if outcome.startswith("located") or outcome.startswith("uncertain_returned"):
        rec = (
            f"Dispatch a recovery drone to {list(est)} ({best_region} sector); "
            f"carry spare cells - signals decayed strongly near the estimate."
        )
    elif outcome in ("battery_exhausted", "time_exhausted"):
        rec = (
            "Refit WALL-E (full cells, hazard shielding) and re-launch from the "
            f"last confirmed scan area near {best_region} sector with the "
            "posterior map as prior."
        )
    else:
        rec = (
            "Continue the sweep but widen the ping cadence near the "
            f"{best_region} sector; the posterior is diffuse and more "
            "independent readings will cut it fastest."
        )

    report = {
        "mission": {
            "id": spec.mission.id,
            "description": spec.mission.description,
            "world": f"{spec.world.width}x{spec.world.height}",
            "seed": spec.rng_seed,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "outcome_definition": {
            "located": f"posterior max {conf:.3f} reached confidence threshold {threshold}",
        },
        "resources": {
            "battery_remaining_pct": round(env.battery, 2),
            "battery_capacity": spec.resources.battery_capacity,
            "mission_time_remaining": round(env.remaining_time, 2),
            "mission_time_budget": spec.resources.mission_time,
        },
        "activity": {
            "steps": len(agent.trace),
            "distance_explored": agent.distance_traveled,
            "successful_moves": agent.successful_moves,
            "blocked_moves": agent.blocked_moves,
            "scans": agent.scans_done,
            "signal_pings": agent.pings_done,
            "recharges": agent.recharges_done,
            "regions_visited": sorted(
                {region_name(x, y, spec.world.width, spec.world.height)
                 for (x, y) in agent.path_history[:: max(1, len(agent.path_history) // 40 or 1)]}
            ),
        },
        "evidence_observed": evidence_observed,
        "inference": {
            "inferred_location_estimate": list(est),
            "region_of_estimate": region_name(est[0], est[1], spec.world.width, spec.world.height),
            "confidence": round(conf, 4),
            "confidence_threshold": threshold,
            "credible_region_80": region,
            "status_probabilities": {k: round(v, 4) for k, v in status_post.items()},
            "region_probabilities": {k: round(v, 4) for k, v in region_post.items() if v > 1e-3},
            "posterior_entropy_nats": round(belief.entropy(), 3),
            "note": (
                "Location is a posterior estimate, NOT a verified contact. "
                "EVE's true coordinates were never observed."
                if not outcome.startswith("located_confirmed") else
                "Location verified by proximity + final signal readings."
            ),
        },
        "resource_assessment": {
            "recharge_stations_discovered": len(agent.known_stations),
            "battery_reserve_policy": "keep >= cost-to-nearest-station * 1.25",
            "hazard_penalties_taken": bool(agent.known_hazards),
            "battery_fell_below_15pct": not safety_ok,
            "assessment": (
                "satisfactory"
                if safety_ok and agent.blocked_moves < agent.successful_moves * 0.5
                else "must improve (battery near reserve or frequent blocked moves)"
            ),
        },
        "recommended_next_action": rec,
    }
    return report


def save_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mission_report.json"
    md_path = out_dir / "mission_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def _to_markdown(r: dict) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# EVE Search Agent - Mission Report: {r['mission']['id']}")
    a("")
    if r["mission"]["description"]:
        a(f"> {r['mission']['description']}")
        a("")
    a(f"*Generated {r['generated_at']} on a {r['mission']['world']} map (seed {r['mission']['seed']})*")
    a("")
    emoji = {"located": "targeting", "located_unverified": "targeting"}.get(
        r["outcome"], "partial"
    )
    a(f"## Outcome: `{r['outcome']}`" + (f" ({emoji})" if emoji != "partial" else ""))
    if r["outcome"] == "located":
        a("")
        a(r["outcome_definition"].get("located", ""))
    a("")
    a("## Activity")
    act = r["activity"]
    a("")
    a(f"| Metric | Value |")
    a(f"|---|---|")
    a(f"| Steps taken | {act['steps']} |")
    a(f"| Cells traveled | {act['distance_explored']} |")
    a(f"| Successful moves | {act['successful_moves']} |")
    a(f"| Blocked moves | {act['blocked_moves']} |")
    a(f"| Local scans | {act['scans']} |")
    a(f"| Signal pings | {act['signal_pings']} |")
    a(f"| Recharges | {act['recharges']} |")
    a("")
    a(f"**Battery left:** {r['resources']['battery_remaining_pct']}% of "
      f"{r['resources']['battery_capacity']:.0f}% capacity")
    a(f"**Mission time left:** {r['resources']['mission_time_remaining']} of "
      f"{r['resources']['mission_time_budget']:.0f}")
    a("")
    a("## Evidence observed (WALL-E's own readings)")
    a("")
    if r["evidence_observed"]:
        for e in r["evidence_observed"]:
            a(f"- **[{e['source']}]** {e['text']}" + (f" (pickup {e['pickup']})" if e.get("pickup") else ""))
    else:
        a("- none collected")
    a("")
    a("## Inference (model posterior - NOT direct observation)")
    a("")
    inf = r["inference"]
    a(f"- **Most likely location:** cell {inf['inferred_location_estimate']} "
      f"({inf['region_of_estimate']} sector)")
    a(f"- **Confidence:** {inf['confidence']:.3f} (threshold {inf['confidence_threshold']})")
    cr = inf["credible_region_80"]
    if cr.get("kind") == "bounding_box":
        a(f"- **80% credible region:** bounding box "
          f"[x {cr['x_min']}..{cr['x_max']}, y {cr['y_min']}..{cr['y_max']}] "
          f"({cr['cells']} cells, mass {cr['posterior_mass']})")
    a(f"- **Status probabilities:** " + ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(inf["status_probabilities"].items(), key=lambda kv: -kv[1])
    ))
    top_regions = sorted(inf["region_probabilities"].items(), key=lambda kv: -kv[1])[:3]
    a(f"- **Top regions:** " + ", ".join(f"{k} {v:.2f}" for k, v in top_regions))
    a(f"- **Posterior entropy:** {inf['posterior_entropy_nats']} nats")
    a("")
    a(f"_{inf['note']}_")
    a("")
    a("## Safety & resource management")
    a("")
    ra = r["resource_assessment"]
    a(f"- Recharge stations discovered: {ra['recharge_stations_discovered']}")
    a(f"- Reserve policy: {ra['battery_reserve_policy']}")
    a(f"- Assessment: **{ra['assessment']}**")
    a("")
    a("## Recommended next action")
    a("")
    a(r["recommended_next_action"])
    a("")
    return "\n".join(lines)
