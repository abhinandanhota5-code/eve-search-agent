"""Optional Matplotlib visualization of a completed mission.

The default view shows only agent-side knowledge: terrain WALL-E scanned, his
path, clue pickups, stations, and the belief heatmap. EVE's true location is
drawn only in an explicitly labelled **evaluation-only** figure fed from
``Environment.evaluation_only_map()`` -- never in the operational view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .agent import SearchAgent
from .environment import Environment
from .models import Orientation


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def save_mission_map(
    env: Environment,
    agent: SearchAgent,
    out_dir: Path,
    include_evaluation_view: bool = False,
) -> Optional[Path]:
    """Draw the operational map (no ground truth). Returns the image path."""
    plt = _import_pyplot()
    if plt is None:
        return None

    spec = env.spec
    w, h = env.width, env.height
    fig, ax = plt.subplots(figsize=(1.1 * w / 3.0, 1.1 * h / 3.0))

    # --- agent-known terrain -------------------------------------------
    for (x, y) in sorted(agent.known_cells):
        if (x, y) in agent.known_obstacles:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#3b3b3b"))
        elif (x, y) in agent.known_hazards:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#e08a1e", alpha=0.7))
        elif (x, y) in agent.known_stations:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#2ecc71"))

    # --- belief heatmap --------------------------------------------------
    heat = np.zeros((h, w))
    p = agent.belief.posterior()
    heat[:, :] = p
    im = ax.imshow(heat, origin="upper", cmap="viridis", alpha=0.55,
                   extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    # --- path -----------------------------------------------------------
    ph = agent.path_history
    if len(ph) >= 2:
        px = [c[0] for c in ph]
        py = [c[1] for c in ph]
        ax.plot(px, py, "-", color="#00b3ff", linewidth=1.4, alpha=0.9, label="WALL-E path")
    for pos in agent.ping_positions:
        ax.plot(pos[0], pos[1], "x", color="#00b3ff", markersize=4, alpha=0.6)
    for pos in agent.scan_positions:
        ax.plot(pos[0], pos[1], "s", color="#9b59b6", markersize=3, alpha=0.5)

    # --- collected clues -------------------------------------------------
    for clue in agent.collected_clues:
        if clue.pickup:
            ax.plot(clue.pickup[0], clue.pickup[1], "d", color="#f1c40f", markersize=8)

    # --- WALL-E start & end ---------------------------------------------
    start = spec.walle.start
    ax.plot(start[0], start[1], "o", color="white", markeredgecolor="black", markersize=9, label="start")
    end = env.current_position
    ax.plot(end[0], end[1], "o", color="#00b3ff", markeredgecolor="black", markersize=9, label="WALL-E end")

    est = agent.belief.map_estimate()
    ax.plot(est[0], est[1], "*", color="red", markersize=14, label="inferred EVE (posterior)")

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Mission {spec.mission.id} - operational view (agent knowledge only)")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, label="P(EVE here)", shrink=0.75)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mission_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    if include_evaluation_view:
        _save_evaluation_map(env, agent, out_dir)
    return path


def _save_evaluation_map(env: Environment, agent: SearchAgent, out_dir: Path) -> Optional[Path]:
    """EVALUATION-ONLY duplicate that also marks EVE's true position."""
    plt = _import_pyplot()
    if plt is None:
        return None
    truth = env.evaluation_only_map()
    spec = env.spec
    w, h = env.width, env.height
    fig, ax = plt.subplots(figsize=(1.1 * w / 3.0, 1.1 * h / 3.0))

    for (x, y) in truth["obstacles"]:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#3b3b3b"))
    for (x, y) in truth["hazards"]:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#e08a1e", alpha=0.7))
    for (x, y) in truth["stations"]:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#2ecc71"))
    for (x, y) in truth["clue_positions"]:
        ax.plot(x, y, "d", color="#f1c40f", markersize=7)

    heat = agent.belief.posterior()
    ax.imshow(heat, origin="upper", cmap="viridis", alpha=0.4,
              extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    ph = agent.path_history
    if len(ph) >= 2:
        ax.plot([c[0] for c in ph], [c[1] for c in ph], "-", color="#00b3ff", linewidth=1.2, alpha=0.85)

    ev = truth["eve_position"]
    ax.plot(ev[0], ev[1], "P", color="magenta", markersize=16, markeredgecolor="black",
            label="EVE TRUE POSITION (evaluation only)")
    ax.set_title("EVALUATION-ONLY VIEW - ground truth; never used by the agent")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")

    path = out_dir / "mission_map_evaluation_only.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
