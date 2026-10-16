"""Animated replay of a completed mission: WALL-E exploring, scanning, and
updating its Bayesian belief about EVE — frame by frame.

The animation is built ONLY from the recorded action trace: it replays each
trace entry through a fresh (deterministic) Environment/Agent pair so the
belief posterior, known terrain, battery, and time evolve exactly as they did
during the real run. No hidden ground truth is ever rendered; EVE's true
position stays out of the animation by design.

Frames are exported as an animated GIF (Pillow writer, no ffmpeg needed).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .agent import SearchAgent
from .config import MissionSpec
from .environment import Environment
from .models import ActionKind, TraceEntry


@dataclass
class ReplayFrame:
    step: int
    action: str
    note: str
    posterior: np.ndarray
    path: list[tuple[int, int]]
    pings: list[tuple[int, int]]
    scans: list[tuple[int, int]]
    known_cells: set[tuple[int, int]]
    known_obstacles: set[tuple[int, int]]
    known_hazards: set[tuple[int, int]]
    known_stations: set[tuple[int, int]]
    pending_markers: set[tuple[int, int]]
    collected_clues: list[str]
    position: tuple[int, int]
    battery: float
    capacity: float
    time_left: float
    time_budget: float
    map_estimate: tuple[int, int]
    max_posterior: float
    entropy: float


def record_replay(
    spec: MissionSpec, trace: Sequence[TraceEntry], seed: Optional[int] = None
) -> list[ReplayFrame]:
    """Deterministically replay the recorded trace and snapshot each step.

    The bookkeeping below mirrors ``SearchAgent.run()`` so the replayed state
    matches the original run; a final assertion guards against drift.
    """
    env = Environment(spec, seed=seed)
    agent = SearchAgent(env, spec)
    frames: list[ReplayFrame] = []

    def snapshot(entry: TraceEntry) -> ReplayFrame:
        return ReplayFrame(
            step=entry.step,
            action=entry.action,
            note="; ".join(entry.observations) or "(no observation returned)",
            posterior=agent.belief.posterior().copy(),
            path=list(agent.path_history),
            pings=list(agent.ping_positions),
            scans=list(agent.scan_positions),
            known_cells=set(agent.known_cells),
            known_obstacles=set(agent.known_obstacles),
            known_hazards=set(agent.known_hazards),
            known_stations=set(agent.known_stations),
            pending_markers=set(agent.pending_clue_markers),
            collected_clues=[c.text for c in agent.collected_clues],
            position=env.current_position,
            battery=env.battery,
            capacity=spec.resources.battery_capacity,
            time_left=env.remaining_time,
            time_budget=spec.resources.mission_time,
            map_estimate=agent.belief.map_estimate(),
            max_posterior=agent.belief.max_posterior(),
            entropy=agent.belief.entropy(),
        )

    for entry in trace:
        action = ActionKind(entry.action)
        result = env.apply_action(action, dict(entry.params))
        agent._observe(result)
        agent.steps_since_ping += 1
        agent.moves_since_scan += 1

        if action is ActionKind.MOVE_FORWARD:
            if result.success:
                agent.moves_since_ping += 1
                agent.successful_moves += 1
                agent.distance_traveled += 1
                agent.path_history.append(result.position)
                agent.known_cells.add(result.position)
                agent.pending_clue_markers.discard(result.position)
            else:
                agent.blocked_moves += 1
        elif action is ActionKind.SCAN:
            agent.scan_positions.append(result.position)
            agent.scans_done += 1
            agent.moves_since_scan = 0
            agent.moves_since_ping = 0
        elif action is ActionKind.SIGNAL_PING:
            agent.steps_since_ping = 0
            if result.success:
                agent.moves_since_ping = 0

        frames.append(snapshot(entry))

    # Drift guard: replay must land where the original run ended.
    if env.current_position != trace[-1].position:
        raise RuntimeError(
            f"replay drifted: ended at {env.current_position} "
            f"but trace ended at {trace[-1].position}"
        )
    return frames


def save_mission_gif(
    spec: MissionSpec,
    trace: Sequence[TraceEntry],
    out_dir: Path,
    seed: Optional[int] = None,
    fps: float = 3.0,
    dpi: int = 90,
) -> Optional[Path]:
    """Render the replay as ``mission_animation.gif``. Returns the path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle
    except ImportError:
        return None

    frames = record_replay(spec, trace, seed=seed)
    if not frames:
        return None
    w, h = spec.world.width, spec.world.height
    vmax = max(max(f.posterior.max() for f in frames), 1e-6)

    fig, (ax, axt) = plt.subplots(
        1, 2, figsize=(11.5, 5.8), gridspec_kw={"width_ratios": [2.1, 1]}
    )

    def draw_frame(i: int) -> None:
        ax.clear()
        axt.clear()
        f = frames[i]

        # ---- MAP PANEL ---------------------------------------------------
        for (x, y) in f.known_cells:
            face = "#f2f2ee"
            edge = "#dddddd"
            if (x, y) in f.known_obstacles:
                face, edge = "#2f2f2f", "#2f2f2f"
            elif (x, y) in f.known_hazards:
                face, edge = "#f0b455", "#c98b2c"
            elif (x, y) in f.known_stations:
                face, edge = "#7fdca4", "#2ea86a"
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor=edge, lw=0.2))

        heat = ax.imshow(
            f.posterior, origin="upper", cmap="YlOrRd", alpha=0.6,
            vmin=0.0, vmax=vmax, extent=(-0.5, w - 0.5, h - 0.5, -0.5),
        )
        for pos in f.scans:
            ax.plot(pos[0], pos[1], "s", color="#8e44ad", markersize=3.5, alpha=0.55)
        for pos in f.pings:
            ax.plot(pos[0], pos[1], "x", color="#0b87c4", markersize=5, alpha=0.8)
        for m in f.pending_markers:
            ax.plot(m[0], m[1], "d", color="#e6b800", markersize=9, markeredgecolor="black", markeredgewidth=0.6)

        if len(f.path) >= 2:
            px = [c[0] for c in f.path]
            py = [c[1] for c in f.path]
            ax.plot(px, py, "-", color="#0b87c4", lw=1.6, alpha=0.9)
        start = spec.walle.start
        ax.plot(start[0], start[1], "o", color="white", markeredgecolor="black", ms=8, zorder=5)
        ax.plot(f.position[0], f.position[1], "o", color="#0b87c4", markeredgecolor="black", ms=11, zorder=6)
        ax.plot(f.map_estimate[0], f.map_estimate[1], "*", color="red", ms=16, zorder=5)

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"step {f.step}/{len(frames)}  |  {f.action}", fontsize=10)
        fig.colorbar(heat, ax=ax, shrink=0.7, label="P(EVE here)")

        # ---- STATUS PANEL ------------------------------------------------
        axt.set_xlim(0, 1)
        axt.set_ylim(0, 1)
        axt.axis("off")

        def bar(y, frac, label, color):
            axt.add_patch(Rectangle((0.05, y), 0.9, 0.07, facecolor="#eeeeee", edgecolor="#999999"))
            frac = max(0.0, min(1.0, frac))
            axt.add_patch(Rectangle((0.05, y), 0.9 * frac, 0.07, facecolor=color, edgecolor="none"))
            axt.text(0.05, y + 0.10, label, fontsize=8.5)

        bat_col = "#2ea86a" if f.battery > 0.35 * f.capacity else ("#e0a33a" if f.battery > 0.15 * f.capacity else "#d03a2b")
        bar(0.86, f.battery / f.capacity, f"battery  {f.battery:.1f}% / {f.capacity:.0f}%", bat_col)
        t_col = "#2ea86a" if f.time_left > 0.30 * f.time_budget else ("#e0a33a" if f.time_left > 0.10 * f.time_budget else "#d03b2b")
        bar(0.66, f.time_left / f.time_budget, f"mission time  {f.time_left:.0f} / {f.time_budget:.0f}", t_col)

        reg_line = "none yet"
        if f.collected_clues:
            reg_line = " | ".join(c[:34] for c in f.collected_clues[-2:])
        lines = [
            f"belief MAP estimate: cell {f.map_estimate}",
            f"posterior peak: p = {f.max_posterior:.3f}",
            f"entropy: {f.entropy:.2f} nats",
            f"cells known: {len(f.known_cells)}  |  clues: {len(f.collected_clues)}",
            f"markers in sight: {len(f.pending_markers)}",
            f"clues: {reg_line}",
        ]
        for k, text in enumerate(lines):
            axt.text(0.05, 0.50 - 0.06 * k, text, fontsize=8.5)
        obs = "\n".join(textwrap.wrap(f"observation: {f.note}", width=42)[:4])
        axt.text(0.05, 0.05, obs, fontsize=7.0, color="#333333")
        axt.set_title("WALL-E status", fontsize=10)

    def update(i: int):
        draw_frame(i)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=350, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mission_animation.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
