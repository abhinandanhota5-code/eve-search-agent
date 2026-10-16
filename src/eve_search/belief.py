"""Bayesian belief state over EVE's location and status.

The belief is a discrete probability map (one weight per grid cell) plus a
small categorical posterior over EVE's possible statuses. Evidence sources:

* signal pings (noisy strength + bearing) -> likelihood over cells
* collected clues (ring / bearing / region / status hints)
* the initial distress-signal fragment

All likelihood uses the *public* signal physics from the mission config.
Nothing in this module touches environment ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .models import (
    EVE_STATUSES,
    Clue,
    angular_distance_deg,
    bearing_between,
    region_name,
)

# Region adjacency softening for qualitative hints.
_REGION_LIKELIHOOD_SELF = 1.0
_REGION_LIKELIHOOD_NEIGHBOUR = 0.35
_REGION_LIKELIHOOD_OTHER = 0.05


class BeliefState:
    """Normalised Bayesian posterior over EVE's cell and status."""

    def __init__(self, width: int, height: int, statuses: tuple[str, ...] = EVE_STATUSES):
        self.width = width
        self.height = height
        self.statuses = tuple(statuses)
        self._log_weights = np.full((height, width), -math.log(width * height))
        self._status = {s: 1.0 / len(statuses) for s in statuses}
        self._region_grid = np.zeros((height, width), dtype=int)
        for y in range(height):
            for x in range(width):
                self._region_grid[y, x] = self._region_index(x, y)

    # ------------------------------------------------------------------
    # Renormalisation
    # ------------------------------------------------------------------
    def _normalize(self) -> None:
        m = self._log_weights.max()
        w = np.exp(self._log_weights - m)
        total = w.sum()
        self._log_weights = np.log(w / total + 1e-300)

    # ------------------------------------------------------------------
    # Likelihood updates
    # ------------------------------------------------------------------
    def apply_signal(self, strength: float, bearing_deg: Optional[float], from_pos: tuple[int, int], physics: dict[str, float]) -> None:
        """Bayes update from a noisy (strength, bearing) ping at ``from_pos``."""
        fx, fy = from_pos
        xs = np.arange(self.width, dtype=float)
        ys = np.arange(self.height, dtype=float)
        gx, gy = np.meshgrid(xs, ys)  # shape (h, w)
        dist = np.hypot(gx - fx, gy - fy)

        mu = physics["reference_strength"] * np.exp(-dist / physics["decay_length"])
        sigma = physics["noise_sigma"]
        strength_like = np.exp(-((strength - mu) ** 2) / (2.0 * sigma * sigma))

        like_log = np.log(strength_like + 1e-300)
        if bearing_deg is not None:
            dx = gx - fx
            dy = fy - gy  # north (decreasing y) positive
            cell_bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
            ang_diff = np.abs(cell_bearing - bearing_deg) % 360.0
            ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
            sigma_rad = math.radians(physics["bearing_noise_deg"])
            kappa = 1.0 / max(sigma_rad * sigma_rad, 1e-9)
            bearing_like = np.exp(kappa * np.cos(np.radians(ang_diff)))
            like_log = like_log + np.log(bearing_like + 1e-300)

        self._log_weights = self._log_weights + like_log
        self._normalize()

    def apply_initial_signal(
        self,
        coordinates: Optional[tuple[int, int]],
        bearing_deg: Optional[float],
        strength: Optional[float],
        from_pos: tuple[int, int],
        physics: dict[str, float],
    ) -> None:
        """Incorporate the mission-start signal fragment."""
        if coordinates is not None:
            # Trusted-but-noisy transponder fix: Gaussian around the fix.
            sigma = max(2.0, physics["decay_length"] / 2.0)
            xs = np.arange(self.width, dtype=float)
            ys = np.arange(self.height, dtype=float)
            gx, gy = np.meshgrid(xs, ys)
            d2 = (gx - coordinates[0]) ** 2 + (gy - coordinates[1]) ** 2
            like = np.exp(-d2 / (2 * sigma * sigma))
            self._log_weights = self._log_weights + np.log(like + 1e-300)
            self._normalize()
        if strength is not None:
            # A signal strength reading implies a noisy distance band from start.
            est_d = -physics["decay_length"] * math.log(max(strength, 1e-6) / physics["reference_strength"])
            est_d = min(max(est_d, 0.0), physics["decay_length"] * 4)
            mu_at_est = physics["reference_strength"] * math.exp(-est_d / physics["decay_length"])
            sigma_d = physics["noise_sigma"] * physics["decay_length"] / max(mu_at_est, 1e-6)
            sigma_d = max(sigma_d, 1.0)
            xs = np.arange(self.width, dtype=float)
            ys = np.arange(self.height, dtype=float)
            gx, gy = np.meshgrid(xs, ys)
            d = np.hypot(gx - from_pos[0], gy - from_pos[1])
            like = np.exp(-((d - est_d) ** 2) / (2 * sigma_d * sigma_d))
            self._log_weights = self._log_weights + np.log(like + 1e-300)
            self._normalize()
        if coordinates is None and bearing_deg is not None:
            # Angular fragment: combine with whatever distance band the
            # strength reading already imposed.
            self._apply_bearing_only(bearing_deg, from_pos)

    def _apply_bearing_only(self, bearing_deg: float, from_pos: tuple[int, int]) -> None:
        xs = np.arange(self.width, dtype=float)
        ys = np.arange(self.height, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        dx = gx - from_pos[0]
        dy = from_pos[1] - gy
        cell_bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
        ang_diff = np.abs(cell_bearing - bearing_deg) % 360.0
        ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
        sigma_deg = 15.0
        sigma_rad = math.radians(sigma_deg)
        kappa = 1.0 / max(sigma_rad * sigma_rad, 1e-9)
        like = np.exp(kappa * np.cos(np.radians(ang_diff)))
        self._log_weights = self._log_weights + np.log(like + 1e-300)
        self._normalize()

    def apply_clue(self, clue: Clue) -> None:
        """Fold one collected clue into the belief. Evidence, not truth."""
        kind = clue.kind
        payload = clue.payload
        if clue.pickup is None:
            return
        px, py = clue.pickup

        if kind.value == "ring":
            radius = float(payload.get("radius", 0.0))
            sigma = max(1.0, 0.25 * radius)
            xs = np.arange(self.width, dtype=float)
            ys = np.arange(self.height, dtype=float)
            gx, gy = np.meshgrid(xs, ys)
            d = np.hypot(gx - px, gy - py)
            over = np.clip(d - radius, 0.0, None)
            like = np.exp(-(over ** 2) / (2 * sigma * sigma))
            self._log_weights = self._log_weights + np.log(like + 1e-300)
            self._normalize()
        elif kind.value == "bearing":
            bearing = float(payload.get("bearing_deg", 0.0))
            sigma_deg = float(payload.get("sigma_deg", 15.0))
            xs = np.arange(self.width, dtype=float)
            ys = np.arange(self.height, dtype=float)
            gx, gy = np.meshgrid(xs, ys)
            dx = gx - px
            dy = py - gy
            cell_bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
            ang_diff = np.abs(cell_bearing - bearing) % 360.0
            ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
            sigma_rad = math.radians(max(sigma_deg, 1.0))
            kappa = 1.0 / max(sigma_rad * sigma_rad, 1e-9)
            like = np.exp(kappa * np.cos(np.radians(ang_diff)))
            self._log_weights = self._log_weights + np.log(like + 1e-300)
            self._normalize()
        elif kind.value == "region":
            target = self._region_index_from_name(payload.get("region", "centre"))
            region_like = np.full((self.height, self.width), _REGION_LIKELIHOOD_OTHER)
            neighbours = _adjacent_regions(target)
            self_like = self._region_grid == target
            nb_like = np.isin(self._region_grid, list(neighbours))
            region_like[self_like] = _REGION_LIKELIHOOD_SELF
            region_like[nb_like] = _REGION_LIKELIHOOD_NEIGHBOUR
            self._log_weights = self._log_weights + np.log(region_like + 1e-300)
            self._normalize()
        elif kind.value == "status":
            status = payload.get("status")
            if status in self.statuses:
                conf = 0.85
                frac = (1.0 - conf) / (len(self.statuses) - 1)
                new = {s: (conf if s == status else frac) for s in self.statuses}
                self._status = {s: self._status[s] * new[s] for s in self.statuses}
                total = sum(self._status.values())
                self._status = {s: v / total for s, v in self._status.items()}
        else:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def posterior(self) -> np.ndarray:
        m = self._log_weights.max()
        w = np.exp(self._log_weights - m)
        return w / w.sum()

    def map_estimate(self) -> tuple[int, int]:
        p = self.posterior()
        y, x = np.unravel_index(int(np.argmax(p)), p.shape)
        return (int(x), int(y))

    def entropy(self) -> float:
        """Posterior entropy in nats (location only)."""
        p = self.posterior().ravel()
        nz = p[p > 1e-12]
        return float(-(nz * np.log(nz)).sum())

    def max_posterior(self) -> float:
        return float(self.posterior().max())

    def mass_within(self, centre: tuple[float, float], radius: float) -> float:
        """Posterior mass within ``radius`` (euclidean) of ``centre``."""
        p = self.posterior()
        xs = np.arange(self.width, dtype=float)
        ys = np.arange(self.height, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        d = np.hypot(gx - centre[0], gy - centre[1])
        mask = d <= radius
        return float(p[mask].sum())

    def status_posterior(self) -> dict[str, float]:
        return dict(self._status)

    def region_posterior(self) -> dict[str, float]:
        p = self.posterior()
        out: dict[str, float] = {}
        for idx, name in enumerate(_REGION_NAME_LIST):
            out[name] = float(p[self._region_grid == idx].sum())
        return out

    def top_cells(self, k: int = 5) -> list[tuple[tuple[int, int], float]]:
        p = self.posterior()
        flat_idx = np.argsort(p.ravel())[::-1][:k]
        out = []
        for fi in flat_idx:
            y, x = np.unravel_index(fi, p.shape)
            out.append(((int(x), int(y)), float(p[y, x])))
        return out

    def summary(self) -> dict[str, Any]:
        est = self.map_estimate()
        return {
            "map_estimate": list(est),
            "max_posterior": round(self.max_posterior(), 5),
            "entropy_nats": round(self.entropy(), 4),
            "mass_within_3": round(self.mass_within(est, 3.0), 4),
            "region_posterior": {k: round(v, 4) for k, v in self.region_posterior().items() if v > 1e-4},
            "status_posterior": {k: round(v, 4) for k, v in self.status_posterior().items()},
            "top_cells": [[list(c), round(m, 5)] for c, m in self.top_cells(3)],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _region_index(self, x: int, y: int) -> int:
        col = 0 if x < self.width / 3 else (1 if x < 2 * self.width / 3 else 2)
        row = 0 if y < self.height / 3 else (1 if y < 2 * self.height / 3 else 2)
        return row * 3 + col

    def _region_index_from_name(self, name: str) -> int:
        if name in _REGION_NAME_LIST:
            return _REGION_NAME_LIST.index(name)
        return 4


_REGION_NAME_LIST = (
    "north-west", "north", "north-east",
    "west", "centre", "east",
    "south-west", "south", "south-east",
)


def _adjacent_regions(idx: int) -> set[int]:
    row, col = divmod(idx, 3)
    out: set[int] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row + dr, col + dc
            if 0 <= r < 3 and 0 <= c < 3 and (dr or dc):
                out.add(r * 3 + c)
    return out
