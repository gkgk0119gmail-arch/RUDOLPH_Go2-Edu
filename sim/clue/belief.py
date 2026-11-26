"""Victim belief map on the 2D planning grid: clue -> prior, spread over time, camera observations, frontier mass.

Grid states: FREE / OCC (known wall or obstacle) / UNK (not mapped yet). Unknown cells are assumed passable, so mass
can sit and move behind frontiers; known walls block both spreading and frontier assignment.

Spreading = the prediction step of Bayesian search for a moving target: each step a fraction of every cell's mass
moves to its passable 4-neighbours (discrete diffusion, coefficient D m^2/s). D comes from what the speaker said:
"갇혀/깔려/쓰러져" -> static (D = 0), "돌아다녀요" -> mobile, otherwise unknown.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FREE, OCC, UNK = 0, 1, 2
D_MOBILITY = {"static": 0.0, "unknown": 0.01, "mobile": 0.1}  # m^2/s; sigma after t: sqrt(2 D t) (10 min: 0, 3.5, 11 m)
CERTAINTY_R = {"high": 0.9, "mid": 0.6, "low": 0.3}


def clue_prior(grid: np.ndarray, cell: float, origin_rc: tuple[int, int], facing: float, clue: dict,
               half_angle: float = np.radians(30), dist_rel: float = 0.2,
               near_far: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 8.0), (8.0, 60.0))) -> np.ndarray:
    """Geometric part of a clue (direction sector x distance ring) mixed with uniform by certainty.

    origin_rc / facing: anchor position (row, col) and heading [rad, atan2(-drow, dcol): 0 = +col, CCW positive].
    Semantic tokens are not used here (they need room labels); they only narrow the map once labels exist.
    """
    h, w = grid.shape
    rr, cc = np.mgrid[0:h, 0:w]
    dy, dx = -(rr - origin_rc[0]) * cell, (cc - origin_rc[1]) * cell
    rng = np.hypot(dx, dy)
    p = np.ones_like(rng)
    if clue.get("direction_deg") is not None:
        bearing = facing - np.radians(clue["direction_deg"])  # clockwise clue angle -> CCW map angle
        ang = np.angle(np.exp(1j * (np.arctan2(dy, dx) - bearing)))
        p *= np.exp(-0.5 * (np.maximum(np.abs(ang) - half_angle, 0) / np.radians(10)) ** 2)  # soft-edged sector
    if clue.get("distance_m") is not None:
        d = clue["distance_m"]
        p *= np.exp(-0.5 * ((rng - d) / max(dist_rel * d, cell)) ** 2)
    elif clue.get("distance_qual") in ("near", "far"):
        lo, hi = near_far[clue["distance_qual"] == "far"]
        p *= ((rng >= lo) & (rng <= hi)).astype(float) + 1e-3
    passable = grid != OCC
    p = p * passable
    uni = passable / passable.sum()
    r = CERTAINTY_R.get(clue.get("certainty") or "high", 0.9)
    return r * p / max(p.sum(), 1e-12) + (1 - r) * uni


def spread(b: np.ndarray, grid: np.ndarray, cell: float, dt: float, D: float) -> np.ndarray:
    """Move mass to passable neighbours for dt seconds. Mass is conserved; walls reflect it."""
    if D <= 0 or dt <= 0:
        return b
    k_total = D * dt / cell ** 2          # fraction per neighbour per unit, split into stable substeps (<= 0.2)
    n = int(np.ceil(k_total / 0.2))
    k = k_total / n
    passable = grid != OCC
    for _ in range(n):
        out = np.zeros_like(b)
        stay = b.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb_ok = np.roll(passable, -sh, axis=ax)          # is the neighbour in this direction passable?
            if ax == 0:
                nb_ok[(-1 if sh == 1 else 0), :] = False
            else:
                nb_ok[:, (-1 if sh == 1 else 0)] = False
            move = k * b * nb_ok * passable
            stay -= move
            out += np.roll(move, sh, axis=ax)
        b = stay + out
    return b


def observe(b: np.ndarray, seen: np.ndarray, p_detect: float = 0.8) -> tuple[np.ndarray, float]:
    """Camera looked at `seen` cells and found no one. -> (renormalised belief, probability the miss happened)."""
    miss = b.sum() - p_detect * b[seen].sum()
    b = b.copy()
    b[seen] *= 1 - p_detect
    return b / max(b.sum(), 1e-12), miss


def frontiers(grid: np.ndarray) -> np.ndarray:
    """FREE cells with an UNK 4-neighbour."""
    unk = grid == UNK
    nb = np.zeros_like(unk)
    nb[1:] |= unk[:-1]
    nb[:-1] |= unk[1:]
    nb[:, 1:] |= unk[:, :-1]
    nb[:, :-1] |= unk[:, 1:]
    return (grid == FREE) & nb


def frontier_mass(b: np.ndarray, grid: np.ndarray, fr: np.ndarray) -> dict[tuple[int, int], float]:
    """Belief mass in UNK cells, each given to the frontier it is reached from first through UNK space (BFS).

    Geodesic, not Euclidean: mass behind a known wall goes to the frontier that actually leads there."""
    owner = -np.ones(grid.shape, int)
    seeds = list(zip(*np.nonzero(fr)))
    q = deque()
    for i, (r, c) in enumerate(seeds):
        owner[r, c] = i
        q.append((r, c))
    h, w = grid.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and owner[rr, cc] < 0 and grid[rr, cc] == UNK:
                owner[rr, cc] = owner[r, c]
                q.append((rr, cc))
    m = np.bincount(owner[(owner >= 0) & (grid == UNK)], weights=b[(owner >= 0) & (grid == UNK)], minlength=len(seeds))
    return {s: float(m[i]) for i, s in enumerate(seeds)}
