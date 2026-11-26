"""Procedural navigation worlds for the HiPAN high-level policy (numpy only; no simulator needed).

A world is a grid of square tiles (rooms). Each tile holds boxes:
  * outer walls, 0-2 partition walls with doorways; some doorways have a low (optionally tilted) lintel above them
  * tables (thin legs + top), floating beams and low slabs (collapsed debris: crouch or roll to pass under)
  * floor clutter (must be avoided)
Every box is axis-aligned in its own frame, rotated by yaw and then by a pitch about its local y axis (tilted beams).

Per tile we also produce the privileged 2.5D map the teacher sees (HiPAN: 2.5D elevation + 3D occupancy; we keep
the same content as two exact 2D channels at 0.1 m):
  * ``elev``  top of the obstacle standing on the floor (0 = free floor)
  * ``clear`` free height above the floor before the first overhanging obstacle (0 where something stands on the
    floor, capped at ZMAX)
and start/goal tasks with a reference path (Dijkstra on a clearance-aware cost grid, used for PGCL sub-goals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TILE = 12.0          # tile edge [m]
RES = 0.1            # map resolution [m]
ZRES = 0.01          # vertical sampling for rasterisation [m]
ZMAX = 1.2           # map height cap [m]
WALL_H = 1.0
ROBOT_TOP = 0.163    # highest point (XT16) above the base origin, assets/go2_xt16/geometry.json
H_MIN, H_STAND = 0.13, 0.32
MIN_CLEAR = H_MIN + ROBOT_TOP + 0.02    # 0.31 m: passable only fully crouched
STAND_CLEAR = H_STAND + ROBOT_TOP + 0.03  # 0.51 m: passable standing
FOOT_R = 0.3         # footprint radius used for planning [m]


@dataclass
class Box:
    c: tuple[float, float, float]
    s: tuple[float, float, float]
    yaw: float = 0.0
    pitch: float = 0.0

    def rot(self) -> np.ndarray:
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        return rz @ ry

    def corners(self) -> np.ndarray:
        h = np.array(self.s) / 2
        sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        return (sg * h) @ self.rot().T + np.array(self.c)


@dataclass
class Tile:
    boxes: list[Box]
    elev: np.ndarray = None
    clear: np.ndarray = None
    starts: np.ndarray = None     # [T, 2] tile frame
    goals: np.ndarray = None      # [T, 2]
    paths: list = field(default_factory=list)  # each [P, 2], tile frame, 0.1 m spacing


def _floating(c_xy, size_xy, bottom, thick, yaw=0.0, pitch=0.0) -> Box:
    """A box whose lowest corner sits at ``bottom``."""
    b = Box((c_xy[0], c_xy[1], 0.0), (size_xy[0], size_xy[1], thick), yaw, pitch)
    z0 = b.corners()[:, 2].min()
    return Box((c_xy[0], c_xy[1], bottom - z0), b.s, yaw, pitch)


def make_tile(rng: np.random.Generator, difficulty: float) -> list[Box]:
    d = float(np.clip(difficulty, 0.0, 1.0))
    L = TILE
    t = 0.2
    bx = [Box((L / 2, t / 2, WALL_H / 2), (L, t, WALL_H)), Box((L / 2, L - t / 2, WALL_H / 2), (L, t, WALL_H)),
          Box((t / 2, L / 2, WALL_H / 2), (t, L, WALL_H)), Box((L - t / 2, L / 2, WALL_H / 2), (t, L, WALL_H))]

    def low_bottom():  # bottom height of an overhanging obstacle: lower with difficulty
        return rng.uniform(MIN_CLEAR + 0.01, STAND_CLEAR + 0.15 - 0.15 * d)

    # partition walls with doorways (some with a low or tilted lintel)
    for axis in rng.permutation(2)[: rng.integers(0, 3)]:
        pos = rng.uniform(3.5, L - 3.5)
        n_door = rng.integers(1, 3)
        doors = sorted(rng.uniform(1.2, L - 1.2, n_door))
        widths = rng.uniform(0.9, 1.6, n_door)
        segs, x0 = [], t
        for dc, w in zip(doors, widths):
            a, b = max(x0, dc - w / 2), dc + w / 2
            if a - x0 > 0.05:
                segs.append((x0, a))
            if rng.random() < 0.3 + 0.5 * d:  # lintel over the doorway
                bot = low_bottom()
                tilt = rng.uniform(-0.25, 0.25) * d if rng.random() < 0.5 else 0.0
                span = b - a + 0.2
                cx = (a + b) / 2
                c = (cx, pos) if axis == 0 else (pos, cx)
                yaw = 0.0 if axis == 0 else np.pi / 2
                bx.append(_floating(c, (span, t), bot, max(0.1, WALL_H - bot), yaw, tilt))
            x0 = b
        if L - t - x0 > 0.05:
            segs.append((x0, L - t))
        for a, b in segs:
            cx, ln = (a + b) / 2, b - a
            bx.append(Box((cx, pos, WALL_H / 2), (ln, t, WALL_H)) if axis == 0 else
                      Box((pos, cx, WALL_H / 2), (t, ln, WALL_H)))

    def rand_xy(margin=0.8):
        return rng.uniform(margin, L - margin, 2)

    # tables: legs + top
    for _ in range(rng.integers(0, 3 + int(3 * d))):
        sx, sy = rng.uniform(0.8, 2.0), rng.uniform(0.6, 1.2)
        top = rng.uniform(MIN_CLEAR + 0.05, 0.8)
        c, yaw = rand_xy(1.2), rng.uniform(-np.pi, np.pi)
        tb = Box((c[0], c[1], top - 0.02), (sx, sy, 0.04), yaw)
        bx.append(tb)
        R = tb.rot()
        for i in (-1, 1):
            for j in (-1, 1):
                o = R @ np.array([i * (sx / 2 - 0.05), j * (sy / 2 - 0.05), 0.0])
                bx.append(Box((c[0] + o[0], c[1] + o[1], (top - 0.04) / 2), (0.05, 0.05, top - 0.04), yaw))
    # floating beams (some tilted) and low slabs
    for _ in range(rng.integers(0, 2 + int(3 * d))):
        ln, w, th = rng.uniform(1.5, 4.0), rng.uniform(0.2, 0.6), rng.uniform(0.15, 0.4)
        bx.append(_floating(rand_xy(), (ln, w), low_bottom(), th, rng.uniform(-np.pi, np.pi),
                            rng.uniform(-0.25, 0.25) * d if rng.random() < 0.5 else 0.0))
    for _ in range(rng.integers(0, 1 + int(2 * d))):
        bx.append(_floating(rand_xy(), rng.uniform(1.0, 2.5, 2), low_bottom(), 0.1, rng.uniform(-np.pi, np.pi)))
    # floor clutter
    for _ in range(rng.integers(2, 5 + int(6 * d))):
        s = rng.uniform(0.3, 1.0, 2)
        h = rng.uniform(0.3, 1.0)
        c = rand_xy()
        bx.append(Box((c[0], c[1], h / 2), (s[0], s[1], h), rng.uniform(-np.pi, np.pi)))
    return bx


def rasterize(boxes: list[Box], n: int = int(round(TILE / RES))) -> tuple[np.ndarray, np.ndarray]:
    """(elev, clear) [n, n] indexed [ix, iy] for cell centres ((i + 0.5) * RES)."""
    nz = int(round(ZMAX / ZRES))
    occ = np.zeros((n, n, nz), bool)
    z = (np.arange(nz) + 0.5) * ZRES
    for b in boxes:
        cr = b.corners()
        lo = np.clip(np.floor(cr[:, :2].min(0) / RES).astype(int), 0, n - 1)
        hi = np.clip(np.ceil(cr[:, :2].max(0) / RES).astype(int), 0, n - 1)
        ix, iy = np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)
        X, Y, Z = np.meshgrid((ix + 0.5) * RES, (iy + 0.5) * RES, z, indexing="ij")
        p = np.stack([X, Y, Z], -1) - np.array(b.c)
        q = p @ b.rot()  # into the box frame (R^T p)
        inside = (np.abs(q) <= np.array(b.s) / 2 + 1e-6).all(-1)
        occ[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] |= inside
    run = np.argmin(occ, axis=2)  # first free sample from the floor
    run = np.where(occ.all(2), nz, run)
    elev = np.where(occ[..., 0], run * ZRES, 0.0)
    above = occ & (np.arange(nz) >= run[..., None])
    first = np.where(above.any(2), np.argmax(above, axis=2) * ZRES, ZMAX)
    clear = np.where(occ[..., 0], 0.0, first)
    return elev.astype(np.float32), clear.astype(np.float32)


def _disk(r_cells: int) -> np.ndarray:
    k = np.arange(-r_cells, r_cells + 1)
    return (k[:, None] ** 2 + k[None] ** 2) <= r_cells**2


def plan_tasks(clear: np.ndarray, rng: np.random.Generator, n_tasks: int, dist_range=(2.5, 15.0)):
    """Start/goal pairs inside the largest passable component, with Dijkstra paths on a clearance-aware cost."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra

    n = clear.shape[0]
    eff = ndimage.minimum_filter(clear, footprint=_disk(int(round(FOOT_R / RES))), mode="constant", cval=0.0)
    ok = eff >= MIN_CLEAR
    dist_obs = ndimage.distance_transform_edt(clear >= MIN_CLEAR) * RES
    cost = 1.0 + 2.0 * (eff < STAND_CLEAR) + 2.0 * np.exp(-np.maximum(dist_obs - FOOT_R, 0.0) / 0.3)
    idx = -np.ones((n, n), int)
    idx[ok] = np.arange(ok.sum())
    rows, cols, w = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = ok & np.roll(np.roll(ok, -dx, 0), -dy, 1)
        if dx:
            a[-dx:, :] = False
        if dy == 1:
            a[:, -1:] = False
        if dy == -1:
            a[:, :1] = False
        i, j = np.nonzero(a)
        u, v = idx[i, j], idx[i + dx, j + dy]
        step = np.hypot(dx, dy) * RES
        ww = step * 0.5 * (cost[i, j] + cost[i + dx, j + dy])
        rows += [u, v]; cols += [v, u]; w += [ww, ww]
    m = int(ok.sum())
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m)).tocsr()
    _, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    cells = np.argwhere(ok)  # node -> (ix, iy)
    comp = lab == big
    standing = comp & (eff[ok] >= STAND_CLEAR) & (dist_obs[ok] >= FOOT_R + 0.1)
    cand = np.nonzero(standing)[0]
    if len(cand) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    starts, goals, paths = [], [], []
    src = rng.choice(cand, size=min(n_tasks, len(cand)), replace=len(cand) < n_tasks)
    D, P = dijkstra(g, indices=src, return_predecessors=True)
    for k, s in enumerate(src):
        reach = np.nonzero(comp & np.isfinite(D[k]) & (D[k] > dist_range[0]) & (D[k] < 2.0 * dist_range[1]))[0]
        for _ in range(20):
            if len(reach) == 0:
                break
            gnode = rng.choice(reach)
            path = [gnode]
            while path[-1] != s:
                path.append(P[k, path[-1]])
            xy = (cells[path[::-1]] + 0.5) * RES
            length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
            if dist_range[0] <= length <= dist_range[1]:
                starts.append(xy[0]); goals.append(xy[-1]); paths.append(xy)
                break
    return np.array(starts), np.array(goals), paths


def build_world(rows: int, cols: int, seed: int = 0, n_tasks: int = 64, difficulty=None) -> dict:
    """Tiles laid out on a grid; tile (r, c) covers x in [r, r+1) * TILE, y in [c, c+1) * TILE (world frame)."""
    rng = np.random.default_rng(seed)
    n = int(round(TILE / RES))
    elev = np.zeros((rows * n, cols * n), np.float32)
    clear = np.zeros((rows * n, cols * n), np.float32)
    boxes, tasks = [], []
    for r in range(rows):
        for c in range(cols):
            d = r / max(rows - 1, 1) if difficulty is None else difficulty
            for _ in range(10):  # regenerate until the tile has enough tasks
                bx = make_tile(rng, d)
                e, cl = rasterize(bx, n)
                s, g, p = plan_tasks(cl, rng, n_tasks)
                if len(s) >= n_tasks // 2:
                    break
            o = np.array([r * TILE, c * TILE])
            elev[r * n:(r + 1) * n, c * n:(c + 1) * n] = e
            clear[r * n:(r + 1) * n, c * n:(c + 1) * n] = cl
            boxes += [Box((b.c[0] + o[0], b.c[1] + o[1], b.c[2]), b.s, b.yaw, b.pitch) for b in bx]
            # pad to n_tasks by repeating
            k = np.arange(n_tasks) % max(len(s), 1)
            tasks.append(dict(tile=r * cols + c, start=s[k] + o, goal=g[k] + o, paths=[p[i] + o for i in k],
                              difficulty=d))
    return dict(rows=rows, cols=cols, elev=elev, clear=clear, boxes=boxes, tasks=tasks)


def world_mesh(world: dict):
    import trimesh

    L = TILE
    R, C = world["rows"], world["cols"]
    floor = trimesh.creation.box(extents=(R * L + 2, C * L + 2, 0.2))
    floor.apply_translation((R * L / 2, C * L / 2, -0.1))
    parts = [floor]
    for b in world["boxes"]:
        m = trimesh.creation.box(extents=b.s)
        T = np.eye(4)
        T[:3, :3] = b.rot()
        T[:3, 3] = b.c
        m.apply_transform(T)
        parts.append(m)
    return trimesh.util.concatenate(parts)


def resample_paths(paths: list[np.ndarray], step: float = 0.1, max_pts: int = 200):
    """Arc-length resampling into fixed arrays: points [T, max_pts, 2], lengths [T]."""
    T = len(paths)
    pts = np.zeros((T, max_pts, 2), np.float32)
    lens = np.zeros(T, np.float32)
    for i, p in enumerate(paths):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        lens[i] = s[-1]
        q = np.minimum(np.arange(max_pts) * step, s[-1])
        pts[i, :, 0] = np.interp(q, s, p[:, 0])
        pts[i, :, 1] = np.interp(q, s, p[:, 1])
    return pts, lens


def build_beam_test(bottoms=(0.35, 0.42, 0.50), beam_x=7.0, beam_w=0.3, beam_t=0.3, n_tasks=4) -> dict:
    """One tile per beam bottom height: outer walls + one beam across the whole tile at x = beam_x."""
    n = int(round(TILE / RES))
    L, t = TILE, 0.2
    cols = len(bottoms)
    elev = np.zeros((n, cols * n), np.float32)
    clear = np.zeros((n, cols * n), np.float32)
    boxes, tasks = [], []
    for c, b in enumerate(bottoms):
        o = np.array([0.0, c * TILE])
        bx = [Box((L / 2, t / 2, WALL_H / 2), (L, t, WALL_H)), Box((L / 2, L - t / 2, WALL_H / 2), (L, t, WALL_H)),
              Box((t / 2, L / 2, WALL_H / 2), (t, L, WALL_H)), Box((L - t / 2, L / 2, WALL_H / 2), (t, L, WALL_H)),
              Box((beam_x, L / 2, b + beam_t / 2), (beam_w, L - 2 * t, beam_t))]
        e, cl = rasterize(bx, n)
        elev[:, c * n:(c + 1) * n], clear[:, c * n:(c + 1) * n] = e, cl
        boxes += [Box((q.c[0] + o[0], q.c[1] + o[1], q.c[2]), q.s, q.yaw, q.pitch) for q in bx]
        p = np.stack([np.linspace(2.0, 10.0, 81), np.full(81, L / 2)], 1)
        tasks.append(dict(tile=c, start=np.repeat(p[:1], n_tasks, 0) + o, goal=np.repeat(p[-1:], n_tasks, 0) + o,
                          paths=[p + o] * n_tasks, difficulty=0.0))
    return dict(rows=1, cols=cols, elev=elev, clear=clear, boxes=boxes, tasks=tasks)
