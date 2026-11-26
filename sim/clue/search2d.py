"""Plan-doc stage 2: does a clue belief map find N victims faster than clue-free exploration? (2D, kinematic)

Building: lobby at the west end, a main corridor, rooms on both sides (door to the corridor, some doors between
rooms), furniture blocks that hide people. Point robot:
  * LiDAR 360 deg, 10 m, line of sight -> map (known FREE / OCC, rest UNK)
  * camera 90 deg forward, 4 m, line of sight -> "searched" cells and detections (p = 0.9 per look);
    on arrival at a goal it turns once on the spot (360 deg look, SPIN_S seconds)
Being mapped is not being searched: a room seen by the LiDAR through a door still has to be looked at by the camera.

Goal choice (both conditions share everything else):
  candidates = reachable known-free cells next to unsearched space (one per 1 m block)
  unsearched cells are split among candidates by geodesic BFS (walls block) ->
    G(f) unsearched area behind f, M(f) clue belief mass behind f, C(f) travel time
  score S = a * G/maxG + b * M/maxM - c * C/maxC          (plan doc form)
  A: b = 0 (coverage only)     B: clue JSON given directly, b > 0
A reporter in the lobby gives clues for some victims at t = 0 (direction to 45 deg, distance rounded to 5 m).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import belief as B

CELL = 0.2
WALK_V, SPIN_S, TALK_S, LIMIT_S = 0.6, 8.0, 20.0, 900.0
LIDAR_R, CAM_R, CAM_FOV, P_SEE = 10.0, 4.0, math.radians(90), 0.9
P_DETECT_MODEL = 0.8          # what the belief update assumes (a bit lower than the true 0.9)
FREE, OCC = B.FREE, B.OCC
UNK = B.UNK


@dataclass
class Building:
    grid: np.ndarray          # FREE / OCC ground truth
    start: tuple
    facing: float             # reporter heading [rad, map: 0 = +col, CCW positive with row up]
    victims: list
    clues: list               # dicts in schema form + "victim" index (oracle match on detection)
    rooms: list = field(default_factory=list)


def make_building(seed: int, n_victims: int = 6, n_clues: int = 3, quality: str = "spoken") -> Building:
    """quality "spoken": direction to 45 deg, distance x U(0.85, 1.15) rounded to 5 m (how people talk)
               "exact":  true bearing and distance, narrow distribution (upper bound for geometric clues)"""
    rng = np.random.default_rng([seed, 11])
    m = lambda v: int(round(v / CELL))  # noqa: E731
    n_side = int(rng.integers(4, 8))
    widths = [m(rng.uniform(4.0, 6.0)) for _ in range(2 * n_side)]
    depth_n, depth_s = m(rng.uniform(4.5, 6.0)), m(rng.uniform(4.5, 6.0))
    corr, lobby_w = m(2.0), m(rng.uniform(5.0, 7.0))
    Wn = sum(widths[:n_side]) + n_side + 1
    Ws = sum(widths[n_side:]) + n_side + 1
    Wc = max(Wn, Ws)
    H = depth_n + corr + depth_s + 4
    W = lobby_w + Wc + 2
    g = np.full((H, W), OCC, np.int8)
    y_c0 = depth_n + 2                       # corridor rows y_c0 .. y_c0+corr-1
    g[y_c0:y_c0 + corr, lobby_w + 1:W - 1] = FREE
    g[1:H - 1, 1:lobby_w + 1] = FREE         # lobby, full height
    rooms = []

    def row_of_rooms(ws, y0, y1, door_row):
        x = lobby_w + 2
        prev = None
        for w in ws:
            g[y0:y1, x:x + w] = FREE
            d = x + int(rng.integers(1, max(2, w - m(1.0))))
            g[door_row, d:d + m(1.0)] = FREE                     # door to the corridor
            if prev is not None and rng.random() < 0.3:          # door to the previous room
                yy = int(rng.integers(y0 + 1, y1 - m(1.0)))
                g[yy:yy + m(1.0), x - 1] = FREE
            for _ in range(int(rng.integers(0, 3))):              # furniture
                fh, fw = m(rng.uniform(0.6, 1.2)), m(rng.uniform(0.6, 1.6))
                fy, fx = int(rng.integers(y0 + 2, max(y0 + 3, y1 - fh - 2))), int(rng.integers(x + 2, max(x + 3, x + w - fw - 2)))
                g[fy:fy + fh, fx:fx + fw] = OCC
            rooms.append((y0, y1, x, x + w))
            prev = (y0, y1, x, x + w)
            x += w + 1

    row_of_rooms(widths[:n_side], 1, y_c0 - 1, y_c0 - 1)
    row_of_rooms(widths[n_side:], y_c0 + corr + 1, H - 1, y_c0 + corr)
    g[y_c0:y_c0 + corr, lobby_w] = FREE                          # lobby opens to the corridor
    start = (y_c0 + corr // 2, m(1.5))
    # only cells reachable from the start count
    lab, _ = ndimage.label(g == FREE)
    g[(lab != lab[start]) & (g == FREE)] = OCC
    free = np.argwhere(g == FREE)
    clear = ndimage.binary_erosion(g == FREE, iterations=1)
    victims = []
    while len(victims) < n_victims:
        if rng.random() < 0.8:
            y0, y1, x0, x1 = rooms[int(rng.integers(len(rooms)))]
            p = (int(rng.integers(y0 + 1, y1 - 1)), int(rng.integers(x0 + 1, x1 - 1)))
        else:
            p = tuple(free[int(rng.integers(len(free)))])
        if clear[p] and math.dist(p, start) * CELL > 8.0 and all(math.dist(p, q) > m(1.5) for q in victims):
            victims.append(p)
    facing = float(rng.choice([0.0, np.pi / 2, -np.pi / 2, np.pi]))  # reporter heading (E, N, S, W)
    clues = []
    for k in rng.permutation(n_victims)[:n_clues]:
        v = victims[int(k)]
        dy, dx = -(v[0] - start[0]) * CELL, (v[1] - start[1]) * CELL
        bearing = math.degrees(facing - math.atan2(dy, dx)) % 360   # clockwise from facing
        dist = math.hypot(dx, dy) * rng.uniform(0.85, 1.15)
        if quality == "exact":
            c = {"direction_deg": bearing, "distance_m": math.hypot(dx, dy), "_half_angle": math.radians(3), "_dist_rel": 0.05}
        else:
            c = {"direction_deg": int(round(bearing / 45) * 45) % 360, "distance_m": float(max(5, round(dist / 5) * 5))}
        clues.append({"utterance": "clue", "anchor": "reporter", "certainty": "high", "count": 1, "victim": int(k), **c})
    return Building(g, start, facing, victims, clues, rooms)


_ANG = np.linspace(-np.pi, np.pi, 360, endpoint=False)


class Search:
    def __init__(self, b: Building, a=1.0, beta=0.0, c=1.0, use_clues=None, seed=0, mobility="static",
                 score="rate", t0=10.0):
        """score "rate": expected victims behind f / (travel time + t0), expected victims =
             beta * sum of clue masses + (victims not covered by an active clue) * G(f) / sum G
           score "doc": a * G/maxG + beta * M/maxM - c * C/maxC (plan doc form)."""
        self.b, self.a, self.beta, self.c, self.score, self.t0 = b, a, beta, c, score, t0
        self.use_clues = beta > 0 if use_clues is None else use_clues
        self.rng = np.random.default_rng([seed, 5])
        H, W = b.grid.shape
        self.known = np.full((H, W), UNK, np.int8)
        self.searched = np.zeros((H, W), bool)
        self.pos, self.head, self.t, self.path = b.start, 0.0, 0.0, 0.0
        self.found = [None] * len(b.victims)
        self.D = B.D_MOBILITY[mobility]
        n = int(LIDAR_R / CELL)
        k = np.arange(1, 2 * n + 1) * 0.5
        self.dy, self.dx = np.sin(_ANG)[:, None] * k, np.cos(_ANG)[:, None] * k
        self.rk = k * CELL
        self.maps = {}
        if self.use_clues:
            self.t += TALK_S
            for i, c in enumerate(b.clues):
                kw = {"half_angle": c["_half_angle"], "dist_rel": c["_dist_rel"]} if "_half_angle" in c else {}
                self.maps[i] = B.clue_prior(np.zeros((H, W), np.int8), CELL, b.start, b.facing, c, **kw)

    # -- sensing ------------------------------------------------------------------------------------------
    def _rays(self):
        H, W = self.b.grid.shape
        y = np.rint(self.pos[0] - self.dy).astype(int).clip(0, H - 1)   # row up = +dy
        x = np.rint(self.pos[1] + self.dx).astype(int).clip(0, W - 1)
        blk = self.b.grid[y, x] == OCC
        before = np.cumsum(blk, axis=1) - blk
        return y, x, before == 0

    def sense(self, spin=False):
        y, x, s = self._rays()
        self.known[y[s], x[s]] = self.b.grid[y[s], x[s]]
        self.known[self.pos] = FREE
        ang = np.angle(np.exp(1j * (_ANG - self.head)))
        cam = s & (self.rk[None, :] <= CAM_R) & ((np.abs(ang) <= CAM_FOV / 2) | spin)[:, None]
        seen = np.zeros(self.known.shape, bool)
        seen[y[cam], x[cam]] = True
        seen &= self.b.grid == FREE
        seen[self.pos] = True
        self.searched |= seen
        for i, v in enumerate(self.b.victims):
            if self.found[i] is None and seen[v] and self.rng.random() < P_SEE:
                self.found[i] = self.t
                for k, c in list(self.maps.items()):
                    if c is not None and self.b.clues[k]["victim"] == i:    # oracle match (appearance in reality)
                        self.maps[k] = None
        for k, m in self.maps.items():
            if m is not None:
                m[self.known == OCC] = 0
                m[seen] *= 1 - P_DETECT_MODEL
                s_ = m.sum()
                self.maps[k] = m / s_ if s_ > 1e-12 else None

    # -- planning ------------------------------------------------------------------------------------------
    def plan(self):
        cost = np.where(self.known == FREE, CELL / WALK_V, np.inf)
        H, W = cost.shape
        idx = np.arange(H * W).reshape(H, W)
        r, c, v = [], [], []
        for dy, dx, s in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            a = (slice(0, H - dy), slice(max(0, -dx), W - max(0, dx)))
            b = (slice(dy, H), slice(max(0, dx), W - max(0, -dx)))
            ok = np.isfinite(cost[a]) & np.isfinite(cost[b])
            if dy and dx:
                ok &= np.isfinite(cost[a[0], b[1]]) & np.isfinite(cost[b[0], a[1]])
            r.append(idx[a][ok]); c.append(idx[b][ok]); v.append((s * 0.5 * (cost[a] + cost[b]))[ok])  # noqa: E702
        r, c, v = map(np.concatenate, (r, c, v))
        G = coo_matrix((v, (r, c)), shape=(H * W, H * W)).tocsr()
        d, p = dijkstra(G, directed=False, indices=idx[self.pos], return_predecessors=True)
        return d.reshape(H, W), p

    def candidates(self, d):
        """-> targets [(y, x)], partition owner map, unsearched mask."""
        open_ = (self.known != OCC) & ~self.searched           # unsearched, possibly passable
        reach = np.isfinite(d) & (self.known == FREE)
        edge = reach & ndimage.binary_dilation(open_, structure=np.ones((3, 3)))
        if not edge.any():
            return [], None, open_
        H, W = d.shape
        blk = 5                                                 # one candidate per 1 m block: the nearest cell
        ys, xs = np.nonzero(edge)
        key = (ys // blk) * (W // blk + 1) + xs // blk
        order = np.lexsort((d[ys, xs], key))
        first = np.r_[True, key[order][1:] != key[order][:-1]]
        tg = list(zip(ys[order][first], xs[order][first]))
        owner = -np.ones((H, W), int)
        q = deque()
        for i, p in enumerate(tg):
            owner[p] = i
            q.append(p)
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and owner[yy, xx] < 0 and open_[yy, xx]:
                    owner[yy, xx] = owner[y, x]
                    q.append((yy, xx))
        return tg, owner, open_

    def choose(self, d):
        tg, owner, open_ = self.candidates(d)
        if not tg:
            return None
        n = len(tg)
        sel = (owner >= 0) & open_
        G = np.bincount(owner[sel], minlength=n).astype(float)
        M = np.zeros(n)
        for m in self.maps.values():
            if m is not None:
                M += np.bincount(owner[sel], weights=m[sel], minlength=n)
        C = np.array([d[p] for p in tg])
        if self.score == "rate":
            active = sum(m is not None for m in self.maps.values())
            rest = max(sum(f is None for f in self.found) - active, 0) if self.beta > 0 else sum(f is None for f in self.found)
            E = self.a * rest * G / max(G.sum(), 1e-9) + self.beta * M
            S = E / (C + self.t0)
        else:
            S = self.a * G / max(G.max(), 1e-9) - self.c * C / max(C.max(), 1e-9)
            if self.beta > 0 and M.max() > 1e-9:
                S = S + self.beta * M / M.max()
        return tg[int(np.argmax(S))]

    # -- execution -----------------------------------------------------------------------------------------
    def go(self, target, d, pred):
        H, W = self.known.shape
        k, cells = target[0] * W + target[1], []
        while k >= 0:
            cells.append((k // W, k % W))
            k = pred[k]
        cells = cells[::-1]
        for i in range(1, len(cells)):
            (y0, x0), (y1, x1) = cells[i - 1], cells[i]
            self.t += float(d[cells[i]] - d[cells[i - 1]])
            self.path += math.dist(cells[i], cells[i - 1]) * CELL
            self.head = math.atan2(-(y1 - y0), x1 - x0)
            self.pos = cells[i]
            if i % 2 == 0:
                n0 = sum(f is not None for f in self.found)
                self.sense()
                if self.known[cells[-1]] == OCC or sum(f is not None for f in self.found) > n0 or i >= 25:
                    return                                           # replan: blocked, found someone, or 5 m walked
        self.t += SPIN_S
        self.sense(spin=True)

    def run(self):
        self.sense(spin=True)
        for _ in range(600):
            if all(f is not None for f in self.found) or self.t > LIMIT_S:
                break
            d, pred = self.plan()
            tgt = self.choose(d)
            if tgt is None:
                break
            if tgt == self.pos:
                self.t += SPIN_S
                self.sense(spin=True)
                self.searched[self.pos] = True
                continue
            self.go(tgt, d, pred)
        f = [x for x in self.found if x is not None and x <= LIMIT_S]
        n = len(self.b.victims)
        ts = np.linspace(0, LIMIT_S, 181)
        curve = np.array([sum(x <= t for x in f) for t in ts]) / n
        cov = self.searched.sum() / (self.b.grid == FREE).sum()
        return dict(n_found=len(f), all_found=len(f) == n, t_all=max(f) if len(f) == n else LIMIT_S,
                    t_first=min(f) if f else LIMIT_S, auc=float(np.trapezoid(curve, ts) / LIMIT_S),
                    path=self.path, coverage=float(cov), t_end=min(self.t, LIMIT_S))


def run_job(job):
    seed, variants, kw = job
    b = make_building(seed, **kw)
    return dict(seed=seed, n_rooms=len(b.rooms), area=float((b.grid == FREE).sum() * CELL ** 2),
                **{name: Search(b, seed=seed, **v).run() for name, v in variants.items()})
