"""Gate G2: does linking clues to deferred low gaps help, if clue reading were perfect?

2D grid world (rooms, doors, low gaps), point robot, 360 deg line of sight that a standing robot cannot get through a
low gap (XT16 optics at ~0.47 m). One witness A gives a direction / distance clue; the run ends when B is seen.

Methods (all share map, sensing, frontier extraction, timing; clue methods share the same relevance function):
  explore     nearest option, low gaps are walls
  posture     nearest option, low gaps allowed
  clue        options weighted by the clue, low gaps are walls
  separate    clue weights frontiers, gaps are scored without the clue (two independent modules)
  linked_nop  clue weights frontiers and gaps, pass probability ignored
  linked      relevance x P_pass / time for frontiers and gaps
  truth       as linked, relevance from B's true position (upper bound on clue reading; not a deployable method)

"Perfect reading" = probability mass of the still-unknown space consistent with the clue, never B's true position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

CELL = 0.2
WALK_V, CROUCH_V, SWITCH_S, FAIL_S, TALK_S = 0.6, 0.2, 6.0, 20.0, 20.0
LIDAR_R, SEE_R, LIMIT_S = 10.0, 5.0, 600.0
P_PASS = {0.40: 0.5, 0.45: 0.8, 0.50: 0.95}
FREE, WALL, GAP, UNK = 0, 1, 2, 3
METHODS = ("explore", "posture", "clue", "separate", "linked_nop", "linked", "truth")


@dataclass
class World:
    grid: np.ndarray
    gap_id: np.ndarray
    gaps: list                 # dict(clear, ok, sides=((y, x), (y, x)))
    start: tuple
    witness: tuple
    victim: tuple
    behind_gap: bool
    clue_dir: float
    clue_far: bool


def make_world(seed: int, rooms=(3, 4)) -> World:
    """Stratified: even seeds hide B behind gaps only, odd seeds in door-connected rooms."""
    rng = np.random.default_rng(seed)
    want_gap = seed % 2 == 0
    while True:
        R, C, h, w = *rooms, 24, 24  # rooms of 4.6 m
        g = np.zeros((R * h + 1, C * w + 1), np.int8)
        g[::h] = WALL
        g[:, ::w] = WALL
        gid = np.full(g.shape, -1, int)
        edges = [((r, c), (r, c + 1)) for r in range(R) for c in range(C - 1)] + \
                [((r, c), (r + 1, c)) for r in range(R - 1) for c in range(C)]
        rng.shuffle(edges)
        comp = {(r, c): (r, c) for r in range(R) for c in range(C)}

        def find(a):
            while comp[a] != a:
                a = comp[a]
            return a

        gaps, doors = [], []
        for a, b in (tuple(map(tuple, e)) for e in edges):
            tree = find(a) != find(b)
            if tree:
                comp[find(a)] = find(b)
            elif rng.random() > 0.3:
                continue
            low = rng.random() < 0.45
            if a[0] == b[0]:
                y, x = a[0] * h + int(rng.integers(5, h - 5)), b[1] * w
                cells, sides = [(y + k, x) for k in range(-1, 3)], ((y, x - 1), (y, x + 1))
            else:
                y, x = b[0] * h, a[1] * w + int(rng.integers(5, w - 5))
                cells, sides = [(y, x + k) for k in range(-1, 3)], ((y - 1, x), (y + 1, x))
            for cy, cx in cells:
                g[cy, cx] = GAP if low else FREE
                if low:
                    gid[cy, cx] = len(gaps)
            if low:
                cl = float(rng.choice(list(P_PASS)))
                gaps.append(dict(clear=cl, ok=bool(rng.random() < P_PASS[cl]), sides=sides))
            else:
                doors.append((a, b))
        # door-connected / solvable rooms
        def reach(allow):
            lab, _ = ndimage.label(allow)
            return lab
        walk = reach(g == FREE)
        ok = g == FREE
        for k, gp in enumerate(gaps):
            if gp["ok"]:
                ok = ok | (gid == k)
        solv = reach(ok)
        cells_rc = [(r, c) for r in range(R) for c in range(C)]
        mid = lambda q: (q[0] * h + h // 2, q[1] * w + w // 2)  # noqa: E731
        start = mid(cells_rc[int(rng.integers(len(cells_rc)))])
        wr = [q for q in cells_rc if walk[mid(q)] == walk[start] and mid(q) != start]
        vr = [q for q in cells_rc if solv[mid(q)] == solv[start] and mid(q) != start and
              (walk[mid(q)] != walk[start]) == want_gap]
        if not wr or not vr:
            continue
        A = wr[int(rng.integers(len(wr)))]
        vr = [q for q in vr if q != A]
        if not vr:
            continue
        B = vr[int(rng.integers(len(vr)))]
        jit = lambda q: (mid(q)[0] + int(rng.integers(-8, 9)), mid(q)[1] + int(rng.integers(-8, 9)))  # noqa: E731
        a, b = jit(A), jit(B)
        true = math.atan2(b[0] - a[0], b[1] - a[1])
        return World(g, gid, gaps, start, a, b, want_gap, true + rng.normal(0, math.radians(15)),
                     math.dist(a, b) * CELL > 6.0)


_ANG = np.linspace(-np.pi, np.pi, 360, endpoint=False)


class Run:
    def __init__(self, w: World, method: str, misread=None, p=0.0, memory="full", seed=0):
        """misread: None | "clue" (the whole clue is misunderstood, once per run, prob p: direction off by 90-180 deg and
        near/far flipped) | "pick" (each decision, prob p: relevances permuted among the options).
        memory of gaps seen before the clue: "full" | "neutral" (kept, but scored without the clue) | "forget" (dropped
        until seen again)."""
        assert method in METHODS and misread in (None, "clue", "pick") and memory in ("full", "neutral", "forget")
        self.w, self.m, self.misread, self.p, self.memory = w, method, misread, p, memory
        self.rng = np.random.default_rng([seed, 77])
        self.clue_dir, self.clue_far = w.clue_dir, w.clue_far
        u, off, sg = self.rng.random(), self.rng.uniform(np.pi / 2, np.pi), self.rng.choice([-1, 1])
        if misread == "clue" and u < p:
            self.clue_dir, self.clue_far = w.clue_dir + sg * off, not w.clue_far
        self.seen_gaps, self.old_gaps = set(), set()
        self.use_gaps = method not in ("explore", "clue")
        self.use_clue = method not in ("explore", "posture")
        self.known = np.full(w.grid.shape, UNK, np.int8)
        self.closed, self.opened = set(), set()
        self.pos, self.t, self.path_m = w.start, 0.0, 0.0
        self.clue = False
        self.found = False
        self.tries = self.fails = 0
        n = int(LIDAR_R / CELL)
        k = np.arange(1, 2 * n + 1) * 0.5
        self.dy, self.dx = np.sin(_ANG)[:, None] * k, np.cos(_ANG)[:, None] * k
        self.yy, self.xx = np.mgrid[:w.grid.shape[0], :w.grid.shape[1]]

    # -- sensing ------------------------------------------------------------------------------------------------
    def visible(self):
        H, W = self.w.grid.shape
        y = np.rint(self.pos[0] + self.dy).astype(int).clip(0, H - 1)
        x = np.rint(self.pos[1] + self.dx).astype(int).clip(0, W - 1)
        v = self.w.grid[y, x] != FREE
        before = np.cumsum(v, axis=1) - v  # blockers strictly before this sample
        s = before == 0
        vis = np.zeros((H, W), bool)
        vis[y[s], x[s]] = True
        vis[self.pos] = True
        return vis

    def sense(self):
        vis = self.visible()
        self.known[vis] = self.w.grid[vis]
        now = set(np.unique(self.w.gap_id[vis & (self.w.grid == GAP)]).tolist())
        self.seen_gaps |= now
        self.old_gaps -= now
        see = lambda p: vis[p] and math.dist(p, self.pos) * CELL <= SEE_R  # noqa: E731
        if see(self.w.victim):
            self.found = True
        elif self.use_clue and not self.clue and see(self.w.witness):
            self.clue = True
            self.t += TALK_S
            self.old_gaps = self.seen_gaps - now
            return True
        return False

    # -- planning -----------------------------------------------------------------------------------------------
    def plan(self):
        free = self.known == FREE
        cost = np.where(free, CELL / WALK_V, np.inf)
        for k in self.opened:
            cost[(self.w.gap_id == k) & (self.known == GAP)] = CELL / CROUCH_V + 2 * SWITCH_S
        H, W = cost.shape
        idx = np.arange(H * W).reshape(H, W)
        r, c, v = [], [], []
        for dy, dx, s in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            a = (slice(0, H - dy), slice(max(0, -dx), W - max(0, dx)))
            b = (slice(dy, H), slice(max(0, dx), W - max(0, -dx)))
            ok = np.isfinite(cost[a]) & np.isfinite(cost[b])
            if dy and dx:
                ok &= np.isfinite(cost[a[0], b[1]]) & np.isfinite(cost[b[0], a[1]])
            e = s * 0.5 * (cost[a] + cost[b])
            r.append(idx[a][ok]); c.append(idx[b][ok]); v.append(e[ok])  # noqa: E702
        r, c, v = map(np.concatenate, (r, c, v))
        G = coo_matrix((v, (r, c)), shape=(H * W, H * W)).tocsr()
        d, p = dijkstra(G, directed=False, indices=idx[self.pos], return_predecessors=True)
        return d.reshape(H, W), p

    def options(self, d):
        """-> list of (kind, id, target cell, time to finish)."""
        near = ndimage.binary_dilation(self.known == UNK)
        lab, n = ndimage.label((self.known == FREE) & near, structure=np.ones((3, 3)))
        out = []
        if n:
            size = np.bincount(lab.ravel(), minlength=n + 1)
            for i in range(1, n + 1):
                if size[i] < 3:
                    continue
                cells = np.argwhere(lab == i)
                q = d[cells[:, 0], cells[:, 1]]
                j = int(q.argmin())
                if np.isfinite(q[j]):
                    out.append(("frontier", i, tuple(cells[j]), float(q[j])))
        if self.use_gaps:
            for k, gp in enumerate(self.w.gaps):
                if k in self.closed or k in self.opened or (self.memory == "forget" and k in self.old_gaps) or not ((self.w.gap_id == k) & (self.known == GAP)).any():
                    continue
                s = min(gp["sides"], key=lambda q: d[q])
                if np.isfinite(d[s]):
                    out.append(("gap", k, s, float(d[s]) + 2 * SWITCH_S + CELL / CROUCH_V))
        return out

    def belief(self):
        """Clue-consistent probability over unknown cells."""
        y, x = (self.yy - self.w.witness[0]), (self.xx - self.w.witness[1])
        da = np.angle(np.exp(1j * (np.arctan2(y, x) - self.clue_dir)))
        dist = np.hypot(y, x) * CELL
        b = np.exp(-0.5 * (da / math.radians(35)) ** 2) * np.where((dist > 6.0) == self.clue_far, 1.0, 0.15)
        b = b * (self.known == UNK)
        return b / max(b.sum(), 1e-12)

    def relevance(self, opts):
        if self.m == "truth":
            b = np.zeros(self.known.shape)
            b[self.w.victim] = 1.0
            b = ndimage.gaussian_filter(b, 2.0 / CELL)
        else:
            b = self.belief()
        return np.array([b[np.hypot(self.yy - o[2][0], self.xx - o[2][1]) * CELL <= 4.0].sum() for o in opts])

    def choose(self, opts):
        t = np.array([o[3] for o in opts])
        gap = np.array([o[0] == "gap" for o in opts])
        R = np.ones(len(opts))
        if self.use_clue and self.clue:
            R = self.relevance(opts) + 0.02
            if self.misread == "pick" and self.rng.random() < self.p:
                R = self.rng.permutation(R)
            if self.memory == "neutral":
                old = np.array([o[0] == "gap" and o[1] in self.old_gaps for o in opts])
                R[old] = R.mean()
            if self.m == "separate" and gap.any():
                R[gap] = R.mean()
        if self.m in ("linked", "truth"):
            R = R * np.array([P_PASS[self.w.gaps[o[1]]["clear"]] if o[0] == "gap" else 1.0 for o in opts])
        return opts[int(np.argmax(R / (t + 5.0)))]

    # -- execution ----------------------------------------------------------------------------------------------
    def follow(self, target, d, pred):
        H, W = self.known.shape
        k, cells = target[0] * W + target[1], []
        while k >= 0:
            cells.append((k // W, k % W))
            k = pred[k]
        cells = cells[::-1]
        for i in range(1, len(cells)):
            self.t += float(d[cells[i]] - d[cells[i - 1]])
            self.path_m += math.dist(cells[i], cells[i - 1]) * CELL
            self.pos = cells[i]
            if i % 2 == 0 or i == len(cells) - 1:
                new = self.sense()
                if self.found or new or self.t > LIMIT_S:
                    return False
        return True

    def attempt(self, k, side):
        gp = self.w.gaps[k]
        self.tries += 1
        if not gp["ok"]:
            self.t += 2 * SWITCH_S + FAIL_S
            self.closed.add(k)
            self.fails += 1
            return
        self.t += 2 * SWITCH_S + 2 * CELL / CROUCH_V
        self.opened.add(k)
        self.pos = gp["sides"][1] if side == gp["sides"][0] else gp["sides"][0]
        self.sense()

    def run(self):
        self.sense()
        for _ in range(400):
            if self.found or self.t > LIMIT_S:
                break
            d, pred = self.plan()
            opts = self.options(d)
            if not opts:
                break
            kind, k, target, _ = self.choose(opts)
            if target != self.pos and not self.follow(target, d, pred):
                continue
            if kind == "gap":
                self.attempt(k, target)
            else:
                before = (self.known == UNK).sum()
                self.sense()
                if target == self.pos and (self.known == UNK).sum() == before:
                    # nothing left to see from here: retire the unknown cells this frontier hangs on
                    y, x = target
                    win = self.known[y - 1:y + 2, x - 1:x + 2]
                    win[win == UNK] = WALL
        ok = self.found and self.t <= LIMIT_S
        return dict(found=ok, t=min(self.t, LIMIT_S) if ok else LIMIT_S, clue=self.clue, tries=self.tries,
                    fails=self.fails, path=self.path_m)


def run_seed(job):
    seed, rooms = job
    w = make_world(seed, rooms)
    return dict(seed=seed, behind_gap=w.behind_gap, n_gaps=len(w.gaps), **{m: Run(w, m).run() for m in METHODS})


def run_variants(job):
    """One world, named variants -> {name: result}. variants: name -> (method, kwargs)."""
    seed, rooms, variants = job
    w = make_world(seed, rooms)
    return dict(seed=seed, behind_gap=w.behind_gap,
                **{n: Run(w, m, seed=seed, **kw).run() for n, (m, kw) in variants.items()})
