"""The 2D search planner on real inputs (no simulator truth): SLAM scans + pose + text clues -> next goal.

Same decision code as the simulator (clue/search2d.py Search: Dijkstra travel time, frontier blocks with geodesic
ownership of unsearched space, rate score E / (C + t0)^kappa, Bayesian per-reporter clue trust). What changes:
  - the map is built from registered LiDAR points (LiveGrid), not read from a ground-truth grid
  - "searched" comes from the camera footprint on that map (ray cast at the robot pose)
  - clues arrive at any time as parsed JSON; the anchor of a direction / distance is the robot pose when the text
    came in (robot frame: x forward, y left)
  - a victim is reported (by a person at the console now, by the head-camera detector later) at a map position

Grid: CELL = 0.2 m (the value the selector was tuned with), fixed square around the start pose.
World (map frame, metres) -> cell: col = (x - x0) / CELL, row = (y1 - y) / CELL; heading = map yaw (0 = +x, CCW).

Semantic room clues ("왼쪽 세 번째 방") need room / door detection on the live map: not done yet, they are logged and
kept but give no probability map until then.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage

from . import belief as B
from .search2d import CELL, FREE, OCC, UNK, Search

CAM_R, CAM_FOV = 4.0, math.radians(90)     # head camera range / field of view used for "searched"
P_DETECT_MODEL = 0.8


class LiveGrid:
    """2D grid (CELL) from registered point clouds.

    floor:    10th percentile of z among points 0.15-1.2 m below the body within 3 m of the robot (running mean)
    hits:     points in [floor + obs_min, floor + obs_max] (obs_max ~ standing robot height)
    misses:   cells crossed by the 2D ray from the sensor to each such point (something was seen through them)
    occupied: hits >= min_hits and hits >= occ_ratio * misses  (a person walking by is crossed later and fades;
              walls are hit every scan)
    free:     crossed by rays (misses >= 2) or floor seen there, and not occupied; the robot footprint is always free
    """

    def __init__(self, size_m: float = 120.0, center_xy=(0.0, 0.0), obs_min=0.12, obs_max=0.5, floor_tol=0.06,
                 min_hits=3, occ_ratio=0.25, foot_r=0.35, max_rays=3000, ray_range=10.0):
        self.n = int(round(size_m / CELL))
        self.x0 = center_xy[0] - size_m / 2
        self.y1 = center_xy[1] + size_m / 2
        self.hits = np.zeros((self.n, self.n), np.int32)
        self.miss = np.zeros((self.n, self.n), np.int32)
        self.floor = np.zeros((self.n, self.n), np.int32)
        self.foot = np.zeros((self.n, self.n), bool)
        self.obs_min, self.obs_max, self.floor_tol, self.min_hits = obs_min, obs_max, floor_tol, min_hits
        self.occ_ratio, self.foot_r, self.max_rays, self.ray_range = occ_ratio, foot_r, max_rays, ray_range
        self.floor_z = None
        self.rng = np.random.default_rng(0)

    def cell(self, x, y):
        c = np.floor((np.asarray(x) - self.x0) / CELL).astype(int)
        r = np.floor((self.y1 - np.asarray(y)) / CELL).astype(int)
        return r, c

    def inside(self, x, y) -> bool:
        r, c = self.cell(x, y)
        return bool(0 <= r < self.n and 0 <= c < self.n)

    def world(self, r, c):
        return self.x0 + (np.asarray(c) + 0.5) * CELL, self.y1 - (np.asarray(r) + 0.5) * CELL

    def _floor(self, pts, pose):
        """Floor height under the robot for THIS scan (FAST-LIO z drifts by tens of cm over a run, so one global
        floor height does not work): 10th percentile of nearby points below the body; when too few are seen, the
        body height minus the last measured body-to-floor offset."""
        d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
        dz = pose[2] - pts[:, 2]
        cand = (d < 3.0) & (dz > 0.15) & (dz < 1.2)
        if cand.sum() > 30:
            est = float(np.percentile(pts[cand, 2], 10))
            off = pose[2] - est
            self.offset = off if getattr(self, "offset", None) is None else 0.8 * self.offset + 0.2 * off
            self.floor_z = est
        elif getattr(self, "offset", None) is not None:
            self.floor_z = pose[2] - self.offset
        return self.floor_z

    def add_scan(self, pts: np.ndarray, pose_xyz):
        """pts: (N, 3) points in the map frame; pose_xyz: sensor / body position in the map frame."""
        fz = self._floor(pts, pose_xyz)
        if fz is None:                              # no floor seen yet: only the footprint
            self.mark_footprint(pose_xyz[0], pose_xyz[1])
            return
        rel = pts[:, 2] - fz
        r, c = self.cell(pts[:, 0], pts[:, 1])
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        obs = ok & (rel >= self.obs_min) & (rel <= self.obs_max)
        flo = ok & (np.abs(rel) <= self.floor_tol)
        np.add.at(self.hits, (r[obs], c[obs]), 1)
        np.add.at(self.floor, (r[flo], c[flo]), 1)
        # ray clearing toward obstacle-band points (and points above it: open space below them is visible too)
        ray = ok & (rel >= self.obs_min)
        ray &= np.hypot(pts[:, 0] - pose_xyz[0], pts[:, 1] - pose_xyz[1]) < self.ray_range
        idx = np.flatnonzero(ray)
        if len(idx) > self.max_rays:
            idx = self.rng.choice(idx, self.max_rays, replace=False)
        if len(idx):
            r0, c0 = self.cell(pose_xyz[0], pose_xyz[1])
            r1, c1 = r[idx], c[idx]
            n = np.maximum(np.abs(r1 - r0), np.abs(c1 - c0))
            steps = int(n.max()) if len(n) else 0
            if steps > 1:
                t = np.arange(1, steps)[None, :] / np.maximum(n, 1)[:, None]          # exclude start and end
                keep = t < 1.0 - 0.5 / np.maximum(n, 1)[:, None]
                rr = np.rint(r0 + (r1 - r0)[:, None] * t).astype(int)[keep]
                cc = np.rint(c0 + (c1 - c0)[:, None] * t).astype(int)[keep]
                inb = (rr >= 0) & (rr < self.n) & (cc >= 0) & (cc < self.n)
                np.add.at(self.miss, (rr[inb], cc[inb]), 1)
        self.mark_footprint(pose_xyz[0], pose_xyz[1])

    def mark_footprint(self, x, y):
        r, c = self.cell(x, y)
        k = int(math.ceil(self.foot_r / CELL))
        rr, cc = np.mgrid[-k:k + 1, -k:k + 1]
        m = np.hypot(rr, cc) * CELL <= self.foot_r
        R, C = (r + rr[m]).clip(0, self.n - 1), (c + cc[m]).clip(0, self.n - 1)
        self.foot[R, C] = True

    def known(self) -> np.ndarray:
        """UNK / FREE / OCC grid."""
        k = np.full((self.n, self.n), UNK, np.int8)
        occ = (self.hits >= self.min_hits) & (self.hits >= self.occ_ratio * self.miss) & ~self.foot
        k[(self.floor > 0) | (self.miss >= 2) | self.foot] = FREE
        k[occ] = OCC
        return k


def camera_footprint(known: np.ndarray, rc, heading: float, rng=CAM_R, fov=CAM_FOV, n_rays=91) -> np.ndarray:
    """Cells the camera sees from cell rc facing heading: rays stop at the first occupied or unknown cell."""
    seen = np.zeros(known.shape, bool)
    H, W = known.shape
    steps = np.arange(1, int(rng / CELL) * 2 + 1) * 0.5
    for a in heading + np.linspace(-fov / 2, fov / 2, n_rays):
        ys = np.rint(rc[0] - np.sin(a) * steps).astype(int)
        xs = np.rint(rc[1] + np.cos(a) * steps).astype(int)
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ys, xs = ys[inb], xs[inb]
        stop = np.flatnonzero(known[ys, xs] != FREE)
        n = stop[0] if len(stop) else len(ys)
        seen[ys[:n], xs[:n]] = True
    seen[rc] = True
    return seen


class LiveSearch(Search):
    """Search's planner fed by the real robot. Call order per cycle: set_map, set_pose, look, (add_clue,
    victim_found), next_goal."""

    def __init__(self, shape, n_unclued: int = 1, beta: float = 0.75, trust_update: str = "reporter", rho=0.7,
                 w_lo=0.1, t0=10.0, kappa=1.0, inflate: int = 1):
        # attributes the inherited planning / trust code uses (no Search.__init__: there is no simulator Building)
        self.a, self.beta, self.c, self.score, self.t0, self.kappa = 1.0, beta, 1.0, "rate", t0, kappa
        self.trust, self.trust_update, self.rho, self.w_lo = None, trust_update, rho, w_lo
        self.keyword, self.vlm, self.D, self.witness = False, None, 0.0, False
        self.known = np.full(shape, UNK, np.int8)
        self.plan_map = self.known
        self.searched = np.zeros(shape, bool)
        self.nodet = np.ones(shape)
        self.pos, self.head, self.t = (shape[0] // 2, shape[1] // 2), 0.0, 0.0
        self.clues, self.active, self.fixed, self.post, self.zpost, self.L_found = [], {}, {}, {}, {}, {}
        self.found = [None] * n_unclued        # one slot per person we expect: unclued + one per clued person
        self.inflate = inflate
        self.t_start = time.time()
        self.log = []

    # -- inputs -------------------------------------------------------------------------------------------------
    def set_map(self, known: np.ndarray):
        self.known = known
        occ = known == OCC
        if self.inflate:
            occ = ndimage.binary_dilation(occ, iterations=self.inflate)
        self.plan_map = np.where(occ, OCC, known).astype(np.int8)

    def set_pose(self, rc, heading: float):
        self.pos = (int(rc[0]), int(rc[1]))
        self.head = float(heading)
        self.t = time.time() - self.t_start

    def look(self, spin: bool = False):
        """Camera footprint at the current pose -> searched, and the no-detection likelihood for the clue maps.
        spin: the robot turned in place (360 deg look, done on arrival at each goal)."""
        seen = camera_footprint(self.known, self.pos, self.head, fov=2 * math.pi if spin else CAM_FOV,
                                n_rays=360 if spin else 91)
        self.searched |= seen
        self.nodet[seen] *= 1 - P_DETECT_MODEL
        return seen

    def add_clue(self, parsed: dict, text: str = "", by: str = "reporter0", anchor=None):
        """Parsed JSON (clue/schema.py) -> clue anchored at the robot pose now, or at anchor = (cell, heading), the
        pose when the sentence was heard (parsing takes a second or two while the robot moves). -> status string."""
        pos, head = anchor if anchor is not None else (self.pos, self.head)
        if not parsed or parsed.get("utterance") != "clue":
            self.log.append((self.t, "not a clue", text))
            return f"not a clue ({(parsed or {}).get('utterance')})"
        n = int(parsed.get("count") or 1)
        c = {k: parsed.get(k) for k in ("direction_deg", "distance_m", "distance_qual", "certainty", "semantic")}
        c.update({"utterance": "clue", "count": n, "by": by, "text": text, "anchor_rc": pos, "anchor_head": head})
        k = len(self.clues)
        self.clues.append(c)
        self.found += [None] * n
        if c["direction_deg"] is None and c["distance_m"] is None and c["distance_qual"] is None:
            self.log.append((self.t, "kept, no map yet (semantic only)", text))
            return f"clue {k}: no direction / distance (semantic {c['semantic']}), kept for later"
        kw = {"half_angle": parsed["_half_angle"], "dist_rel": parsed["_dist_rel"]} if "_half_angle" in parsed else {}
        self.fixed[k] = B.clue_prior(np.zeros(self.known.shape, np.int8), CELL, pos, head, c, **kw)
        self.active[k] = True
        self.log.append((self.t, f"clue {k}", text))
        return f"clue {k}: direction {c['direction_deg']} distance {c['distance_m'] or c['distance_qual']} x{n}"

    def prior(self, k):
        return self.fixed[k]

    def victim_found(self, rc) -> str:
        """A person was confirmed at cell rc: fill a slot, credit the clue whose map explains it best."""
        rc = (int(rc[0]), int(rc[1]))
        best, best_lr = None, 1.0
        for k, on in self.active.items():
            if on:
                lr = self._density_ratio(k, rc)
                if lr > best_lr:
                    best, best_lr = k, lr
        slot = next((i for i, f in enumerate(self.found) if f is None), None)
        if slot is None:
            self.found.append(self.t)
        else:
            self.found[slot] = self.t
        if best is not None:
            self.L_found[best] = best_lr
            self.active[best] = False
        self.log.append((self.t, "victim", f"at {rc}, credited to clue {best} (LR {best_lr:.1f})"))
        return f"victim at {rc}: credited to clue {best}" if best is not None else f"victim at {rc}: no clue"

    # -- planning -------------------------------------------------------------------------------------------------
    def plan(self):
        k, self.known = self.known, self.plan_map      # plan on the inflated map, keep the raw one for the rest
        try:
            return Search.plan(self)
        finally:
            self.known = k

    def candidates(self, d):
        k, self.known = self.known, self.plan_map
        try:
            return Search.candidates(self, d)
        finally:
            self.known = k

    def next_goal(self):
        """-> (goal cell or None, travel-time map). None when nothing reachable is left unsearched."""
        if self.plan_map[self.pos] == OCC:           # inflated obstacle under the robot: plan from the raw map
            self.plan_map = self.plan_map.copy()
            self.plan_map[self.pos] = FREE
        # plan only inside the mapped area (+ margin): the full 60 m square is mostly unknown and 10x slower on the NX
        ys, xs = np.nonzero(self.known != UNK)
        m = 10
        y0, y1 = max(min(ys.min(), self.pos[0]) - m, 0), min(max(ys.max(), self.pos[0]) + m + 1, self.known.shape[0])
        x0, x1 = max(min(xs.min(), self.pos[1]) - m, 0), min(max(xs.max(), self.pos[1]) + m + 1, self.known.shape[1])
        win = (slice(y0, y1), slice(x0, x1))
        full = (self.known, self.plan_map, self.searched, self.nodet, self.fixed, self.pos)
        self.known, self.plan_map, self.searched, self.nodet = (a[win] for a in full[:4])
        self.fixed = {k: v[win] for k, v in full[4].items()}
        self.pos = (self.pos[0] - y0, self.pos[1] - x0)
        try:
            d, _ = self.plan()
            self._wkey = None
            g = self.choose(d)
        finally:
            self.known, self.plan_map, self.searched, self.nodet, self.fixed, self.pos = full
        dd = np.full(self.known.shape, np.inf)
        dd[win] = d
        return (None if g is None else (int(g[0]) + y0, int(g[1]) + x0)), dd

    def heat(self) -> np.ndarray:
        """Clue probability x trust (for display)."""
        h = np.zeros(self.known.shape)
        for w, m in (self.beliefs() if self.active else []):
            h += w * m
        return h
