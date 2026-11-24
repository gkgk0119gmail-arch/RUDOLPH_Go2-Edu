"""Posture-dependent traversability from a 3D voxel map (0 unknown, 1 free, 2 occupied; [nx, ny, nz], flat floor z=0).

The inspection map marks a column blocked by anything between 0.05 and 0.6 m. Here a column keeps the free height
under its lowest overhang, so a low gap is CROUCH instead of blocked. Cell classes are conservative: an occupied voxel
counts from its lower face, and unknown space is never assumed passable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

UNKNOWN, BLOCKED, CROUCH, WALK = 0, 1, 2, 3
_E8 = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Body:
    """Highest point of the robot incl. sensors. Defaults: URDF collision model, NOT yet measured on the robot."""

    stand_top: float = 0.50   # tools/plot_go2_xt16.py, standing
    crouch_top: float = 0.35  # example crouched pose; walking in it is unproven
    margin: float = 0.05
    half_width: float = 0.20  # collision half width 0.16 + clearance
    step: float = 0.10        # ground band the legs step over

    @property
    def stand_need(self):
        return self.stand_top + self.margin

    @property
    def crouch_need(self):
        return self.crouch_top + self.margin


def classify(grid: np.ndarray, z0: float, voxel: float, body: Body = Body()):
    """-> (class [nx, ny], clearance [nx, ny] in m: lower face of the first obstacle above the step band, inf if none)."""
    z_lo = z0 + np.arange(grid.shape[2]) * voxel
    eps = 1e-6
    above = z_lo >= body.step - eps
    if z_lo[-1] + voxel < body.stand_need - eps:
        raise ValueError("voxel map is lower than the standing robot")
    occ = (grid == 2) & above
    clear = np.where(occ.any(2), z_lo[occ.argmax(2)], np.inf)

    def seen_free(top):
        need = above & (z_lo < top - eps)
        return (grid[:, :, need] == 1).all(2)

    out = np.full(grid.shape[:2], UNKNOWN, np.int8)
    out[clear < body.crouch_need - eps] = BLOCKED
    out[seen_free(body.crouch_need) & (clear >= body.crouch_need - eps)] = CROUCH
    out[seen_free(body.stand_need) & (clear >= body.stand_need - eps)] = WALK
    # crouch-high space is free but what is above it is unknown: passable crouched, which is what CROUCH says
    return out, clear


def robot_classes(cls: np.ndarray, cell: float, body: Body = Body()) -> np.ndarray:
    """Class of the robot *centred* on each cell: the worst class under its footprint disc (map border = unknown)."""
    r = int(np.ceil(body.half_width / cell - 1e-6))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (yy * yy + xx * xx) * cell * cell <= body.half_width ** 2 + 1e-9

    def worst_is(mask):
        return ndimage.binary_dilation(mask, structure=disc)

    padded = np.pad(cls, r, constant_values=UNKNOWN)
    out = np.full(padded.shape, WALK, np.int8)
    out[worst_is(padded == CROUCH)] = CROUCH
    out[worst_is(padded == UNKNOWN)] = UNKNOWN
    out[worst_is(padded == BLOCKED)] = BLOCKED
    return out[r:-r, r:-r] if r else out


@dataclass
class Gap:
    id: int
    cells: np.ndarray      # [K, 2] robot-centre cells
    clearance: float       # lowest overhang over the gap [m]
    entries: list          # centroids (cell coords) of the WALK areas it touches, one per side
    connects: bool         # both sides already seen (>= 2 separate entries)

    def to_json(self, origin, cell):
        xy = lambda c: [round(float(origin[k] + (c[k] + 0.5) * cell), 3) for k in (0, 1)]  # noqa: E731
        return {"id": self.id, "center": xy(self.cells.mean(0)), "entries": [xy(e) for e in self.entries],
                "clearance_m": round(self.clearance, 3), "posture": "crouch", "beyond_seen": self.connects,
                "status": "deferred"}


def find_gaps(cls: np.ndarray, clear: np.ndarray, cell: float, body: Body = Body(), min_cells: int = 3):
    """Low gaps that can be approached from WALK space, from the outputs of ``classify``.

    The far side may still be unknown, so an overhang counts when the robot's centre would fit under it with unknown
    space taken as passable (this drops openings narrower than the robot). ``cells`` and ``connects`` stay conservative.
    """
    robot = robot_classes(cls, cell, body)
    optimistic = robot_classes(np.where(cls == UNKNOWN, CROUCH, cls), cell, body)
    under = (cls == CROUCH) & (optimistic >= CROUCH)
    r = int(np.ceil(body.half_width / cell - 1e-6)) + 1
    lab, n = ndimage.label(robot == CROUCH, structure=_E8)
    gaps = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < min_cells:
            continue
        near = ndimage.binary_dilation(m, structure=np.ones((2 * r + 1, 2 * r + 1), bool)) & under
        if not near.any():
            continue
        rim = ndimage.binary_dilation(m, structure=_E8) & (robot == WALK)
        sides, ns = ndimage.label(rim, structure=_E8)
        if ns == 0:
            continue
        entries = [np.argwhere(sides == s).mean(0) for s in range(1, ns + 1)]
        gaps.append(Gap(len(gaps), np.argwhere(m), float(clear[near].min()), entries, ns >= 2))
    return gaps
