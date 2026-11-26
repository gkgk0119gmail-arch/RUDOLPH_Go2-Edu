"""3D animation of the victim belief map: clue -> spreading through doors (walls block) -> robot camera sweeps the
corridor and the mass it sees drains into the rooms.  -> results/clue/belief_3d.gif

Map is fully known here (walls block the spread) to make the picture easy to read."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import belief as B  # noqa: E402

CELL = 0.5
H, W = 48, 64                       # 24 m x 32 m
g = np.full((H, W), B.FREE)
walls = []                          # (r0, r1, c0, c1) boxes for drawing


def wall(r0, r1, c0, c1):
    g[r0:r1, c0:c1] = B.OCC
    walls.append((r0, r1, c0, c1))


wall(0, 1, 0, W); wall(H - 1, H, 0, W); wall(0, H, 0, 1); wall(0, H, W - 1, W)  # noqa: E702
for r in (20, 28):                  # corridor rows 21-27, doors 2 cells wide
    for c0, c1 in ((1, 5), (7, 18), (20, 31), (33, 44), (46, 57), (59, 63)):
        wall(r, r + 1, c0, c1)
for c in (13, 26, 39, 52):          # room walls
    wall(1, 20, c, c + 1)
    wall(29, H - 1, c, c + 1)

reporter = (24, 3)
clue = {"anchor": "reporter", "direction_deg": 90, "distance_m": 18.0, "certainty": "mid"}
prior = B.clue_prior(g, CELL, reporter, np.pi / 2, clue)    # faces north -> "right" = east along the corridor

# timeline: 0-11 clue appears, 12-41 ten minutes pass (unknown mobility), 42-89 robot walks the corridor looking ahead
frames, texts, robots, fovs = [], [], [], []
b = prior.copy()
for i in range(12):
    frames.append(prior * (i + 1) / 12)
    texts.append("1) Clue: 'on my right, ~18 m, probably'  ->  probability mass appears")
    robots.append(None)
    fovs.append(None)
for i in range(30):
    b = B.spread(b, g, CELL, 20.0, B.D_MOBILITY["unknown"])
    frames.append(b.copy())
    texts.append(f"2) Time passes ({(i + 1) * 20 / 60:.0f} min): mass leaks to neighbour cells,\nwalls block, doors let it through")
    robots.append(None)
    fovs.append(None)
rr, cc = np.mgrid[0:H, 0:W]
for i in range(48):
    rc = (24, 3 + i * 1.2)
    seen = (np.hypot(rr - rc[0], cc - rc[1]) * CELL < 4.0) & (rr > 20) & (rr < 28) & (cc >= rc[1])
    b = B.spread(b, g, CELL, 10.0, B.D_MOBILITY["unknown"])
    b, _ = B.observe(b, seen, 0.8)
    frames.append(b.copy())
    texts.append("3) Robot camera (green) sees the corridor, nobody there\n-> that mass drains, rooms behind the doors grow")
    robots.append(rc)
    fovs.append(seen)

zmax = max(f.max() for f in frames) * 1.05
fig = plt.figure(figsize=(9, 6.2))
ax = fig.add_subplot(111, projection="3d")
X, Y = np.meshgrid(np.arange(W) * CELL, np.arange(H) * CELL)


def draw(k):
    ax.clear()
    z = frames[k] * (g != B.OCC)
    ax.plot_surface(X, Y, z, cmap="inferno", vmin=0, vmax=zmax, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, alpha=0.95)
    for r0, r1, c0, c1 in walls:   # walls as outlines on the floor
        xs = np.array([c0, c1, c1, c0, c0]) * CELL
        ys = np.array([r0, r0, r1, r1, r0]) * CELL
        ax.plot(xs, ys, 0, color="#4aa3ff", lw=1.2)
    if fovs[k] is not None:
        fr, fc = np.nonzero(fovs[k])
        ax.scatter(fc * CELL, fr * CELL, 0, c="lime", s=6, alpha=0.5, depthshade=False)
    ax.scatter([reporter[1] * CELL], [reporter[0] * CELL], [0], c="deepskyblue", s=70, marker="o", depthshade=False)
    ax.text(reporter[1] * CELL, reporter[0] * CELL - 1.5, zmax * 0.15, "reporter", color="deepskyblue", fontsize=8)
    if robots[k] is not None:
        r, c = robots[k]
        ax.scatter([c * CELL], [r * CELL], [0], c="lime", s=90, marker="^", depthshade=False)
        ax.text(c * CELL, r * CELL - 1.5, zmax * 0.15, "robot", color="green", fontsize=8)
    ax.set_zlim(0, zmax)
    ax.set_xlim(0, W * CELL)
    ax.set_ylim(H * CELL, 0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("P(victim)")
    ax.set_zticks([])
    ax.view_init(elev=42, azim=-70 + 0.3 * k)
    ax.set_title(texts[k], fontsize=10)


anim = FuncAnimation(fig, draw, frames=len(frames))
out = ROOT / "results/clue/belief_3d.gif"
anim.save(out, writer=PillowWriter(fps=8), dpi=72)
print(out, len(frames), "frames")
