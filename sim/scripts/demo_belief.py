"""Picture of the clue belief map on a small floor plan: prior, spreading for 10 min (static / unknown / mobile),
and a camera pass that finds no one.  -> results/clue/belief_demo.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import belief as B  # noqa: E402

CELL = 0.25
H, W = 96, 128  # 24 m x 32 m
g = np.full((H, W), B.FREE)
g[[0, -1], :] = B.OCC
g[:, [0, -1]] = B.OCC
g[40, :] = g[56, :] = B.OCC                     # corridor rows 41-55
for c in range(1, W, 25):                        # room walls above and below the corridor
    g[:40, c] = g[57:, c] = B.OCC
for c in range(13, W, 25):                       # doors into the corridor
    g[40, c - 2:c + 2] = g[56, c - 2:c + 2] = B.FREE
truth_free = g.copy()
known = np.zeros_like(g, bool)
known[30:66, :40] = True                          # robot has mapped the left part
grid = np.where(known, g, B.UNK)
robot = (48, 8)
clue = {"anchor": "reporter", "direction_deg": 90, "distance_m": 18.0, "certainty": "mid"}  # "제 오른쪽 18 m쯤일 거예요"
prior = B.clue_prior(grid, CELL, robot, np.pi / 2, clue)  # reporter faces north -> right = east, along the corridor
panels = [("clue: 'my right, ~18 m, probably'", prior)]
for name in ("static", "unknown", "mobile"):
    panels.append((f"+10 min, {name} (D={B.D_MOBILITY[name]} m2/s)", B.spread(prior, grid, CELL, 600.0, B.D_MOBILITY[name])))
seen = np.zeros_like(known)
seen[41:56, 40:100] = True                        # camera looked down the corridor the clue points at
obs, miss = B.observe(panels[2][1], seen, 0.8)
panels.append((f"corridor seen, nobody (P(miss) {miss:.2f})", obs))
fr = B.frontiers(grid)
fm = B.frontier_mass(obs, grid, fr)
fig, ax = plt.subplots(1, 6, figsize=(30, 5))
for a, (t, b) in zip(ax, panels):
    a.imshow(np.where(truth_free == B.OCC, np.nan, b), cmap="magma")
    a.contour(known, [0.5], colors="cyan", linewidths=0.8)
    a.plot(robot[1], robot[0], "c^")
    a.set_title(t, fontsize=10)
    a.axis("off")
r, c = np.array(list(fm.keys())).T
v = np.array(list(fm.values()))
ax[5].imshow(np.where(grid == B.OCC, 1, np.where(grid == B.UNK, 0.5, 0)), cmap="gray_r", vmin=0, vmax=1.5)
s = ax[5].scatter(c, r, c=v, s=8 + 400 * v / v.max(), cmap="viridis")
ax[5].set_title("mass behind each frontier (geodesic)", fontsize=10)
ax[5].axis("off")
out = ROOT / "results/clue/belief_demo.png"
fig.savefig(out, dpi=80, bbox_inches="tight")
print(out, "frontiers", len(fm), "max mass", v.max().round(3))
