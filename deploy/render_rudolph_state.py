"""Draw the search node's saved state (logs/rudolph_state.npz): mapped grid, camera-searched area, clue map x trust,
robot trail, goals it chose, where each clue was heard.

    python3 render_rudolph_state.py logs/rudolph_state.npz --out rudolph_state.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f.replace(" ", ""):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--out", default="rudolph_state.png")
ap.add_argument("--title", default="")
a = ap.parse_args()
z = np.load(a.state, allow_pickle=False)
k, seen, heat, cell = z["known"], z["searched"], z["heat"], float(z["cell"])
x0, y1 = float(z["x0"]), float(z["y1"])
n = k.shape[0]
ys, xs = np.nonzero(k != -1)
m = 15
r0, r1, c0, c1 = max(ys.min() - m, 0), min(ys.max() + m, n), max(xs.min() - m, 0), min(xs.max() + m, n)
ext = (x0 + c0 * cell, x0 + c1 * cell, y1 - r1 * cell, y1 - r0 * cell)
img = np.full(k.shape + (3,), 0.80)
img[k == 0] = (0.97, 0.97, 0.97)
img[seen & (k == 0)] = (0.84, 0.94, 0.84)
img[k == 1] = (0.12, 0.12, 0.12)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img[r0:r1, c0:c1], extent=ext, interpolation="nearest")
h = heat[r0:r1, c0:c1]
if h.max() > 0:
    ax.imshow(np.ma.masked_less(h, h.max() * 0.02), extent=ext, cmap="inferno_r", alpha=0.7, interpolation="nearest")
tr = z["trail"]
if len(tr):
    ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=1.5, label="로봇 경로 (FAST-LIO)")
    ax.scatter(*tr[-1], marker="^", s=120, c="cyan", edgecolors="k", zorder=5)
gl = z["goals"]
if len(gl):
    ax.scatter(gl[:, 0], gl[:, 1], marker="*", s=70, c="magenta", edgecolors="k", lw=0.4, zorder=6,
               label=f"선택한 목표 ({len(gl)}개)")
    ax.scatter(*gl[-1], marker="*", s=260, c="magenta", edgecolors="k", zorder=7)
for i, c in enumerate(json.loads(str(z["clues"]))):
    r, cc = c["anchor_rc"]
    ax.scatter(x0 + (cc + 0.5) * cell, y1 - (r + 0.5) * cell, marker="D", s=60, c="orange", edgecolors="k", zorder=6)
    ax.annotate(f"단서{i + 1}: {c.get('text', '')[:24]}", (x0 + (cc + 0.5) * cell, y1 - (r + 0.5) * cell),
                textcoords="offset points", xytext=(8, 8), fontsize=8, color="darkorange", weight="bold")
ax.set_xlabel("x [m] (map frame)")
ax.set_ylabel("y [m]")
ax.legend(loc="lower right", fontsize=8)
ax.set_title(a.title or "RUDOLPH 수색 노드 상태: 검정 = 벽, 초록 = 카메라로 본 곳, 색 = 단서 확률 × 신뢰도", fontsize=10)
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print(a.out)
