"""Animate one 2D search: building map (true walls / what the robot has mapped / camera-searched), robot and trail,
clue probability map weighted by the clue trust w_k, true victim positions, the rooms each clue names, and w_k over
time. Top row: fixed trust (B, w = 1). Bottom row: per-reporter Bayesian trust update (w 0.75, rho 0.7, w_lo 0.1).

    python scripts/anim_search2d.py --seed 115 --p_wrong 0.33   -> results/clue/search2d_s115.{gif,png}
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import search2d as S  # noqa: E402

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f.replace(" ", "") or "Noto Sans CJK" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=115)
ap.add_argument("--quality", default="semantic")
ap.add_argument("--p_wrong", type=float, default=0.33)
ap.add_argument("--every", type=int, default=3, help="keep every n-th sensing step as a frame")
ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()

b = S.make_building(a.seed, quality=a.quality, p_wrong=a.p_wrong)
RUNS = [("고정 신뢰도 (w = 1)", dict(beta=1.0)),
        ("제보자별 신뢰도 갱신 (베이즈)", dict(beta=0.75, trust_update="reporter"))]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def clue_room(c):
    """Room box the clue names (side + ordinal), or None."""
    sem = c.get("semantic") or []
    side = next((t.split(":")[1] for t in sem if t.startswith(("side:", "turn:"))), None)
    n = next((t.split(":")[1] for t in sem if t.startswith("room:")), None)
    for r in b.rooms:
        if r["side"] == side and n and n.isdigit() and r["ordinal"] == int(n):
            return r["box"]
    return None


def record(kw):
    s = S.Search(b, seed=a.seed, **kw)
    frames, n = [], [0]
    orig = s.sense

    def snap():
        heat = np.zeros(b.grid.shape)
        ws = {}
        for kk in range(len(s.clues)):
            ws[kk] = s.weight(kk) if s.active.get(kk) else None
        for w, m in (s.beliefs() if s.active else []):
            heat += w * m
        frames.append(dict(t=s.t, pos=s.pos, known=s.known.copy(), searched=s.searched.copy(), heat=heat,
                           found=list(s.found), w=ws, trail=None))

    def sense(*args, **k):
        orig(*args, **k)
        n[0] += 1
        if n[0] % a.every == 0:
            snap()
    s.sense = sense
    res = s.run()
    snap()                               # the moment the search ends
    trail, last = [], None
    for f in frames:                     # robot trail from the recorded positions
        if f["pos"] != last:
            trail.append(f["pos"])
            last = f["pos"]
        f["trail"] = list(trail)
    return res, frames


runs = [(name, *record(kw)) for name, kw in RUNS]
n_frames = max(len(fr) for _, _, fr in runs)
H, W = b.grid.shape
ext = (0, W * S.CELL, H * S.CELL, 0)
hmax = max(f["heat"].max() for _, _, fr in runs for f in fr) or 1.0

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.35, wspace=0.12)
axm = [fig.add_subplot(gs[i, 0]) for i in range(2)]
axw = [fig.add_subplot(gs[i, 1]) for i in range(2)]


def draw(k):
    for row, (name, res, fr) in enumerate(runs):
        f = fr[min(k, len(fr) - 1)]
        ax = axm[row]
        ax.clear()
        img = np.ones((H, W, 3))
        truth_wall = b.grid == S.OCC
        img[truth_wall] = (0.85, 0.85, 0.85)                       # true walls not yet seen: light gray
        img[f["known"] == S.UNK] *= 0.93                           # unmapped: slightly darker
        img[(f["known"] == S.FREE)] = (0.97, 0.97, 0.97)
        img[f["searched"] & (b.grid == S.FREE)] = (0.86, 0.95, 0.86)  # seen by the camera: green tint
        img[(f["known"] == S.OCC)] = (0.15, 0.15, 0.15)              # mapped walls: black
        ax.imshow(img, extent=ext, interpolation="nearest")
        h = np.ma.masked_less(f["heat"], hmax * 0.01)
        ax.imshow(h, extent=ext, cmap="inferno_r", vmin=0, vmax=hmax, alpha=0.75, interpolation="nearest")
        for ci, c in enumerate(b.clues):                           # rooms the clues name
            box = clue_room(c)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            on = f["w"].get(ci) is not None
            ax.add_patch(Rectangle((x0 * S.CELL, y0 * S.CELL), (x1 - x0) * S.CELL, (y1 - y0) * S.CELL, fill=False,
                                   ec=COL[ci], lw=2.2 if on else 1.0, ls="-" if on else ":"))
            ax.text(x0 * S.CELL + 0.3, y0 * S.CELL + 1.3, f"단서{ci + 1}" + (" (틀림)" if c.get("wrong") else ""),
                    color=COL[ci], fontsize=8, weight="bold")
        for vi, v in enumerate(b.victims):                         # true victim positions
            y, x = v
            done = f["found"][vi] is not None
            ax.scatter(x * S.CELL, y * S.CELL, marker="o" if done else "X", s=70,
                       c="limegreen" if done else "red", edgecolors="k", linewidths=0.6, zorder=5)
            for ci, c in enumerate(b.clues):                       # whose position each clue is about
                if c["victim"] == vi:
                    ax.text(x * S.CELL + 0.5, y * S.CELL - 0.5, f"단서{ci + 1}의 사람", color=COL[ci], fontsize=7,
                            weight="bold", zorder=6)
        if f["trail"]:
            ty, tx = zip(*f["trail"])
            ax.plot(np.array(tx) * S.CELL, np.array(ty) * S.CELL, color="#1f77b4", lw=1.2, alpha=0.8)
        y, x = f["pos"]
        ax.scatter(x * S.CELL, y * S.CELL, marker="^", s=120, c="cyan", edgecolors="k", zorder=6)
        sy, sx = b.start
        ax.scatter(sx * S.CELL, sy * S.CELL, marker="s", s=50, c="white", edgecolors="k", zorder=4)
        nf = sum(x is not None for x in f["found"])
        ax.set_title(f"{name}   t = {f['t']:.0f} s   발견 {nf}/{len(b.victims)}"
                     + (f"   (전원 발견 {res['t_all']:.0f} s)" if nf == len(b.victims) else ""), fontsize=10)
        ax.set_xlim(0, W * S.CELL)
        ax.set_ylim(H * S.CELL, 0)
        ax.set_xticks([])
        ax.set_yticks([])

        aw = axw[row]
        aw.clear()
        t = [g["t"] for g in fr[:k + 1]]
        for ci, c in enumerate(b.clues):
            if ci >= 3:
                break
            wv = [g["w"].get(ci) for g in fr[:k + 1]]
            aw.plot(t, [np.nan if v is None else v for v in wv], color=COL[ci], lw=2,
                    label=f"단서{ci + 1}" + (" (틀림)" if c.get("wrong") else ""))
        aw.set_xlim(0, max(g["t"] for g in fr) + 5)
        aw.set_ylim(-0.03, 1.05)
        aw.set_title("단서 신뢰도 w (사람 찾으면 끝)", fontsize=9)
        aw.set_xlabel("시간 [s]", fontsize=8)
        aw.tick_params(labelsize=7)
        aw.legend(fontsize=7, loc="lower left")
        aw.grid(alpha=0.3)
    fig.suptitle(f"2D 수색 시뮬레이션 (건물 {a.seed}): X = 아직 못 찾은 요구조자 실제 위치, ● = 찾음, ▲ = 로봇, "
                 "색 = 단서 확률 × 신뢰도, 초록 = 카메라로 본 곳", fontsize=9)


out = ROOT / f"results/clue/search2d_s{a.seed}"
draw(n_frames - 1)
fig.savefig(f"{out}_end.png", dpi=110)
k_mid = min(range(n_frames), key=lambda i: abs(runs[1][2][min(i, len(runs[1][2]) - 1)]["t"] - 120))
draw(k_mid)                              # mid-search: clue maps and trust still in play
fig.savefig(f"{out}.png", dpi=110)
anim = FuncAnimation(fig, draw, frames=n_frames)
anim.save(f"{out}.gif", writer=PillowWriter(fps=a.fps), dpi=70)
print(out, n_frames, "frames", [(name, round(r["t_all"])) for name, r, _ in runs])
