"""Run the search node's grid + planner offline on recorded scans (record_scans.py), same call order as the node.

    python offline_search.py scans.npz --clue 40 "로봇 오른쪽 앞으로 8미터쯤에 한 명 쓰러져 있어요" --parsed '{...}'
    -> logs/offline_state.npz (render with render_rudolph_state.py) + timing
The robot path is the recorded one (it does not follow the goals); this checks mapping, searched area, clue maps and
which goals the planner would send.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from clue.live import LiveGrid, LiveSearch  # noqa: E402
from clue.search2d import CELL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scans")
ap.add_argument("--size", type=float, default=120.0)
ap.add_argument("--clue", nargs=2, action="append", default=[], metavar=("SCAN_INDEX", "TEXT"))
ap.add_argument("--parsed", action="append", default=[], help="parsed JSON for each --clue (else a direct HTTP call)")
ap.add_argument("--parser", default="http://165.132.135.75:8765")
ap.add_argument("--every", type=int, default=5, help="plan every n scans (node: every 0.5 s at 10 Hz scans)")
ap.add_argument("--out", default=str(Path(__file__).parent / "logs/offline_state.npz"))
a = ap.parse_args()

z = np.load(a.scans)
pts, idx, poses = z["pts"], z["idx"], z["poses"]
g = LiveGrid(size_m=a.size, center_xy=poses[0][:2])
s = LiveSearch((g.n, g.n), n_unclued=1)
clues = {int(i): t for i, t in a.clue}
parsed = [json.loads(p) for p in a.parsed]
trail, goals, tg, tp = [], [], [], []
for i in range(len(poses)):
    px, py, pz, yaw = poses[i]
    if not g.inside(px, py):
        continue
    t = time.time()
    g.add_scan(pts[idx[i]:idx[i + 1]], (px, py, pz))
    tg.append(time.time() - t)
    trail.append((px, py))
    if i in clues:
        if parsed:
            p = parsed.pop(0)
        else:
            import urllib.request
            req = urllib.request.Request(a.parser + "/parse", json.dumps({"text": clues[i]}).encode(),
                                         {"Content-Type": "application/json"})
            p = json.loads(urllib.request.urlopen(req, timeout=60).read())["parsed"]
        s.set_pose(g.cell(px, py), yaw)
        print(i, s.add_clue(p, text=clues[i]))
    if i % a.every == 0:
        t = time.time()
        s.set_map(g.known())
        s.set_pose(g.cell(px, py), yaw)
        s.look()
        goal, _ = s.next_goal()
        tp.append(time.time() - t)
        if goal is not None:
            goals.append(tuple(float(v) for v in g.world(*goal)))
s.set_map(g.known())
np.savez_compressed(a.out, known=s.known, searched=s.searched, heat=s.heat(), trail=np.array(trail),
                    goals=np.array(goals), x0=g.x0, y1=g.y1, cell=CELL, floor_z=g.floor_z,
                    clues=json.dumps([dict(c, anchor_rc=[int(v) for v in c["anchor_rc"]]) for c in s.clues],
                                     ensure_ascii=False))
print(f"{len(poses)} scans, floor z {g.floor_z:.2f}, grid {1000 * np.mean(tg):.1f} ms/scan, "
      f"plan cycle {1000 * np.mean(tp):.0f} ms (max {1000 * np.max(tp):.0f}), {len(goals)} goals -> {a.out}")
