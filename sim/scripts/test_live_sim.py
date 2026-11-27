"""Check clue/live.py LiveSearch against the simulator: the simulator only senses and walks, LiveSearch picks every
goal from what it was given (map, pose, camera footprint, parsed clues, victim reports). Compare with Search on the
same buildings.

    python scripts/test_live_sim.py --seeds 0:40 --quality spoken
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clue import search2d as S  # noqa: E402
from clue.live import LiveSearch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0:40")
ap.add_argument("--quality", default="spoken")
ap.add_argument("--p_wrong", type=float, default=0.0)
ap.add_argument("--inflate", type=int, default=1)
a = ap.parse_args()
INFLATE = a.inflate


def live_run(seed):
    b = S.make_building(seed, quality=a.quality, p_wrong=a.p_wrong)
    sim = S.Search(b, beta=0.0, seed=seed, use_clues=False)    # truth: sensing + walking only
    live = LiveSearch(b.grid.shape, n_unclued=len(b.victims) - len(b.clues), inflate=INFLATE)
    orig = sim.sense

    def sense(spin=False):                 # the real node looks at every pose update, not only when it decides
        orig(spin)
        live.set_map(sim.known.copy())
        live.set_pose(sim.pos, sim.head)
        live.t = sim.t
        live.look(spin)
    sim.sense = sense
    sim.sense(spin=True)
    live.set_map(sim.known.copy())
    live.set_pose(sim.pos, sim.head)
    live.t = sim.t
    live.head = b.facing                   # simulator clues are in the reporter's frame (real robot: robot frame)
    for c in b.clues:
        live.add_clue({k: v for k, v in c.items() if k != "victim"} | {"utterance": "clue"})
    sim.t += S.TALK_S
    nf = 0
    for _ in range(600):
        if all(f is not None for f in sim.found) or sim.t > S.LIMIT_S:
            break
        live.set_map(sim.known.copy())
        live.set_pose(sim.pos, sim.head)
        live.t = sim.t
        live.look()
        for i, f in enumerate(sim.found):
            if f is not None and i not in getattr(live, "_reported", set()):
                live._reported = getattr(live, "_reported", set()) | {i}
                live.victim_found(sim.vpos[i])
        tgt, _ = live.next_goal()
        if tgt is None:
            break
        d, pred = sim.plan()
        if not np.isfinite(d[tgt]):
            break
        if tgt == sim.pos:
            sim.t += S.SPIN_S
            sim.sense(spin=True)
            continue
        sim.go(tgt, d, pred)
    f = [x for x in sim.found if x is not None and x <= S.LIMIT_S]
    t_all = max(f) if len(f) == len(sim.found) else S.LIMIT_S
    ref = S.Search(b, beta=0.75, trust_update="reporter", seed=seed).run()
    A = S.Search(b, beta=0.0, seed=seed).run()
    return dict(seed=seed, live=t_all, search=ref["t_all"], A=A["t_all"])


if __name__ == "__main__":
    lo, hi = map(int, a.seeds.split(":"))
    with Pool(16) as p:
        rows = p.map(live_run, range(lo, hi))
    m = {k: round(float(np.mean([r[k] for r in rows])), 1) for k in ("live", "search", "A")}
    d = np.array([r["live"] - r["search"] for r in rows])
    print(json.dumps(m), "live - search mean", round(d.mean(), 1), "median", round(float(np.median(d)), 1))
