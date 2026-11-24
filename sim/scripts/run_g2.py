"""Gate G2 (2D). python scripts/run_g2.py --start 1000 --n 400 --out results/gap_search/g2.json

Fixed before the test run: primary = restricted mean time to find B (cap 600 s), linked vs separate, all test worlds;
pass if, in BOTH building sizes (3x4 and 5x7 rooms), the mean is >= 15 % lower and the paired bootstrap 95 % CI of the
reduction excludes 0. Other references (posture, clue, linked_nop) are reported next to it.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "go2_lab" / "gap_search"))
from g2_sim import METHODS, run_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out")
a = ap.parse_args()
summ, res_all = {}, {}
for size, rooms in (("3x4", (3, 4)), ("5x7", (5, 7))):
    with Pool(min(32, os.cpu_count())) as p:
        res = p.map(run_seed, [(k, rooms) for k in range(a.start, a.start + a.n)], chunksize=1)
    res_all[size] = res
    rng = np.random.default_rng(0)
    for name, sel in (("all", lambda r: True), ("B behind gap", lambda r: r["behind_gap"]),
                      ("B in walkable room", lambda r: not r["behind_gap"])):
        rs = [r for r in res if sel(r)]
        T = {m: np.array([r[m]["t"] for r in rs]) for m in METHODS}
        s = {"n": len(rs)}
        for m in METHODS:
            s[m] = dict(mean_t=round(float(T[m].mean()), 1), found=round(float(np.mean([r[m]["found"] for r in rs])), 3),
                        tries=round(float(np.mean([r[m]["tries"] for r in rs])), 2),
                        fails=round(float(np.mean([r[m]["fails"] for r in rs])), 2))
        B = rng.integers(0, len(rs), (5000, len(rs)))
        for ref in ("posture", "clue", "separate", "linked_nop"):
            red = 1 - T["linked"][B].mean(1) / T[ref][B].mean(1)
            s[f"linked_vs_{ref}"] = dict(reduction=round(float(1 - T["linked"].mean() / T[ref].mean()), 4),
                                         ci95=[round(float(q), 4) for q in np.percentile(red, [2.5, 97.5])])
        summ[f"{size} | {name}"] = s
print(json.dumps(summ, indent=1))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(args=vars(a), summary=summ, runs=res_all), indent=0))
