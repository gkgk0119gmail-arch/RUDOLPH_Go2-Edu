"""G2 follow-ups on fresh worlds: (1) misreading the clue, (2) memory of deferred gaps. Fixed before the run:

base = linked_nop (pass probability showed no effect in G2). Reference without language = posture.
(1) The break-even misreading rate is where base is no longer faster than posture (mean, cap 600 s).
(2) Memory matters if base is faster than memory="forget" AND "neutral" with paired 95 % CI excluding 0 in both sizes.
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
from g2_sim import run_variants  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out")
a = ap.parse_args()
PS = (0.1, 0.2, 0.3, 0.5, 1.0)
V = {"posture": ("posture", {}), "separate": ("separate", {}), "base": ("linked_nop", {}),
     "mem_neutral": ("linked_nop", dict(memory="neutral")), "mem_forget": ("linked_nop", dict(memory="forget"))}
V |= {f"clue{p}": ("linked_nop", dict(misread="clue", p=p)) for p in PS}
V |= {f"pick{p}": ("linked_nop", dict(misread="pick", p=p)) for p in PS}
out = {}
for size, rooms in (("3x4", (3, 4)), ("5x7", (5, 7))):
    with Pool(min(32, os.cpu_count())) as pool:
        rs = pool.map(run_variants, [(k, rooms, V) for k in range(a.start, a.start + a.n)], chunksize=1)
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(rs), (5000, len(rs)))
    T = {n: np.array([r[n]["t"] for r in rs]) for n in V}
    gap = np.array([r["behind_gap"] for r in rs])
    tab = {}
    for n in V:
        d = T[n][B].mean(1) - T["base"][B].mean(1)
        e = T[n][B].mean(1) - T["posture"][B].mean(1)
        tab[n] = dict(mean=round(T[n].mean(), 1), gap=round(T[n][gap].mean(), 1), room=round(T[n][~gap].mean(), 1),
                      found=round(float(np.mean([r[n]["found"] for r in rs])), 3),
                      slower_than_base=[round(float(x), 1) for x in (T[n].mean() - T["base"].mean(), *np.percentile(d, [2.5, 97.5]))],
                      minus_posture=[round(float(x), 1) for x in (T[n].mean() - T["posture"].mean(), *np.percentile(e, [2.5, 97.5]))])
        print(size, n, tab[n])
    out[size] = dict(table=tab, runs=rs)
if a.out:
    Path(a.out).write_text(json.dumps(dict(args=vars(a), **out)))
