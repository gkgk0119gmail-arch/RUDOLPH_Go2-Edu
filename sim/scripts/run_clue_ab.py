"""Plan-doc stage 2: A (coverage) vs B (clue belief map, gold JSON) on generated buildings, paired by seed.

    python scripts/run_clue_ab.py --seeds 0:120 --out results/clue/ab_v1.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import search2d as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0:120")
ap.add_argument("--out", required=True)
ap.add_argument("--n_victims", type=int, default=6)
ap.add_argument("--n_clues", type=int, default=3)
ap.add_argument("--quality", default="spoken")
ap.add_argument("--variants", default="A,B")
ap.add_argument("--p_wrong", type=float, default=0.0)
ap.add_argument("--mobile", type=float, default=0.0, help="victims' random-walk D [m^2/s]")
ap.add_argument("--clue_age", type=float, default=0.0, help="seconds between the sighting and the clue")
ap.add_argument("--parsed", help="parse output jsonl over results/clue/c_sentences.jsonl -> adds condition C")
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ALL = {"A": dict(beta=0.0), "B": dict(beta=1.0), "A_doc": dict(beta=0.0, score="doc"), "B_doc": dict(beta=1.0, score="doc"),
       "B_mobile": dict(beta=1.0, mobility="unknown"), "B_spread": dict(beta=1.0, mobility="mobile"),
       "A_noexit": dict(beta=0.0, early_exit=False), "B_noexit": dict(beta=1.0, early_exit=False),
       "B_kw": dict(beta=1.0, keyword=True),
       "B_tc": dict(beta=1.0, trust_update="clue"), "B_tr": dict(beta=1.0, trust_update="reporter"),
       "B90": dict(beta=0.9), "B90_tc": dict(beta=0.9, trust_update="clue"), "B90_tr": dict(beta=0.9, trust_update="reporter"),
       "B75": dict(beta=0.75), "B75_tc": dict(beta=0.75, trust_update="clue"), "B75_tr": dict(beta=0.75, trust_update="reporter"),
       "A_wit": dict(beta=1.0, witness=True, use_clues=True, clues=[]), "B_wit": dict(beta=1.0, witness=True)}
ap2 = None
V = {k: ALL[k] for k in a.variants.split(",") if k in ALL}
if a.parsed:   # condition C: what the parser made of each clue sentence (unusable -> the clue is dropped)
    from clue import speech  # noqa: E402
    rows_p = [json.loads(l) for l in open(a.parsed)]
    per = {}
    for r in rows_p:
        q, seed, k = r["id"].rsplit("_", 2)
        if q != a.quality:
            continue
        c = speech.to_sim(r["pred"], r["sim"]["victim"])
        per.setdefault(int(seed), []).append(c)
    V["C"] = dict(beta=1.0, clues={sd: [c for c in cs if c is not None] for sd, cs in per.items()})
lo, hi = map(int, a.seeds.split(":"))
kw = dict(n_victims=a.n_victims, n_clues=a.n_clues, quality=a.quality, p_wrong=a.p_wrong, mobile=a.mobile, clue_age=a.clue_age)


def boot(x, n=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


if __name__ == "__main__":
    t = time.time()
    with Pool(a.workers) as p:
        rows = p.map(S.run_job, [(s, V, kw) for s in range(lo, hi)])
    summ = {}
    for k in V:
        summ[k] = {m: round(float(np.mean([r[k][m] for r in rows])), 3) for m in ("t_all", "t_clued", "t_first", "auc", "n_found", "all_found", "path", "coverage", "t_mission", "n_witness")}
    base = list(V)[0]
    for k in list(V)[1:]:
        for m in ("t_all", "t_clued", "auc", "t_first", "t_mission"):
            mu, l, h = boot([r[k][m] - r[base][m] for r in rows])
            summ[k][f"d_{m}_vs_{base}"] = [round(mu, 3), round(l, 3), round(h, 3)]
    if "C" in V:
        cl = [c for cs in V["C"]["clues"].values() for c in cs]
        summ["C"]["usable_clues"] = len(cl)
        V["C"]["clues"] = "per seed, from " + a.parsed
    out = dict(args=vars(a), seconds=round(time.time() - t, 1), summary=summ, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(args=vars(a), summary=summ), indent=1))
