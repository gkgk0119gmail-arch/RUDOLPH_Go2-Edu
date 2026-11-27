"""Learn the goal-selector weights (plan doc stage 4) by black-box optimisation on training buildings.

theta = (w: trust in clues, log10 t0: travel-time offset, kappa: travel-time exponent);  S = E / (C + t0)^kappa,
E = w * clue mass + (unclued people + (1 - w) * clues) * unsearched share.
Training: seeds 0-59 x a mix of clue conditions; objective = mean time to find everyone. Test: seeds 1000-1119.

    python scripts/tune_selector.py --out results/clue/tune_v1.json
    python scripts/tune_selector.py --mode trust --out results/clue/tune_trust_v1.json
        theta = (w, rho, w_lo) of the per-reporter Bayesian trust update (t0 10, kappa 1 kept)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import search2d as S  # noqa: E402

CONDS = [dict(quality="spoken", n_clues=3), dict(quality="semantic", n_clues=3),
         dict(quality="spoken", n_clues=3, p_wrong=0.33), dict(quality="semantic", n_clues=3, p_wrong=0.33),
         dict(quality="semantic", n_clues=1), dict(quality="spoken", n_clues=6)]
BOUNDS = [(0.0, 1.0), (0.0, 2.3), (0.5, 2.0)]
CONDS_TRUST = [dict(quality="semantic", n_clues=3), dict(quality="semantic", n_clues=3, p_wrong=0.33),
               dict(quality="semantic", n_clues=3, p_wrong=1.0), dict(quality="spoken", n_clues=3),
               dict(quality="spoken", n_clues=3, p_wrong=0.33), dict(quality="spoken", n_clues=6)]
BOUNDS_TRUST = [(0.3, 1.0), (0.05, 0.99), (0.0, 0.6)]
MODE = "weights"


def params(th):
    if MODE == "trust":
        return dict(beta=float(th[0]), rho=float(th[1]), w_lo=float(th[2]), trust_update="reporter")
    return dict(beta=float(th[0]), t0=float(10 ** th[1]), kappa=float(th[2]))


def job(args):
    seed, cond, v = args
    return S.run_job((seed, {"x": v}, cond))["x"]


def evaluate(pool, th, seeds):
    v = dict(beta=0.0) if th is None else th if isinstance(th, dict) else params(th)
    res = pool.map(job, [(sd, c, v) for c in CONDS for sd in seeds], chunksize=4)
    t = np.array([r["t_all"] for r in res]).reshape(len(CONDS), -1)
    auc = np.array([r["auc"] for r in res]).reshape(len(CONDS), -1)
    tc = np.array([r["t_clued"] for r in res]).reshape(len(CONDS), -1)
    return t, auc, tc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="0:60")
    ap.add_argument("--test", default="1000:1120")
    ap.add_argument("--maxiter", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", default="weights", choices=["weights", "trust"])
    a = ap.parse_args()
    if a.mode == "trust":
        MODE, CONDS, BOUNDS = "trust", CONDS_TRUST, BOUNDS_TRUST
    tr = range(*map(int, a.train.split(":")))
    te = range(*map(int, a.test.split(":")))
    hist = []
    t_start = time.time()
    with Pool(a.workers) as pool:
        def f(th):
            t, _, _ = evaluate(pool, th, tr)
            hist.append(dict(theta=[float(x) for x in th], t_all=float(t.mean())))
            print(f"{len(hist):4d} {params(th)} t_all {t.mean():.1f}  ({time.time() - t_start:.0f}s)", flush=True)
            return float(t.mean())
        r = differential_evolution(f, BOUNDS, maxiter=a.maxiter, popsize=a.popsize, seed=0, polish=False, tol=1e-3)
        best = r.x
        out = dict(best=params(best), train_t_all=float(r.fun), history=hist, conds=CONDS, test={})
        tests = (("A_coverage", None), ("B_default", np.array([1.0, 1.0, 1.0])), ("B_learned", best))
        if MODE == "trust":   # B fixed w = 1; reporter update with the hand-set (0.75, 0.7, 0.1); learned
            tests = (("A_coverage", None), ("B_fixed", dict(beta=1.0)), ("B_tr_hand", np.array([0.75, 0.7, 0.1])),
                     ("B_tr_learned", best))
        for name, th in tests:
            t, auc, tc = evaluate(pool, th, te)
            out["test"][name] = dict(t_all=t.mean(1).round(1).tolist(), auc=auc.mean(1).round(3).tolist(),
                                     t_clued=tc.mean(1).round(1).tolist(), t_all_mean=round(float(t.mean()), 1),
                                     per_seed_t_all=t.tolist())
            print(name, out["test"][name]["t_all"], out["test"][name]["t_all_mean"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
