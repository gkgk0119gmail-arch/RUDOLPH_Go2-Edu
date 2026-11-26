"""Clue weight beta vs wrong-clue rate (2D search, rate score). -> results/clue/beta_sweep.json"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import search2d as S  # noqa: E402

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
WRONG = (0.0, 0.33, 1.0)
SEEDS = range(0, 120)

if __name__ == "__main__":
    V = {f"b{b}": dict(beta=b) for b in BETAS}
    res = {}
    with Pool(20) as p:
        for pw in WRONG:
            rows = p.map(S.run_job, [(s, V, dict(n_clues=3, p_wrong=pw)) for s in SEEDS])
            res[str(pw)] = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ("t_all", "auc", "t_first")} for k in V}
            print(pw, {k: round(v["t_all"], 1) for k, v in res[str(pw)].items()}, flush=True)
    Path(ROOT / "results/clue/beta_sweep.json").write_text(json.dumps(res, indent=1))
