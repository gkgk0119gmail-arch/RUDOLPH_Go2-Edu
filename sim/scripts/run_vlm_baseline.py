"""COMRES-style baseline: the VLM picks every next goal from a candidate table (no belief map), vs our A / B on the
same buildings. The VLM reads the actual clue sentences (results/clue/c_sentences_s60.jsonl); B gets the gold JSON.

    python scripts/run_vlm_baseline.py --seeds 0:60 --quality semantic --out results/clue/vlm_baseline_semantic.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import search2d as S  # noqa: E402
from clue.vlm_planner import VLMPlanner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0:60")
ap.add_argument("--quality", default="semantic")
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--sentences", default=str(ROOT / "results/clue/c_sentences_s60.jsonl"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

text = {}
for l in open(a.sentences):
    r = json.loads(l)
    q, sd, k = r["id"].rsplit("_", 2)
    if q == a.quality:
        text.setdefault(int(sd), []).append(r["sim"] | {"text": r["clue"]})

vlm = VLMPlanner(a.model, a.device)
lo, hi = map(int, a.seeds.split(":"))
M = ("t_all", "t_clued", "t_first", "auc", "n_found", "path", "coverage")
rows = []
out_path = Path(a.out)
for sd in range(lo, hi):
    b = S.make_building(sd, quality=a.quality)
    c0, l0, bad0 = vlm.calls, vlm.lat, vlm.bad
    r = dict(seed=sd,
             A=S.Search(b, beta=0.0, seed=sd).run(),
             B=S.Search(b, beta=1.0, seed=sd).run(),
             VLM=S.Search(b, beta=1.0, seed=sd, score="vlm", vlm=vlm, clues=text[sd]).run())
    r["vlm_calls"], r["vlm_lat"], r["vlm_bad"] = vlm.calls - c0, vlm.lat - l0, vlm.bad - bad0
    rows.append(r)
    print(sd, {k: round(r[k]["t_all"]) for k in ("A", "B", "VLM")}, "clued", {k: round(r[k]["t_clued"]) for k in ("A", "B", "VLM")},
          f"calls {r['vlm_calls']} {r['vlm_lat'] / max(r['vlm_calls'], 1):.2f}s bad {r['vlm_bad']}", flush=True)
    summ = {k: {m: round(float(np.mean([x[k][m] for x in rows])), 3) for m in M} for k in ("A", "B", "VLM")}
    summ["vlm"] = dict(calls_per_run=float(np.mean([x["vlm_calls"] for x in rows])),
                       latency_s=vlm.lat / max(vlm.calls, 1), bad_replies=vlm.bad)
    out_path.write_text(json.dumps(dict(args=vars(a), n=len(rows), summary=summ, rows=rows), indent=1))
print(json.dumps(summ, indent=1))
