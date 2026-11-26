"""Sentences for condition C: every clue of buildings 0-119 (semantic and spoken), one sentence each.
-> results/clue/c_sentences.jsonl  (id, clue sentence, gold JSON, sim clue)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clue import schema as S, search2d as S2, speech  # noqa: E402

out = []
for q in ("semantic", "spoken"):
    for seed in range(120):
        b = S2.make_building(seed, n_clues=3, quality=q)
        rng = np.random.default_rng([seed, 99, q == "semantic"])
        for k, c in enumerate(b.clues):
            sim = {kk: v for kk, v in c.items() if not kk.startswith("_") and kk not in ("victim", "wrong")}
            gold, _ = S.normalize(sim)
            out.append({"id": f"{q}_{seed}_{k}", "clue": speech.sentence(c, rng), "gold": gold, "sim": c, "source": "llm-slot-patterns"})
p = ROOT / "results/clue/c_sentences.jsonl"
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(len(out), p)
for r in out[:4] + out[-4:]:
    print(r["clue"], "|", r["sim"].get("semantic") or (r["sim"]["direction_deg"], r["sim"]["distance_m"]))
