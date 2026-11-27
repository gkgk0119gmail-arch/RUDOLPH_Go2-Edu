"""Run the clue parser with a prompt variant from clue/prompts.py (same scoring as clue/parse.py).

    python scripts/parse_variant.py --prompt v3 --data results/clue/pilot_llm_v1.jsonl --out results/clue/parse_8b_v3_pilot.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clue import parse as P  # noqa: E402
from clue import schema as S  # noqa: E402
from clue.prompts import PROMPTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True, choices=sorted(PROMPTS))
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
P.PROMPT = PROMPTS[a.prompt]          # Parser.batch formats the module-level PROMPT
data = [json.loads(l) for l in open(a.data)]
p = P.Parser(a.model, a.device)
rows = []
with open(a.out, "w") as f:
    for i in range(0, len(data), a.batch):
        chunk = data[i:i + a.batch]
        raws, lat = p.batch([d["clue"] for d in chunk])
        for d, raw in zip(chunk, raws):
            pred, bad = S.normalize(S.extract_json(raw))
            gold = S.normalize(d["sim"])[0] if "sim" in d else d["gold"]   # c_sentences: gold from the sim clue
            r = {**d, "gold": gold, "raw": raw, "pred": pred, "problems": bad, "lat": lat, "ok": S.compare(pred, gold),
                 "complexity": d.get("complexity") or S.complexity(gold), "prompt": a.prompt}
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{d['id']} {'OK ' if r['ok']['all'] else 'ERR'} {lat:.2f}s {len(raw)}ch {d['clue'][:36]}", flush=True)
s = P.summarize(rows)
s["out_chars"] = round(sum(len(r["raw"]) for r in rows) / len(rows), 1)
Path(a.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
print(json.dumps({k: s[k] for k in ("latency_s", "out_chars")} | {"all": s["all"]["all"]}, ensure_ascii=False))
