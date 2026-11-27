"""Clue parser benchmark against any OpenAI-compatible server (llama.cpp `llama-server` on the Orin NX, or the
desktop). Same prompt (sim/clue/parse.py PROMPT) and same scoring as the desktop runs, so numbers compare directly.

On the NX (llama.cpp built with CUDA, text-only GGUF of Qwen3-VL-8B-Instruct, e.g. Q4_K_M ~5 GB):
    llama-server -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
    python3 deploy/bench_parse.py --url http://127.0.0.1:8080 --limit 30 --out deploy/logs/parse_nx_q4.jsonl
Only the standard library is needed (no torch / transformers).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from clue import schema as S  # noqa: E402
from clue.parse import PROMPT, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--data", default=str(ROOT / "sim/results/clue/pilot_llm_v1.jsonl"))
ap.add_argument("--limit", type=int)
ap.add_argument("--max_tokens", type=int, default=220)
ap.add_argument("--out", required=True)
a = ap.parse_args()

data = [json.loads(l) for l in open(a.data)][: a.limit]
rows = []
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    for d in data:
        body = json.dumps({"messages": [{"role": "user", "content": PROMPT.format(u=d["clue"])}],
                           "temperature": 0, "max_tokens": a.max_tokens}).encode()
        req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t = time.time()
        rep = json.loads(urllib.request.urlopen(req, timeout=600).read())
        lat = time.time() - t
        raw = rep["choices"][0]["message"]["content"]
        pred, bad = S.normalize(S.extract_json(raw))
        r = d | {"raw": raw, "pred": pred, "problems": bad, "lat": lat, "ok": S.compare(pred, d["gold"]),
                 "complexity": d.get("complexity") or S.complexity(d["gold"]),
                 "tokens_out": rep.get("usage", {}).get("completion_tokens")}
        rows.append(r)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        print(f"{d['id']} {'OK ' if r['ok']['all'] else 'ERR'} {lat:5.1f}s {r['tokens_out']} tok  {d['clue'][:36]}", flush=True)
s = summarize(rows)
Path(a.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
print(json.dumps({"latency_s": s["latency_s"], "all": s["all"]}, indent=1, ensure_ascii=False))
