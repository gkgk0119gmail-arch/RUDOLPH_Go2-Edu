"""Score the fine-tuned GGUF parser behind a llama-server (NX or PC) on labelled sentences; prints time and tokens.

    llama-server -m rudolph_parser_8b_v1_Q4_K_M.gguf -ngl 99 -c 1024 --port 8080      # on the NX
    python3 scripts/check_gguf_parser.py --url http://127.0.0.1:8080 --n 30
Python 3.8 safe (runs on the NX).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clue import schema as S  # noqa: E402
from clue.prompts import PROMPTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "results/clue/pilot_llm_v1.jsonl"))
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.data)][: a.n]
res = []
for d in rows:
    body = json.dumps({"messages": [{"role": "user", "content": PROMPTS["ft1"].format(u=d["clue"])}],
                       "temperature": 0, "max_tokens": 200}).encode()
    t = time.time()
    rep = json.loads(urllib.request.urlopen(urllib.request.Request(
        a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=120).read())
    dt = time.time() - t
    raw = rep["choices"][0]["message"]["content"]
    pred, _ = S.normalize(S.extract_json(raw))
    ok = S.compare(pred, d["gold"])
    res.append({"id": d["id"], "s": dt, "prompt_tok": rep["usage"]["prompt_tokens"],
                "out_tok": rep["usage"]["completion_tokens"], "all": ok["all"], "raw": raw})
    print("%.1fs %d+%d tok %s %s" % (dt, res[-1]["prompt_tok"], res[-1]["out_tok"], "OK " if ok["all"] else "ERR", raw))
n = len(res)
print("n %d  all-fields %.3f  mean %.2fs (min %.2f max %.2f)  out tokens %.1f" % (
    n, sum(r["all"] for r in res) / n, sum(r["s"] for r in res) / n, min(r["s"] for r in res),
    max(r["s"] for r in res), sum(r["out_tok"] for r in res) / n))
if a.out:
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
