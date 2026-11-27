"""Score a parser (base model + prompt, or base + LoRA adapter + ft1 prompt) on a labelled sentence file.

    python -m clue.ft_eval --data results/clue/pilot_llm_v1.jsonl --adapter OUT/lora_8b_v1 --out parse_ft1_pilot.jsonl
Same scoring as clue.parse (schema.compare, summarize) so numbers compare directly with parse_8b_v3f_*.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import schema as S
from .prompts import PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--adapter")
ap.add_argument("--prompt", default="ft1", choices=sorted(PROMPTS))
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int)
a = ap.parse_args()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402


def summarize(rows):
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by["all"].append(r)
        by[r["complexity"]].append(r)
    out = {}
    for k, rs in by.items():
        out[k] = {"n": len(rs), "json_ok": round(sum(r["pred"] is not None for r in rs) / len(rs), 3)}
        for f in (*S.KEYS, "all"):
            out[k][f] = round(sum(r["ok"][f] for r in rs) / len(rs), 3)
    out["latency_s"] = round(sum(r["lat"] for r in rows) / len(rows), 3)
    out["reply_chars"] = round(sum(len(r["raw"]) for r in rows) / len(rows), 1)
    return out


proc = AutoProcessor.from_pretrained(a.model)
proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()
data = [json.loads(l) for l in open(a.data)][: a.limit]
rows = []
with open(a.out, "w") as f:
    for i in range(0, len(data), a.batch):
        ch = data[i:i + a.batch]
        msgs = [[{"role": "user", "content": [{"type": "text", "text": PROMPTS[a.prompt].format(u=d["clue"])}]}] for d in ch]
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt", padding=True).to("cuda:0")
        t = time.time()
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=220, do_sample=False)
        lat = (time.time() - t) / len(ch)
        for d, raw in zip(ch, proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)):
            pred, bad = S.normalize(S.extract_json(raw))
            r = dict(d, raw=raw, pred=pred, problems=bad, lat=lat, ok=S.compare(pred, d["gold"]),
                     complexity=d.get("complexity") or S.complexity(d["gold"]))
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)}/{len(data)} all-fields {sum(r['ok']['all'] for r in rows) / len(rows):.3f}", flush=True)
s = summarize(rows)
Path(a.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
print(json.dumps(s["all"], ensure_ascii=False), "reply chars", s["reply_chars"])
