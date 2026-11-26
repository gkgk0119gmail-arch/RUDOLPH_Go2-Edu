"""Sentence -> structured clue with a local VLM (text only). The model only translates; probability maps are code.

    python -m clue.parse --data results/clue/pilot_llm_v1.jsonl --model Qwen/Qwen3-VL-8B-Instruct \
        --device cuda:0 --out results/clue/parse_8b_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from . import schema as S

PROMPT = """재난 현장 수색 로봇이 사람의 말을 들었습니다. 이 말을 아래 JSON으로 바꾸세요. 말에 없는 내용은 null, 빈 리스트, 빈 객체로 두고 추측하지 마세요.

필드
- utterance: "clue"(다른 요구조자의 위치·인원·인상착의를 말함, "한 명 더 있어요"도 포함) | "help"(말한 사람 본인의 구조 요청·상태만) | "other"(요구조자와 무관). 구조 요청과 다른 사람 정보가 같이 있으면 "clue".
- anchor: 방향·거리의 기준. "reporter"(말한 사람: 제, 저, 내, 여기서) | "robot"(듣는 로봇: 너, 네, 로봇) | "landmark:<장소>"(입구에서, 계단에서) | null(방향·거리 없음)
- direction_deg: 기준의 정면에서 시계방향 각도 0,45,90,135,180,225,270,315 (앞 0, 오른쪽 앞 45, 오른쪽 90, 오른쪽 뒤 135, 뒤 180, 왼쪽 뒤 225, 왼쪽 270, 왼쪽 앞 315)
- distance_m: 말한 미터 숫자. distance_qual: 숫자 없이 "near"(가까운, 근처) | "far"(멀리, 한참)
- floor: 층 번호(3층 -> 3). floor_rel: 위층 1, 아래층 -1
- semantic: 경로·장소 토큰을 말한 순서대로. 허용 토큰:
  corridor:inner|end|main|left|right (안쪽/끝/큰/왼쪽/오른쪽 복도)
  turn:left|right (왼쪽으로 꺾다, 좌회전 / 오른쪽, 우회전)
  room:1..5|last|next (n번째 방, 마지막·맨 끝 방, 옆방)
  place:<장소>, near:<물체 또는 장소>(옆, 근처), under:<물체>(밑, 아래)
  장소: entrance 입구, lobby 로비, stairs 계단, elevator 엘리베이터, toilet 화장실, kitchen 주방, office 사무실, classroom 강의실, storage 창고, hall 강당
  물체: desk 책상, bed 침대, table 테이블, window 창문, door 문, shelf 선반, sofa 소파
- count: 인원 숫자. count_qual: 숫자 없이 여러 명이면 "several"
- appearance: {{"age": "child"|"adult"|"elderly", "sex": "male"|"female", "top_color": "red"|"blue"|"yellow"|"green"|"black"|"white"|"gray"}} 중 말한 것만
- certainty: 말한 사람의 확신 "high"(단정) | "mid"(아마, ~것 같아요) | "low"(잘 모르겠어요, 확실하지 않아요)

예시
말: "제 왼쪽으로 10미터 가면 화장실 있는데 그 옆방에 두 명 있을 거예요."
{{"utterance": "clue", "anchor": "reporter", "direction_deg": 270, "distance_m": 10, "distance_qual": null, "floor": null, "floor_rel": null, "semantic": ["place:toilet", "room:next"], "count": 2, "count_qual": null, "appearance": {{}}, "certainty": "mid"}}
말: "아이고 허리야… 좀 꺼내 주세요."
{{"utterance": "help", "anchor": null, "direction_deg": null, "distance_m": null, "distance_qual": null, "floor": null, "floor_rel": null, "semantic": [], "count": null, "count_qual": null, "appearance": {{}}, "certainty": "high"}}

말: "{u}"
JSON만 출력:"""


class Parser:
    def __init__(self, model: str, device: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map=device).eval()

    def __call__(self, text: str, max_new_tokens: int = 220) -> tuple[str, float]:
        msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(u=text)}]}]
        x = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt").to(self.device)
        t = time.time()
        with self.torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0], time.time() - t


def summarize(rows: list[dict]) -> dict:
    """Field accuracy overall and by complexity; utterance-type confusion."""
    by = defaultdict(list)
    for r in rows:
        by["all"].append(r)
        by[r.get("complexity", "?")].append(r)
    out = {}
    for k, rs in by.items():
        out[k] = {"n": len(rs), "json_ok": sum(r["pred"] is not None for r in rs) / len(rs)}
        for f in (*S.KEYS, "all"):
            out[k][f] = round(sum(r["ok"][f] for r in rs) / len(rs), 3)
    conf = defaultdict(int)
    for r in rows:
        conf[f"{r['gold']['utterance']}->{(r['pred'] or {}).get('utterance')}"] += 1
    out["utterance_confusion"] = dict(conf)
    out["latency_s"] = round(sum(r["lat"] for r in rows) / len(rows), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    data = [json.loads(l) for l in open(a.data)][: a.limit]
    p = Parser(a.model, a.device)
    rows = []
    with open(a.out, "w") as f:
        for d in data:
            raw, lat = p(d["clue"])
            pred, bad = S.normalize(S.extract_json(raw))
            r = d | {"raw": raw, "pred": pred, "problems": bad, "lat": lat, "ok": S.compare(pred, d["gold"]),
                     "complexity": d.get("complexity") or S.complexity(d["gold"])}
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{d['id']} {'OK ' if r['ok']['all'] else 'ERR'} {lat:.1f}s {d['clue'][:40]}", flush=True)
    s = summarize(rows)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps(s, indent=1, ensure_ascii=False))
