"""COMRES-VLM-style baseline: the VLM itself picks the next goal from a table of candidates (no belief map).

Contrast with our pipeline, where the VLM only translates the sentence and code keeps the probability map.
Each decision: up to K candidates (nearest first by travel time), each described by position relative to the robot
and the corridor, travel time, unsearched area behind it, and whether it lies inside an identified room (side, n).

    Search(b, beta=1.0, score="vlm", vlm=VLMPlanner(model, device, sentences))
"""
from __future__ import annotations

import time

from .parse import Parser
from .schema import extract_json

PROMPT = """재난 현장에서 요구조자를 찾는 사족보행 로봇입니다. 다음에 갈 곳을 하나 고르세요.
건물: 서쪽 끝 로비, 동쪽으로 긴 복도, 복도 양쪽에 방들. "왼쪽"/"오른쪽"은 로비에서 동쪽(복도 안쪽)을 보고 걸을 때 기준.
남은 요구조자 {left}명 (전체 {n}명). 들은 말:
{clues}
후보 (번호, 위치, 걸리는 시간, 그 뒤 아직 안 본 넓이):
{table}
요구조자를 가장 빨리 찾을 후보 하나를 고르세요. JSON만 출력: {{"pick": 번호}}"""


class VLMPlanner:
    def __init__(self, model: str, device: str, k: int = 12):
        self.p = Parser(model, device)
        self.k, self.calls, self.lat, self.bad = k, 0, 0.0, 0

    def __call__(self, search, tg, G, C) -> int:
        b = search.b
        # one candidate per region (a room, or a 4 m corridor stretch): the one with most unsearched area behind it,
        # so far rooms the clues name are on the table, not twelve cells of the nearest room
        best = {}
        for i, (y, x) in enumerate(tg):
            ri = search.room_of((y, x))
            key = ("room", ri) if ri is not None else ("corr", int((x - b.corr_x0) * 0.2 // 4))
            if G[i] > 0 and (key not in best or G[i] > G[best[key]]):
                best[key] = i
        cand = list(best.values()) or list(range(len(tg)))
        order = sorted(cand, key=lambda i: C[i])[: self.k]
        rows = []
        for j, i in enumerate(order):
            y, x = tg[i]
            ri = search.room_of((y, x))
            where = "복도/로비"
            if ri is not None:
                r = b.rooms[ri]
                where = f"{'왼쪽' if r['side'] == 'left' else '오른쪽'} {r['ordinal']}번째 방 (문 확인됨)" if all(
                    search.known[q] == 0 for q in r["door"]) else "방 안 (몇 번째인지 모름)"
            dx = (x - b.corr_x0) * 0.2
            rows.append(f"{j}: {where}, 로비 끝에서 동쪽 {dx:.0f} m, {C[i]:.0f}초, 넓이 {G[i] * 0.04:.0f} m²")
        said = [c.get("text") or str({k: v for k, v in c.items() if k in ("semantic", "direction_deg", "distance_m")})
                for k, c in enumerate(search.clues) if search.active.get(k)]
        text = PROMPT.format(left=sum(f is None for f in search.found), n=len(search.found),
                             clues="\n".join(f"- {s}" for s in said) or "- (없음)", table="\n".join(rows))
        t = time.time()
        msgs = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        x = self.p.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt").to(self.p.device)
        with self.p.torch.no_grad():
            y = self.p.model.generate(**x, max_new_tokens=20, do_sample=False)
        raw = self.p.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.calls += 1
        self.lat += time.time() - t
        d = extract_json(raw)
        try:
            j = int(d["pick"])
            return order[j]
        except (TypeError, KeyError, ValueError, IndexError):
            self.bad += 1
            return order[0]
