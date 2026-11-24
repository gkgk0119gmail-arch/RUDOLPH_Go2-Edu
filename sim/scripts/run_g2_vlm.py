"""Zero-shot Qwen3-VL-8B as the clue reader in the G2 world (closed loop, model latency added to the clock).

posture: no clue | rule: true meaning of the sentence, code geometry | parse: model reads the sentence only, code
geometry | score_txt: model scores every option from a table | score_img: same plus the map image.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "go2_lab" / "gap_search"))
import g2_sim as g  # noqa: E402
import utter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
a = ap.parse_args()
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()


def ask(text, image=None, n=200):
    c = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    x = proc.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt").to("cuda:0")
    t = time.time()
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=n, do_sample=False)
    return proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0], time.time() - t


def js(s):
    m = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


PARSE = """재난 현장 로봇이 생존자의 말을 들었습니다. 말한 사람을 기준으로 다른 요구조자가 있는 방향과 거리를 고르세요.
말: "{u}"
direction은 front, front-right, right, back-right, back, back-left, left, front-left 중 하나, distance는 near, far 중 하나입니다.
JSON만 출력: {{"direction": "...", "distance": "..."}}"""
SCORE = """재난 현장 수색 로봇입니다. 생존자 A의 말을 듣고, 다음에 갈 후보마다 다른 요구조자가 그쪽에 있을 가능성을 0(무관)~3(매우 높음)으로 매기세요.
지도 좌표: x는 오른쪽, y는 아래쪽(m). A 위치 ({ax:.1f}, {ay:.1f}), A가 바라보는 방향 벡터 ({fx:.2f}, {fy:.2f}). 로봇 위치 ({rx:.1f}, {ry:.1f}).
왼쪽/오른쪽/앞/뒤는 A가 바라보는 방향 기준이고, 가까움은 A에서 6 m 이내입니다.
A의 말: "{u}"
후보:
{opts}
{img}JSON만 출력: {{"scores": [{n}개의 정수, 후보 번호 순서]}}"""


class VRun(g.Run):
    def __init__(self, w, seed, mode):
        super().__init__(w, "posture" if mode == "posture" else "linked_nop", seed=seed)
        self.mode, self.calls, self.bad, self.lat, self.parsed_ok = mode, 0, 0, 0.0, None
        self.top1 = []
        self.u, self.facing, self.meaning = utter.make(seed, w.clue_dir, w.clue_far)
        self.clue_dir, self.clue_far = utter.to_map(self.meaning, self.facing)
        self.parsed = False

    def call(self, *k):
        s, dt = ask(*k)
        self.calls += 1
        self.lat += dt
        self.t += dt
        return js(s)

    def relevance(self, opts):
        truth = super().relevance(opts)
        if self.mode == "rule":
            return truth
        if self.mode == "parse":
            if not self.parsed:
                self.parsed = True
                r = self.call(PARSE.format(u=self.u))
                ok = isinstance(r, dict) and r.get("direction") in utter.DIRS and r.get("distance") in ("near", "far")
                self.bad += not ok
                self.parsed_ok = ok and {k: r[k] for k in self.meaning} == self.meaning
                if ok:
                    self.clue_dir, self.clue_far = utter.to_map(r, self.facing)
                else:
                    self.unread = True
            return np.ones(len(opts)) / len(opts) if getattr(self, "unread", False) else super().relevance(opts)
        C = g.CELL
        lines = "\n".join(f"{i}: {'낮은 틈' if o[0] == 'gap' else '미탐사 경계'} ({o[2][1] * C:.1f}, {o[2][0] * C:.1f})"
                          for i, o in enumerate(opts))
        img = self.draw(opts) if self.mode == "score_img" else None
        r = self.call(SCORE.format(ax=self.w.witness[1] * C, ay=self.w.witness[0] * C, fx=math.cos(self.facing),
                                   fy=math.sin(self.facing), rx=self.pos[1] * C, ry=self.pos[0] * C, u=self.u, opts=lines,
                                   img="이미지: 흰색=빈 곳, 검정=벽, 주황=낮은 틈, 회색=모름, 파랑=A와 시선 화살표, 초록=로봇, 숫자=후보.\n"
                                   if img else "", n=len(opts)), img, 60 + 4 * len(opts))
        s = r.get("scores") if isinstance(r, dict) else None
        if not (isinstance(s, list) and len(s) == len(opts) and all(isinstance(v, (int, float)) for v in s)):
            self.bad += 1
            return np.ones(len(opts)) / len(opts)
        s = np.clip(np.array(s, float), 0, 3)
        self.top1.append(bool(truth[int(s.argmax())] >= 0.5 * truth.max()))
        q = np.array([0.0, 0.1, 0.3, 1.0])[np.rint(s).astype(int)]
        return q / max(q.sum(), 1e-9)

    def draw(self, opts, k=8):
        col = np.array([[255, 255, 255], [0, 0, 0], [255, 140, 0], [150, 150, 150]], np.uint8)
        im = Image.fromarray(col[self.known]).resize((self.known.shape[1] * k, self.known.shape[0] * k), 0)
        d = ImageDraw.Draw(im)
        P = lambda c: ((c[1] + .5) * k, (c[0] + .5) * k)  # noqa: E731
        x, y = P(self.w.witness)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(0, 80, 255))
        d.line([x, y, x + 45 * math.cos(self.facing), y + 45 * math.sin(self.facing)], fill=(0, 80, 255), width=5)
        x, y = P(self.pos)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(0, 170, 0))
        for i, o in enumerate(opts):
            x, y = P(o[2])
            d.rectangle([x - 9, y - 8, x + 9, y + 8], fill=(255, 255, 0), outline=(200, 0, 0))
            d.text((x - 6, y - 6), str(i), fill=(200, 0, 0))
        return im


MODES = ("posture", "rule", "parse", "score_txt", "score_img")
out = {}
for size, rooms in (("3x4", (3, 4)), ("5x7", (5, 7))):
    rows = []
    for seed in range(a.start, a.start + a.n):
        w = g.make_world(seed, rooms)
        row = dict(seed=seed, behind_gap=w.behind_gap)
        for m in MODES:
            r = VRun(w, seed, m)
            row[m] = dict(r.run(), calls=r.calls, bad=r.bad, lat=r.lat, top1=r.top1, parsed_ok=r.parsed_ok)
            if m == "parse":
                row["utterance"], row["meaning"] = r.u, r.meaning
        rows.append(row)
        out[size] = rows
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    for m in MODES:
        T = np.array([r[m]["t"] for r in rows])
        c, b = sum(r[m]["calls"] for r in rows), sum(r[m]["bad"] for r in rows)
        top = sum((r[m]["top1"] for r in rows), [])
        po = [r[m]["parsed_ok"] for r in rows if r[m]["parsed_ok"] is not None]
        print(size, m, f"mean {T.mean():.1f}s found {np.mean([r[m]['found'] for r in rows]):.3f} calls {c} bad {b} "
              f"lat {sum(r[m]['lat'] for r in rows) / max(c, 1):.2f}s top1ok {np.mean(top) if top else float('nan'):.2f} "
              f"parse_ok {np.mean(po) if po else float('nan'):.2f} (n={len(po)})", flush=True)
