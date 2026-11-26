"""Synthetic Korean clue sentences with their gold JSON (schema.py).

Label first, sentence second (Hi Robot style): sample what the person means, then say it in one of many surface
forms. Written by us from templates -- good for building and checking the parser, NOT evidence about how real people
talk. An LLM paraphrase pass and human-written clues come later.

    python -m clue.gen --n 1000 --seed 0 --out results/clue/synth_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import schema as S

DIR_WORDS = {
    0: ["앞쪽", "정면", "바로 앞"], 45: ["오른쪽 앞", "앞에서 약간 오른쪽"], 90: ["오른쪽", "오른편"],
    135: ["오른쪽 뒤", "뒤에서 오른편"], 180: ["뒤쪽", "등 뒤"], 225: ["왼쪽 뒤", "뒤에서 왼편"],
    270: ["왼쪽", "왼편"], 315: ["왼쪽 앞", "앞에서 약간 왼쪽"],
}
ANCHOR_WORDS = {"reporter": ["제 ", "저 기준으로 ", "제가 서 있는 데서 "], "robot": ["네 ", "로봇 기준으로 ", "너 보는 방향에서 "]}
PLACE_KO = {"entrance": "입구", "lobby": "로비", "stairs": "계단", "elevator": "엘리베이터", "toilet": "화장실",
            "kitchen": "주방", "office": "사무실", "classroom": "강의실", "storage": "창고", "hall": "강당"}
OBJ_KO = {"desk": "책상", "bed": "침대", "table": "테이블", "window": "창문", "door": "문", "shelf": "선반", "sofa": "소파"}
COLOR_KO = {"red": "빨간", "blue": "파란", "yellow": "노란", "green": "초록", "black": "검은", "white": "흰", "gray": "회색"}
NUM_KO = {1: ["한 명", "1명"], 2: ["두 명", "2명", "둘"], 3: ["세 명", "3명", "셋"], 4: ["네 명", "4명"],
          5: ["다섯 명", "5명"]}
ORD_KO = {"1": "첫 번째", "2": "두 번째", "3": "세 번째", "4": "네 번째", "5": "다섯 번째", "last": "마지막"}
CORR_KO = {"inner": "안쪽 복도", "end": "복도 끝", "main": "큰 복도", "left": "왼쪽 복도", "right": "오른쪽 복도"}
NOISE = ["", "", "", "콜록… ", "저기요! ", "빨리요, ", "제발요. "]
TAIL = {"high": ["있어요.", "있습니다.", "갇혀 있어요.", "쓰러져 있어요."],
        "mid": ["있는 것 같아요.", "아마 있을 거예요.", "있었던 것 같은데요."],
        "low": ["있을지도 몰라요, 잘 모르겠어요.", "있을 수도 있어요, 확실하진 않아요.", "있는 것 같기도 해요."]}


def josa(word: str, a: str, b: str) -> str:
    """Korean particle by the last syllable: a after a final consonant (이/을/으로), b otherwise (가/를/로)."""
    ch = word.rstrip()[-1]
    if not "가" <= ch <= "힣":
        return word + b
    jong = (ord(ch) - 0xAC00) % 28
    if a == "으로" and jong == 8:  # ㄹ final takes 로
        return word + b
    return word + (a if jong else b)


def _pick(rng, a):
    return a[int(rng.integers(len(a)))]


def sample_label(rng) -> dict:
    """What the person means. At least one location element unless it is an appearance-only report."""
    c = S.empty()
    kinds = [k for k, p in (("dir", .5), ("dist", .4), ("sem", .5), ("floor", .2)) if rng.random() < p]
    if not kinds and rng.random() < .85:
        kinds = [_pick(rng, ["dir", "dist", "sem"])]
    if "dir" in kinds or "dist" in kinds:
        r = rng.random()
        c["anchor"] = "reporter" if r < .6 else "robot" if r < .8 else f"landmark:{_pick(rng, S.PLACES)}"
    if "dir" in kinds:
        c["direction_deg"] = int(_pick(rng, S.DIRS))
    if "dist" in kinds:
        if rng.random() < .6:
            c["distance_m"] = float(_pick(rng, [5, 10, 15, 20, 30, 40, 50]))
        else:
            c["distance_qual"] = _pick(rng, ["near", "far"])
    if "sem" in kinds:
        c["semantic"] = _sample_route(rng)
    if "floor" in kinds:
        if rng.random() < .6:
            c["floor"] = int(rng.integers(1, 6))
        else:
            c["floor_rel"] = int(_pick(rng, [1, -1]))
    r = rng.random()
    if r < .6:
        c["count"] = int(rng.integers(1, 6)) if rng.random() < .4 else int(rng.integers(1, 3))
    elif r < .75:
        c["count_qual"] = "several"
    if rng.random() < .35:
        ap = {}
        for k, vals, p in (("age", ("child", "adult", "elderly"), .6), ("sex", ("male", "female"), .4),
                           ("top_color", S.COLORS, .6)):
            if rng.random() < p:
                ap[k] = _pick(rng, vals)
        c["appearance"] = ap
    c["certainty"] = _pick(rng, ["high"] * 3 + ["mid"] * 2 + ["low"])
    return c


def _sample_route(rng) -> list[str]:
    r = rng.random()
    if r < .4:   # corridor -> turn -> nth room
        t = [f"corridor:{_pick(rng, ['inner', 'main', 'left', 'right'])}"] if rng.random() < .7 else []
        t += [f"turn:{_pick(rng, ['left', 'right'])}"] if rng.random() < .7 else []
        return t + [f"room:{_pick(rng, ['1', '2', '3', '4', 'last'])}"]
    if r < .6:   # pass a place, arrive at place
        return [f"place:{_pick(rng, S.PLACES)}", f"turn:{_pick(rng, ['left', 'right'])}", f"room:{_pick(rng, ['1', '2', '3'])}"]
    if r < .8:   # near / under an object or place
        k = _pick(rng, ["near", "under"])
        return [f"{k}:{_pick(rng, S.OBJECTS if k == 'under' else S.OBJECTS + S.PLACES)}"]
    return [f"place:{_pick(rng, S.PLACES)}"] + ([f"under:{_pick(rng, S.OBJECTS)}"] if rng.random() < .5 else [])


def _route_ko(rng, sem: list[str]) -> str:
    out = []
    for i, t in enumerate(sem):
        k, v = t.split(":")
        last = i == len(sem) - 1
        if k == "corridor":
            out.append(_pick(rng, [josa(CORR_KO[v], "으로", "로") + " 가서", CORR_KO[v] + " 따라가다가", josa(CORR_KO[v], "으로", "로") + " 쭉 가서"]) if not last else CORR_KO[v] + "에")
        elif k == "turn":
            out.append(("오른쪽" if v == "right" else "왼쪽") + _pick(rng, ["으로 꺾어서", "으로 돌면", "으로 들어가서"]))
        elif k == "room":
            out.append(f"{ORD_KO[v]} 방" + ("에" if last else "을 지나"))
        elif k == "place":
            out.append(PLACE_KO[v] + ("에" if last else _pick(rng, [" 지나서", " 쪽으로 가서", "에서"])))
        elif k == "near":
            out.append((OBJ_KO.get(v) or PLACE_KO[v]) + _pick(rng, [" 옆에", " 근처에"]))
        elif k == "under":
            out.append(OBJ_KO[v] + _pick(rng, [" 밑에", " 아래에"]))
    return " ".join(out)


def _who(rng, c) -> str:
    ap, cnt = c["appearance"], c["count"]
    head = ""
    if ap.get("top_color"):
        head += f"{COLOR_KO[ap['top_color']]} 옷 입은 "
    noun = {("child", None): "아이", ("child", "male"): "남자아이", ("child", "female"): "여자아이",
            ("elderly", None): "어르신", ("elderly", "male"): "할아버지", ("elderly", "female"): "할머니",
            ("adult", None): "어른", ("adult", "male"): "성인 남자", ("adult", "female"): "성인 여자",
            (None, "male"): "남자", (None, "female"): "여자"}.get((ap.get("age"), ap.get("sex")))
    if noun is None:
        noun = "사람" if head else ""
    if cnt is not None:
        n = _pick(rng, NUM_KO[cnt])
        return f"{head}{noun} {josa(n, '이', '가')} " if noun else f"사람 {josa(n, '이', '가')} "
    if c["count_qual"] == "several":
        return f"{head}{noun}들이 여러 명 " if noun else _pick(rng, ["사람들이 여러 명 ", "여러 명이 "])
    return f"{head}{josa(noun, '이', '가')} " if noun else _pick(rng, ["사람이 ", "누가 ", "다른 사람이 "])


def say(rng, c: dict) -> str:
    parts = []
    if c["floor"] is not None:
        parts.append(f"{c['floor']}층")
    elif c["floor_rel"] is not None:
        parts.append("위층" if c["floor_rel"] > 0 else "아래층")
    a = c["anchor"]
    loc = ""
    if a is not None:
        base = _pick(rng, ANCHOR_WORDS[a]) if a in ANCHOR_WORDS else PLACE_KO[a[9:]] + _pick(rng, ["에서 ", " 기준으로 "])
        if c["direction_deg"] is not None:
            loc = base + _pick(rng, DIR_WORDS[c["direction_deg"]])
        else:
            loc = base.rstrip() + ("에서" if a in ANCHOR_WORDS and base.endswith("제 ") else "")
            loc = loc.replace("제에서", "여기서").replace("네에서", "너 있는 데서")
        if c["distance_m"] is not None:
            loc += _pick(rng, [f" {c['distance_m']:.0f}미터쯤", f" 한 {c['distance_m']:.0f} m", f" {c['distance_m']:.0f}미터 정도 떨어진 곳"])
        elif c["distance_qual"]:
            loc += _pick(rng, [" 가까운 데", " 바로 근처"] if c["distance_qual"] == "near" else [" 멀리", " 한참 떨어진 데", " 저 안쪽 멀리"])
        parts.append(loc + "에")
    if c["semantic"]:
        parts.append(_route_ko(rng, c["semantic"]))
    who = _who(rng, c)
    where = " ".join(p for p in parts if p)
    body = f"{where} {who}" if rng.random() < .6 else f"{who}{where} "
    if not where:
        body = who + _pick(rng, ["어딘가에 ", "건물 안에 ", ""])
    return (_pick(rng, NOISE) + " ".join(body.split()) + " " + _pick(rng, TAIL[c["certainty"]])).strip()


def make(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        c = sample_label(rng)
        rows.append({"id": f"s{seed}_{i:04d}", "clue": say(rng, c), "gold": c, "source": "template_v1"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = make(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} clues -> {a.out}")
