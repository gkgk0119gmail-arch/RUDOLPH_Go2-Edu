"""Training sentences for fine-tuning the clue parser (LoRA): label first, then one of many surface forms.

Wider than gen.py: help / other utterances, mobility, side-of-corridor rooms, 옆방, Korean number words, honorific
and casual speech, fillers, broken word order, help call + clue. Written from the schema rules (schema.py, prompt v3f)
by us, NOT from the test sentences (pilot_llm_v1, c_sentences); exact and near duplicates of those are removed.
Same caveat as gen.py: patterns, not recorded human speech.

    python -m clue.gen_train --n 8000 --seed 1 --out results/clue/train_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import schema as S

R = None   # np.random.Generator, set in make()


def pick(a):
    return a[int(R.integers(len(a)))]


def p(x):
    return R.random() < x


def jong(w: str) -> bool:
    ch = w.rstrip()[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def ro(w):
    ch = w.rstrip()[-1]
    if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 not in (0, 8):
        return w + "으로"
    return w + "로"


def ga(w):
    return w + ("이" if jong(w) else "가")


# ---------------------------------------------------------------- words
DIRW = {0: ["앞", "앞쪽", "정면", "바로 앞"], 45: ["오른쪽 앞", "오른쪽 대각선", "앞 오른쪽", "우측 앞"],
        90: ["오른쪽", "오른편", "우측"], 135: ["오른쪽 뒤", "뒤 오른쪽", "우측 뒤", "오른쪽 뒤편"],
        180: ["뒤", "뒤쪽", "등 뒤", "뒤편"], 225: ["왼쪽 뒤", "뒤 왼쪽", "좌측 뒤", "왼쪽 뒤편"],
        270: ["왼쪽", "왼편", "좌측"], 315: ["왼쪽 앞", "왼쪽 대각선", "앞 왼쪽", "좌측 앞"]}
PLACE = {"entrance": ["입구", "출입구", "현관"], "lobby": ["로비"], "stairs": ["계단", "계단실"],
         "elevator": ["엘리베이터", "엘베"], "toilet": ["화장실"], "kitchen": ["주방", "부엌", "탕비실"],
         "office": ["사무실"], "classroom": ["강의실", "교실"], "storage": ["창고", "창고방"], "hall": ["강당", "홀"]}
OBJ = {"desk": ["책상"], "bed": ["침대"], "table": ["테이블", "탁자"], "window": ["창문", "창가"], "door": ["문", "문 앞"],
       "shelf": ["선반", "책장"], "sofa": ["소파"]}
COLOR = {"red": ["빨간", "빨간색", "붉은"], "blue": ["파란", "파란색", "남색"], "yellow": ["노란", "노란색"],
         "green": ["초록", "초록색", "녹색"], "black": ["검은", "검정", "까만"], "white": ["흰", "하얀", "흰색"],
         "gray": ["회색", "회색빛"]}
CLOTH = ["옷", "옷", "티", "점퍼", "잠바", "티셔츠", "상의", "후드티"]
CORR = {"inner": ["안쪽 복도", "안쪽 통로"], "end": ["복도 끝", "복도 맨 끝"], "main": ["큰 복도", "중앙 복도", "메인 복도"],
        "left": ["왼쪽 복도"], "right": ["오른쪽 복도"]}
SIDEW = {"left": ["왼쪽", "왼편", "좌측"], "right": ["오른쪽", "오른편", "우측"]}
ORD = {"1": ["첫 번째", "첫번째", "1번째", "맨 처음"], "2": ["두 번째", "두번째", "2번째"], "3": ["세 번째", "세번째", "3번째"],
       "4": ["네 번째", "네번째", "4번째"], "5": ["다섯 번째", "5번째"], "6": ["여섯 번째", "6번째"],
       "7": ["일곱 번째", "7번째"], "8": ["여덟 번째", "8번째"], "9": ["아홉 번째", "9번째"],
       "last": ["마지막", "맨 끝", "제일 끝"]}
SINO = {1: "일", 2: "이", 3: "삼", 4: "사", 5: "오", 6: "육", 7: "칠", 8: "팔", 9: "구"}
CNT = {1: ["한 명", "1명", "한 사람", "하나", "한 분"], 2: ["두 명", "2명", "둘", "두 사람", "두 분"],
       3: ["세 명", "3명", "셋", "세 사람", "세 분"], 4: ["네 명", "4명", "넷", "네 사람"], 5: ["다섯 명", "5명", "다섯"],
       6: ["여섯 명", "6명"], 7: ["일곱 명", "7명"]}
SEVERAL = ["여러 명", "여럿", "몇 명", "사람들 여럿", "여러 사람", "꽤 많이", "많이"]
NOUN = {("child", None): ["아이", "애", "꼬마", "애기"], ("child", "male"): ["남자아이", "남자애", "남자 꼬마"],
        ("child", "female"): ["여자아이", "여자애", "여자 꼬마"], ("elderly", None): ["어르신", "노인", "노인분"],
        ("elderly", "male"): ["할아버지", "할아버님"], ("elderly", "female"): ["할머니", "할머님"],
        ("adult", None): ["어른", "성인"], ("adult", "male"): ["아저씨", "성인 남자", "남자 어른"],
        ("adult", "female"): ["아주머니", "아줌마", "성인 여자"], (None, "male"): ["남자", "남자분", "남성"],
        (None, "female"): ["여자", "여자분", "여성"]}
STATIC = ["갇혀 있어요", "깔려 있어요", "쓰러져 있어요", "못 움직여요", "다리가 끼어서 못 나와요", "기절해 있어요",
          "누워서 안 움직여요", "잔해에 깔렸어요"]
MOBILE = ["돌아다니고 있어요", "움직이고 있어요", "나가려고 했어요", "출구 찾고 있었어요", "걸어 다니고 있었어요",
          "이리저리 헤매고 있어요"]
TAIL = {"high": ["있어요", "있습니다", "있어", "계세요", "있어요, 확실해요", "있어요, 제가 봤어요", "있다니까요"],
        "mid": ["있을 거예요", "있는 것 같아요", "아마 있을 거예요", "있었던 것 같아요", "있을 걸요", "있는 듯해요", "있지 싶어요"],
        "low": ["있을지도 몰라요", "있을 수도 있어요, 확실하진 않아요", "있는지 잘 모르겠어요", "있었나… 모르겠어요",
                "있을 수도 있고 없을 수도 있어요", "있는 것 같기도 한데 잘 모르겠어요"]}
VOICE_TAIL = {"high": ["목소리가 들려요", "소리가 나요", "신음 소리가 들려요"],
              "mid": ["목소리가 들린 것 같아요", "소리가 났던 것 같아요"],
              "low": ["소리가 들린 것 같기도 한데 잘 모르겠어요", "무슨 소리가 난 것 같기도 해요"]}
PRE = ["", "", "", "", "저기요, ", "저기… ", "콜록… ", "빨리요! ", "제발요, ", "아 그게, ", "음… ", "로봇아, ", "여기요! ",
       "아까 봤는데요, ", "잘 들어요, ", "헉헉… "]
HELP_PRE = ["살려주세요! ", "도와주세요, ", "여기요 살려주세요, "]
HELP = ["살려주세요!", "도와주세요…", "여기요, 여기 사람 있어요! 꺼내 주세요!", "다리가 너무 아파요.", "숨쉬기가 힘들어요…",
        "저 좀 꺼내 주세요.", "팔이 끼어서 못 움직이겠어요.", "아파요… 빨리 와 주세요.", "여기 갇혔어요, 문이 안 열려요.",
        "물 좀 주세요…", "저 여기 있어요! 들려요?", "피가 나요, 도와주세요.", "머리가 어지러워요.", "제발 나가게 해 주세요.",
        "허리를 다쳤어요, 못 일어나겠어요.", "숨이 막혀요, 살려주세요.", "저 혼자예요, 빨리 와 주세요.", "발목이 부러진 것 같아요."]
OTHER = ["연기가 너무 많아요.", "여기 어디예요?", "불이 났어요, 119 불렀어요?", "전기가 다 나갔어요.", "가스 냄새가 나요.",
         "밖에 비 와요?", "로봇이네, 신기하다.", "몇 시예요?", "건물 무너지는 거 아니에요?", "물이 새고 있어요.",
         "계단이 막혔어요.", "소방관은 언제 와요?", "너 말할 줄 알아?", "여기 너무 어두워요.", "천장이 내려앉았어요.",
         "엘리베이터는 멈췄어요.", "휴대폰이 안 터져요.", "창문이 다 깨졌어요."]


def num_m(m: float) -> str:
    m = int(m)
    r = R.random()
    if r < .55:
        return f"{m}미터" if p(.7) else f"{m} m"
    if m < 10:
        return SINO[m] + " 미터" if p(.5) else SINO[m] + "미터"
    t, o = divmod(m, 10)
    return (SINO[t] if t > 1 else "") + "십" + (SINO[o] if o else "") + "미터"


# ---------------------------------------------------------------- labels
def sample_route():
    r = R.random()
    if r < .25:      # corridor -> turn -> nth room
        t = [f"corridor:{pick(list(S.SEM_KINDS['corridor']))}"] if p(.6) else []
        t += [f"turn:{pick(['left', 'right'])}"] if p(.7) else []
        return t + [f"room:{pick(list(ORD))}"]
    if r < .45:      # side of the corridor, nth room (no turn)
        t = [f"corridor:{pick(['inner', 'main', 'left', 'right'])}"] if p(.4) else []
        return t + [f"side:{pick(['left', 'right'])}", f"room:{pick(list(ORD))}"]
    if r < .58:      # pass a place, turn, nth room
        return [f"place:{pick(S.PLACES)}"] + ([f"turn:{pick(['left', 'right'])}"] if p(.7) else []) + \
            [f"room:{pick(['1', '2', '3', '4', 'last'])}"]
    if r < .66:      # place, next room
        return [f"place:{pick(S.PLACES)}", "room:next"] if p(.7) else ["room:next"]
    if r < .82:      # near / under
        k = pick(["near", "under"])
        t = [f"{k}:{pick(S.OBJECTS if k == 'under' else S.OBJECTS + S.PLACES)}"]
        return ([f"place:{pick(S.PLACES)}"] + t) if p(.3) else t
    t = [f"place:{pick(S.PLACES)}"]
    return t + ([f"under:{pick(S.OBJECTS)}"] if p(.3) else [])


def sample_label():
    c = S.empty()
    u = R.random()
    if u < .08:
        c["utterance"] = "help"
        return c
    if u < .15:
        c["utterance"] = "other"
        return c
    kinds = [k for k, q in (("dir", .45), ("dist", .4), ("sem", .5), ("floor", .15)) if p(q)]
    if not kinds and p(.8):
        kinds = [pick(["dir", "dist", "sem"])]
    if "dir" in kinds or "dist" in kinds:
        r = R.random()
        c["anchor"] = "reporter" if r < .5 else "robot" if r < .8 else f"landmark:{pick(S.PLACES)}"
    if "dir" in kinds:
        c["direction_deg"] = int(pick(S.DIRS))
    if "dist" in kinds:
        if p(.6):
            c["distance_m"] = float(pick([2, 3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]))
        else:
            c["distance_qual"] = pick(["near", "far"])
    if "sem" in kinds:
        c["semantic"] = sample_route()
    if "floor" in kinds:
        if p(.6):
            c["floor"] = int(R.integers(1, 8))
        else:
            c["floor_rel"] = int(pick([1, -1]))
    r = R.random()
    if r < .6:
        c["count"] = int(R.integers(1, 8)) if p(.3) else int(pick([1, 1, 2, 2, 3]))
    elif r < .72:
        c["count_qual"] = "several"
    if p(.35):
        ap = {}
        for k, vals, q in (("age", ("child", "adult", "elderly"), .6), ("sex", ("male", "female"), .45),
                           ("top_color", S.COLORS, .55)):
            if p(q):
                ap[k] = pick(vals)
        c["appearance"] = ap
    c["mobility"] = pick(["static", "mobile"]) if p(.25) else None
    c["certainty"] = pick(["high"] * 3 + ["mid"] * 2 + ["low"])
    return c


# ---------------------------------------------------------------- surface
def anchor_phrase(c):
    a, d = c["anchor"], c["direction_deg"]
    if a is None:
        return ""
    if a == "reporter":
        base = pick(["제 ", "저 기준으로 ", "제가 있는 데서 ", "여기서 ", "제 위치에서 ", "내 "]) if d is not None else \
            pick(["저 기준으로 ", "제가 있는 데서 ", "여기서 ", "제 위치에서 ", "저한테서 "])
    elif a == "robot":
        base = pick(["네 ", "너 기준으로 ", "로봇 기준으로 ", "로봇 ", "너한테서 ", "너 있는 데서 ", "로봇 너 "]) if d is not None \
            else pick(["너 기준으로 ", "로봇 기준으로 ", "너한테서 ", "너 있는 데서 ", "로봇 너한테서 "])
    else:
        w = pick(PLACE[a[9:]])
        base = w + pick(["에서 ", " 기준으로 ", "에서부터 ", " 쪽에서 "])
    s = base
    if d is not None:
        dw = pick(DIRW[d])
        if c["distance_m"] is None and c["distance_qual"] is None:
            s += dw + pick(["에", "에요", "쪽에", "으로 가면" if jong(dw) else "로 가면", ""])
        else:
            s += ro(dw) + " " if p(.6) else dw + " "
    if c["distance_m"] is not None:
        s += pick(["", "한 ", "대략 "]) + num_m(c["distance_m"]) + pick(["쯤", " 정도", "", " 가면", " 떨어진 데", " 거리에"])
    elif c["distance_qual"] == "near":
        s += pick(["가까운 데", "바로 근처", "가까이에", "코앞에", "얼마 안 떨어진 데"])
    elif c["distance_qual"] == "far":
        s += pick(["멀리", "한참 떨어진 데", "저 멀리", "한참 가면", "꽤 멀리"])
    return s.strip()


def route_phrase(sem):
    out = []
    for i, t in enumerate(sem):
        k, v = t.split(":")
        last = i == len(sem) - 1
        if k == "corridor":
            w = pick(CORR[v])
            out.append(w + "에" if last else pick([ro(w) + " 가서", w + " 따라가다가", ro(w) + " 쭉 가서", w + " 들어가서"]))
        elif k == "turn":
            w = pick(SIDEW[v])
            out.append(pick([ro(w) + " 꺾어서", ro(w) + " 돌면", ro(w) + " 꺾으면", "좌회전해서" if v == "left" else "우회전해서"]))
        elif k == "side":
            w = pick(SIDEW[v])
            out.append(pick([f"복도 {w}", f"{w}", f"{w}에 있는 방들 중에", f"{w} 방들 중"]))
        elif k == "room":
            if v == "next":
                out.append(pick(["옆방에", "바로 옆방에", "옆 방에"]) if last else "옆방 지나서")
            else:
                out.append(f"{pick(ORD[v])} 방" + (pick(["에", "에서", "이요,"]) if last else "을 지나"))
        elif k == "place":
            w = pick(PLACE[v])
            out.append(w + pick(["에", " 안에", "에요,"]) if last else w + pick([" 지나서", " 쪽으로 가서", "에서", " 나오면"]))
        elif k == "near":
            w = pick(OBJ.get(v) or PLACE[v])
            out.append(w + pick([" 옆에", " 근처에", " 옆쪽에", " 주변에"]))
        elif k == "under":
            w = pick(OBJ[v])
            out.append(w + pick([" 밑에", " 아래에", " 아래쪽에"]))
    return " ".join(out)


def who_phrase(c):
    ap = c["appearance"]
    cloth = ""
    if ap.get("top_color"):
        cloth = f"{pick(COLOR[ap['top_color']])} {pick(CLOTH)}" + pick([" 입은 ", " 입은 ", "에 "]) if p(.8) else \
            f"{pick(COLOR[ap['top_color']])} {pick(CLOTH)} "
    key = (ap.get("age"), ap.get("sex"))
    noun = pick(NOUN[key]) if key in NOUN else ""
    plural = (c["count"] or 0) > 1 or c["count_qual"] == "several"
    if noun and plural and p(.4):
        noun += "들"
    head = (cloth + noun).strip() or (pick(["사람", "사람"]) if cloth or c["count"] is None else "")
    if c["count"] is not None:
        n = pick(CNT[c["count"]])
        return f"{head} {ga(n)}" if head and p(.7) else (f"{head} {n}" if head else ga(n) if p(.6) else f"사람 {ga(n)}")
    if c["count_qual"] == "several":
        n = pick(SEVERAL)
        return f"{head or '사람들'}{'이' if jong(head or '사람들') else '가'} {n}" if p(.6) else f"{n} {head or '사람'}"
    if head:
        return ga(head) if p(.7) else head
    return pick(["사람이", "누가", "누군가", "다른 사람이", "한 명 더", "사람"])


def floor_phrase(c):
    if c["floor"] is not None:
        f = c["floor"]
        return pick([f"{f}층", f"{f}층에서", f"{f}층, {f}층이요,"])
    if c["floor_rel"] is not None:
        return pick(["위층", "윗층", "위층에", "한 층 위"]) if c["floor_rel"] > 0 else pick(["아래층", "아랫층", "한 층 밑", "밑에 층"])
    return ""


def tail_phrase(c, honor):
    if c["mobility"] == "static":
        t = pick(STATIC)
        if c["certainty"] == "mid":
            t = t[:-3] + pick(["있을 거예요", "있는 것 같아요"]) if t.endswith("있어요") else t + pick([" 아마", ", 그런 것 같아요"])
        elif c["certainty"] == "low":
            t += pick([", 확실하진 않아요", "… 잘 모르겠어요", ", 아닐 수도 있어요"])
        return t
    if c["mobility"] == "mobile":
        t = pick(MOBILE)
        if c["certainty"] == "mid":
            t += pick([" 아마", ", 그런 것 같아요"])
        elif c["certainty"] == "low":
            t += pick([", 확실하진 않아요", "… 잘 모르겠어요"])
        return t
    t = pick(TAIL[c["certainty"]])
    if not honor:
        t = t.replace("있어요", "있어").replace("있을 거예요", "있을 거야").replace("같아요", "같아").replace("몰라요", "몰라")
    return t


def say_clue(c):
    honor = p(.75)
    where = [x for x in (floor_phrase(c), anchor_phrase(c), route_phrase(c["semantic"])) if x]
    who = who_phrase(c)
    voice = not c["mobility"] and c["count"] is None and c["count_qual"] is None and p(.25)
    tail = pick(VOICE_TAIL[c["certainty"]]) if voice else tail_phrase(c, honor)
    if voice:
        key = (c["appearance"].get("age"), c["appearance"].get("sex"))
        who = (pick(NOUN[key]) if key in NOUN else "사람") + " "
    loc = " ".join(where)
    r = R.random()
    if not loc:
        body = f"{who} {pick(['', '', '더 ', '건물 안에 ', '어딘가에 '])}{tail}"
    elif r < .55:
        body = f"{loc}{pick([' ', ', ', ' 거기 ', ', 거기에 '])}{who} {tail}"
    elif r < .8:
        body = f"{who} {loc} {tail}"
    else:   # spoken order: location first as its own sentence
        body = f"{loc}{pick(['요.', '!', '… ', '요, '])} {pick(['거기 ', '', '거기에 '])}{who} {tail}"
    pre = pick(HELP_PRE) if p(.08) else pick(PRE)
    s = pre + body + pick([".", ".", "!", "…", "", " 빨리요.", " 제발요."])
    s = re.sub(r",+", ",", s.replace(" ,", ",")).replace(",.", ".").replace("요요", "요")
    return " ".join(s.split())


def say(c):
    if c["utterance"] == "help":
        return pick(HELP)
    if c["utterance"] == "other":
        return pick(OTHER)
    return say_clue(c)


def compact(gold: dict) -> dict:
    """Training target: only the fields that differ from the empty clue (normalize() fills the rest back in)."""
    e = S.empty()
    return {k: v for k, v in gold.items() if v != e[k]}


def _key(t: str) -> str:
    return re.sub(r"[\s.,!?…~]", "", t)


def make(n: int, seed: int, exclude: set[str]) -> list[dict]:
    global R
    R = np.random.default_rng(seed)
    rows, seen = [], set()
    while len(rows) < n:
        c = sample_label()
        s = say(c)
        k = _key(s)
        if k in exclude or (k in seen and c["utterance"] == "clue"):
            continue
        seen.add(k)
        g, bad = S.normalize(c)
        assert not bad, (bad, c)
        rows.append({"id": f"t{seed}_{len(rows):05d}", "clue": s, "gold": g, "target": compact(g),
                     "source": "template_train_v1"})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=["results/clue/pilot_llm_v1.jsonl", "results/clue/c_sentences.jsonl",
                                                     "results/clue/c_sentences_s60.jsonl"])
    a = ap.parse_args()
    ex = {_key(json.loads(l)["clue"]) for f in a.exclude if Path(f).exists() for l in open(f)}
    rows = make(a.n, a.seed, ex)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} sentences -> {a.out} (excluded {len(ex)} test sentences)")
    for r in rows[:25]:
        print(r["clue"], "|", json.dumps(r["target"], ensure_ascii=False))
