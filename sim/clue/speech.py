"""Condition C glue: a simulator clue -> a Korean sentence, and a parsed JSON -> a simulator clue.

The phrasings below were written by an LLM (Claude) as slot patterns, so they vary in wording, fillers and order but
are still patterns: C measures what parsing costs on such sentences, not on recorded human speech.
"""
from __future__ import annotations

import numpy as np

SIDE = {"left": ["왼쪽", "왼편", "좌측"], "right": ["오른쪽", "오른편", "우측"]}
ORD = {1: ["첫 번째", "첫번째", "맨 앞"], 2: ["두 번째", "두번째", "2번째"], 3: ["세 번째", "세번째", "3번째"],
       4: ["네 번째", "네번째", "4번째"], 5: ["다섯 번째", "5번째"], 6: ["여섯 번째", "6번째"], 7: ["일곱 번째", "7번째"]}
SEM = [
    "복도로 쭉 가서 {s} {o} 방에 한 명 있어요.",
    "{s} {o} 방이요! 거기 사람 있어요!",
    "복도 따라가다 보면 {s}에 방들 있잖아요, {o} 방에 한 명 갇혀 있어요.",
    "콜록… 복도… {s} {o} 방에… 한 명…",
    "{s}으로 {o} 방, 한 명 쓰러져 있어요, 빨리요.",
    "저기 복도 들어가서 {s} 방들 중에 {o} 방이요, 거기 한 사람 있어요.",
    "어… 그러니까 복도에서 {s}, {o} 방이었어요. 한 명이요.",
    "복도 쭉 가면 {s}에 {o} 방, 거기서 살려달라는 소리 났어요.",
]
DIRW = {0: ["앞쪽", "정면"], 45: ["오른쪽 앞", "오른쪽 대각선"], 90: ["오른쪽", "오른편"], 135: ["오른쪽 뒤", "뒤 오른편"],
        180: ["뒤쪽", "등 뒤"], 225: ["왼쪽 뒤", "뒤 왼편"], 270: ["왼쪽", "왼편"], 315: ["왼쪽 앞", "왼쪽 대각선"]}
GEO = [
    "제 {d}으로 {m}미터쯤 가면 한 명 있어요.",
    "여기서 {d} {m}미터 정도요, 거기 한 명 있어요.",
    "{d}이요, {d}! 한 {m}미터? 한 명 있어요.",
    "콜록… 제 {d}… {m}미터쯤에… 한 사람…",
    "저 기준으로 {d}으로 {m} m 떨어진 데 사람 있어요.",
    "음… 제가 보는 데서 {d}, {m}미터쯤이었던 것 같은데, 한 명이요.",
]


def josa_ro(word: str) -> str:
    ch = word[-1]
    if "가" <= ch <= "힣":
        j = (ord(ch) - 0xAC00) % 28
        return word + ("로" if j in (0, 8) else "으로")
    return word + "로"


def sentence(clue: dict, rng: np.random.Generator) -> str:
    pick = lambda a: a[int(rng.integers(len(a)))]  # noqa: E731
    if clue.get("semantic"):
        side = next(t.split(":")[1] for t in clue["semantic"] if t.split(":")[0] in ("side", "turn"))
        n = int(next(t.split(":")[1] for t in clue["semantic"] if t.startswith("room:")))
        t = pick(SEM)
        s, o = pick(SIDE[side]), pick(ORD[n])
        return t.replace("{s}으로", josa_ro(s)).format(s=s, o=o)
    t = pick(GEO)
    d = pick(DIRW[int(clue["direction_deg"]) % 360])
    return t.replace("{d}으로", josa_ro(d)).format(d=d, m=f"{clue['distance_m']:.0f}")


def to_sim(parsed: dict | None, victim: int) -> dict | None:
    """Parsed JSON -> simulator clue (or None when nothing usable came out). Side + ordinal beat direction/distance."""
    if not parsed or parsed.get("utterance") != "clue":
        return None
    sem = parsed.get("semantic") or []
    side = next((t.split(":")[1] for t in sem if t.startswith(("side:", "turn:"))), None)
    if side is None:   # "오른쪽 복도 n번째 방" read as corridor:right is the same room row in a one-corridor building
        side = next((t.split(":")[1] for t in sem if t in ("corridor:left", "corridor:right")), None)
    room = next((t.split(":")[1] for t in sem if t.startswith("room:")), None)
    base = {"utterance": "clue", "certainty": parsed.get("certainty") or "high", "count": parsed.get("count") or 1,
            "victim": victim}
    if side and room and room.isdigit():
        return base | {"semantic": ["corridor:main", f"side:{side}", f"room:{room}"]}
    if parsed.get("direction_deg") is not None or parsed.get("distance_m") is not None:
        c = base | {"anchor": "reporter", "direction_deg": parsed.get("direction_deg"), "distance_m": parsed.get("distance_m")}
        if parsed.get("distance_m") is None and parsed.get("distance_qual"):
            c["distance_qual"] = parsed["distance_qual"]
        return c
    return None
