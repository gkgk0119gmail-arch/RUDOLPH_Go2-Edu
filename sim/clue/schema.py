"""Structured clue: what one Korean sentence from a person at the scene says about where victims are.

The VLM only maps sentence -> this JSON. Turning it into a probability map and updating it is code (see the plan doc,
"단서 → 확률 지도 → 프론티어 가중치"). Fields are null when the sentence does not say them.

Conventions
  utterance     "clue"  says something about where / who OTHER victims are (even only "한 명 더 있어요")
                "help"  the speaker asks for help or describes their own state ("살려주세요", "다리가 아파요");
                        the speaker is already found, the map does not change
                "other" nothing about victims ("연기가 너무 많아요", "여기 어디예요?")
                A help call that also mentions others ("살려주세요, 안에 애가 있어요") is a clue.
  anchor        whose position / facing the direction and distance are measured from
                "reporter"            the speaker ("제 오른쪽", "여기서 20 m")
                "robot"               the robot that listens ("네 왼쪽", "로봇 기준으로")
                "landmark:<place>"    a named place ("입구에서", "계단에서 오른쪽") -- <place> from PLACES
                null                  no direction / distance given
  direction_deg clockwise from the anchor's facing: 0 front, 90 right, 180 back, 270 left (8 sectors, multiples of 45)
  distance_m    stated metric distance; distance_qual "near" / "far" when only said in words
  floor         absolute floor number; floor_rel +1 / -1 for "위층" / "아래층"
  semantic      ordered route / place tokens "kind:value" (grammar below) -- the path a person would describe
  count         number of victims, null if not said ("여러 명" -> null, count_qual "several")
  appearance    {"age": child|adult|elderly, "sex": male|female, "top_color": <COLORS>} keys only when said
  mobility      "static" trapped / pinned / collapsed ("갇혀", "깔려", "쓰러져", "못 움직여") | "mobile" moving ("돌아다녀",
                "나가려고 했어") | null -- sets how fast the belief spreads over time
  certainty     speaker's own certainty: "high" (plain statement) | "mid" ("아마", "~것 같아요") | "low" ("잘 모르겠는데")
"""
from __future__ import annotations

import json
import re

DIRS = (0, 45, 90, 135, 180, 225, 270, 315)
PLACES = ("entrance", "lobby", "stairs", "elevator", "toilet", "kitchen", "office", "classroom", "storage", "hall")
OBJECTS = ("desk", "bed", "table", "window", "door", "shelf", "sofa")
COLORS = ("red", "blue", "yellow", "green", "black", "white", "gray")
# semantic token grammar
SEM_KINDS = {
    "corridor": ("inner", "end", "main", "left", "right"),   # 안쪽 / 끝 / 큰 / 왼쪽 / 오른쪽 복도
    "turn": ("left", "right"),                               # 좌회전 / 우회전 (a change of direction)
    "side": ("left", "right"),                               # 복도 왼편 / 오른편에 있는 (which side, no turn)
    "room": tuple(str(i) for i in range(1, 6)) + ("last", "next"),  # n번째 방 / 마지막 방 / 옆방
    "place": PLACES,                                         # 지나가거나 도착하는 장소
    "near": OBJECTS + PLACES,                                # ~ 옆 / 근처
    "under": OBJECTS,                                        # ~ 밑
}
CERTAINTY = {"high": 0.9, "mid": 0.6, "low": 0.3}
MOBILITY = ("static", "mobile")
KEYS = ("utterance", "mobility", "anchor", "direction_deg", "distance_m", "distance_qual", "floor", "floor_rel", "semantic", "count",
        "count_qual", "appearance", "certainty")


def empty() -> dict:
    return {k: None for k in KEYS} | {"utterance": "clue", "semantic": [], "appearance": {}, "certainty": "high"}


def complexity(c: dict) -> str:
    """easy / medium / hard by the number of facts the sentence carries (non-clues: "none")."""
    if c.get("utterance", "clue") != "clue":
        return "none"
    n = sum(c.get(k) is not None for k in ("anchor", "direction_deg", "distance_m", "distance_qual", "floor",
                                            "floor_rel", "count", "count_qual"))
    n += len(c.get("semantic") or []) + len(c.get("appearance") or {})
    return "easy" if n <= 2 else "medium" if n <= 4 else "hard"


def valid_token(t: str) -> bool:
    k, _, v = t.partition(":")
    return k in SEM_KINDS and v in SEM_KINDS[k]


def normalize(d: dict | None) -> tuple[dict | None, list[str]]:
    """Coerce a model output to the schema. -> (clue or None, problems). Unknown values are dropped, not guessed."""
    if not isinstance(d, dict):
        return None, ["not a JSON object"]
    out, bad = empty(), []
    u = d.get("utterance", "clue")
    if u in ("clue", "help", "other"):
        out["utterance"] = u
    else:
        bad.append(f"utterance {u!r}")
    a = d.get("anchor")
    if a in ("reporter", "robot") or (isinstance(a, str) and a.startswith("landmark:") and a[9:] in PLACES):
        out["anchor"] = a
    elif a is not None:
        bad.append(f"anchor {a!r}")
    try:
        if d.get("direction_deg") is not None:
            v = float(d["direction_deg"]) % 360
            out["direction_deg"] = min(DIRS, key=lambda x: abs((v - x + 180) % 360 - 180))
    except (TypeError, ValueError):
        bad.append(f"direction {d.get('direction_deg')!r}")
    for k, typ in (("distance_m", float), ("floor", int), ("floor_rel", int), ("count", int)):
        if d.get(k) is not None:
            try:
                out[k] = typ(d[k])
            except (TypeError, ValueError):
                bad.append(f"{k} {d[k]!r}")
    for k, ok in (("distance_qual", ("near", "far")), ("count_qual", ("several",)), ("certainty", tuple(CERTAINTY)),
                  ("mobility", MOBILITY)):
        if d.get(k) in ok:
            out[k] = d[k]
        elif d.get(k) is not None:
            bad.append(f"{k} {d[k]!r}")
    sem = d.get("semantic") or []
    if isinstance(sem, list):
        out["semantic"] = [t for t in sem if isinstance(t, str) and valid_token(t)]
        bad += [f"token {t!r}" for t in sem if not (isinstance(t, str) and valid_token(t))]
    ap = d.get("appearance") or {}
    if isinstance(ap, dict):
        for k, ok in (("age", ("child", "adult", "elderly")), ("sex", ("male", "female")), ("top_color", COLORS)):
            if ap.get(k) in ok:
                out["appearance"][k] = ap[k]
            elif ap.get(k) is not None:
                bad.append(f"appearance.{k} {ap[k]!r}")
    return out, bad


def extract_json(text: str) -> dict | None:
    """First {...} block of a model reply (tolerates ```json fences and trailing words)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s[: s.rfind("}") + 1]):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def compare(pred: dict | None, gold: dict) -> dict:
    """Per-field correctness of a parsed clue (None pred = every field wrong). distance_m within 20 %."""
    if pred is None:
        return {k: False for k in KEYS} | {"all": False}
    r = {}
    for k in KEYS:
        p, g = pred.get(k), gold.get(k)
        if k == "distance_m" and p is not None and g is not None:
            r[k] = abs(p - g) <= 0.2 * g
        else:
            r[k] = p == g
    r["all"] = all(r.values())
    return r
