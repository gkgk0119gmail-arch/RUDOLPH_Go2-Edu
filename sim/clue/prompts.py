"""Parser prompt variants (clue/parse.py PROMPT is v2). Same JSON schema; fields left out mean null / empty.

v2c: v2 rules, but the model writes only the fields the sentence says (shorter output -> faster).
v3f: v2 full-field output (every field written, null when not said) + the v3 rules below. Compact output (v2c, v3)
     halved latency but broke JSON and made the 8B model copy example numbers into unsaid fields, so v3f keeps the
     full checklist and only adds the rules.
v3:  v2c + fixes from the v2 error analysis (results/clue/parse_8b_v2_pilot.*):
     someone else's whereabouts inside "살려주세요" -> clue; place-only short answers -> clue; fire / smoke / 119 -> other;
     side only for "n-th room on that side", a bare "제 왼쪽" is a direction; "근처" + place -> near:<place>.
"""
from .parse import PROMPT as V2

_FIELDS_END = V2.index("예시")
_RULES = V2[:_FIELDS_END]

_COMPACT = """출력 규칙: 말에 나온 필드만 쓰세요. null, 빈 리스트, 빈 객체인 필드는 아예 쓰지 마세요. utterance는 항상 쓰세요.

예시
말: "제 왼쪽으로 10미터 가면 화장실 있는데 그 옆방에 두 명 있을 거예요."
{{"utterance": "clue", "anchor": "reporter", "direction_deg": 270, "distance_m": 10, "semantic": ["place:toilet", "room:next"], "count": 2, "certainty": "mid"}}
말: "복도 오른편 네 번째 방에 할머니가 깔려 계세요."
{{"utterance": "clue", "mobility": "static", "semantic": ["side:right", "room:4"], "appearance": {{"age": "elderly", "sex": "female"}}, "certainty": "high"}}
말: "아이고 허리야… 좀 꺼내 주세요."
{{"utterance": "help", "certainty": "high"}}
"""

_V3_EXTRA = """추가 규칙
- 본인 구조 요청이 섞여 있어도 다른 사람(가족, 동료, 사람들)이 어디 있는지·있다는 걸 말하면 "clue" ("도와주세요, 제 동생이 아직 저 안에 있어요" -> clue)
- 장소·층만 짧게 답한 말도 "clue" ("주방이요." -> place:kitchen, "3층이요 3층" -> floor 3, "얼마 안 멀어요" -> distance_qual near)
- 불, 연기, 119, 대피 상황만 묻거나 말하면 "other" ("숨쉬기가 힘들어요, 연기 때문에" -> other)
- side는 "그쪽 편의 n번째 방·방들"처럼 복도의 편을 말할 때만. "제 오른쪽에서요"처럼 방향만 말하면 side 없이 direction_deg
- "<장소> 근처·옆"은 near:<장소>. place:<장소>는 그 장소 안에 있을 때만
- 방향·거리를 말하지 않았으면 anchor를 쓰지 마세요

말: "누가 좀… 사무실에 사람이 있었는데… 아직 있는지는 몰라요."
{{"utterance": "clue", "semantic": ["place:office"], "certainty": "low"}}
말: "제 오른쪽 앞이요! 노란 옷 입은 아이 하나요."
{{"utterance": "clue", "anchor": "reporter", "direction_deg": 45, "count": 1, "appearance": {{"age": "child", "top_color": "yellow"}}, "certainty": "high"}}
"""

_TAIL = """
말: "{u}"
JSON만 출력:"""

PROMPTS = {
    "v2": V2,
    "v2c": _RULES + _COMPACT + _TAIL,
    "v3": _RULES + _COMPACT + _V3_EXTRA + _TAIL,
}
_RULES_ONLY = _V3_EXTRA.split("\n\n말:")[0] + "\n"        # v3 rules without the compact examples
PROMPTS["v3f"] = V2[:V2.index("말: \"{u}\"")] + _RULES_ONLY + _TAIL
