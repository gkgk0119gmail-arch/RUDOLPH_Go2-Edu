import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clue import gen, schema as S  # noqa: E402


def test_normalize_snaps_and_drops():
    c, bad = S.normalize({"utterance": "clue", "anchor": "reporter", "direction_deg": 100, "semantic": ["room:next", "place:shelf"],
                          "count": "2", "appearance": {"age": "teen"}})
    assert c["direction_deg"] == 90 and c["semantic"] == ["room:next"] and c["count"] == 2 and c["appearance"] == {}
    assert any("place:shelf" in b for b in bad) and any("teen" in b for b in bad)


def test_extract_json_fenced():
    assert S.extract_json('```json\n{"count": 1}\n```')["count"] == 1
    assert S.extract_json("no json") is None


def test_compare_distance_tolerance():
    g = S.empty() | {"anchor": "reporter", "distance_m": 20.0}
    assert S.compare(g | {"distance_m": 23.0}, g)["all"]
    assert not S.compare(g | {"distance_m": 25.0}, g)["distance_m"]
    assert not S.compare(None, g)["all"]


def test_complexity():
    assert S.complexity(S.empty() | {"semantic": ["turn:right"]}) == "easy"
    assert S.complexity(S.empty() | {"utterance": "help"}) == "none"


def test_josa():
    assert gen.josa("여자", "이", "가") == "여자가" and gen.josa("사람", "이", "가") == "사람이"
    assert gen.josa("복도", "으로", "로") == "복도로" and gen.josa("화장실", "으로", "로") == "화장실로"


def test_generated_labels_valid():
    for r in gen.make(200, 1):
        c, bad = S.normalize(r["gold"])
        assert not bad and c == r["gold"]


def test_pilot_labels_valid():
    p = Path(__file__).resolve().parents[2] / "results/clue/pilot_llm_v1.jsonl"
    for l in p.read_text().splitlines():
        r = json.loads(l)
        c, bad = S.normalize(r["gold"])
        assert not bad and c == r["gold"], r["id"]
