"""Re-score a condition-C parse file: gold rebuilt from the simulator clue with the current schema (the first
c_sentences were labelled when room:n only went to 5, so "6번째 방" golds lost the room), plus a map-relevant metric:
does the parsed JSON become the same simulator clue (same room, or same direction and distance) as the gold.

    python scripts/rescore_parse.py results/clue/parse_8b_v2_c.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clue import schema as S  # noqa: E402
from clue.speech import to_sim  # noqa: E402


def same_place(p, g):
    if g is None:
        return p is None
    if p is None:
        return False
    if g.get("semantic"):
        return p.get("semantic") == g["semantic"]
    return p.get("direction_deg") == g.get("direction_deg") and p.get("distance_m") == g.get("distance_m")


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    fields, place = Counter(), Counter()
    n = Counter()
    for r in rows:
        gold, _ = S.normalize(r["sim"]) if "sim" in r else (r["gold"], None)
        ok = S.compare(r["pred"], gold)
        kind = r["id"].split("_")[0]
        for k in (*S.KEYS, "all"):
            fields[k] += ok[k]
        n[kind] += 1
        place[kind] += same_place(to_sim(r["pred"], 0), to_sim(gold, 0))
    out = {"n": len(rows), "fields": {k: round(v / len(rows), 3) for k, v in fields.items()},
           "same_place": {k: round(place[k] / n[k], 3) for k in n}}
    Path(path).with_suffix(".rescored.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(path, json.dumps(out, ensure_ascii=False))
