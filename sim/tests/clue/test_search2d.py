import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clue import search2d as S  # noqa: E402


def test_building_valid():
    for seed in range(5):
        for q in ("spoken", "semantic", "exact"):
            b = S.make_building(seed, quality=q)
            assert b.grid[b.start] == S.FREE
            assert all(b.grid[v] == S.FREE for v in b.victims)
            assert len(b.clues) == 3 and all(0 <= c["victim"] < 6 for c in b.clues)
            for r in b.rooms:
                assert r["side"] in ("left", "right") and r["ordinal"] >= 1


def test_semantic_clue_points_at_victim_room():
    b = S.make_building(3, quality="semantic")
    for c in b.clues:
        if c.get("semantic"):
            side = c["semantic"][1].split(":")[1]
            n = int(c["semantic"][2].split(":")[1])
            r = next(r for r in b.rooms if r["side"] == side and r["ordinal"] == n)
            y0, y1, x0, x1 = r["box"]
            vy, vx = b.victims[c["victim"]]
            assert y0 <= vy < y1 and x0 <= vx < x1


def test_beta0_ignores_clues_and_runs_deterministically():
    b = S.make_building(2)
    r1 = S.Search(b, beta=0.0, seed=2).run()
    r2 = S.Search(b, beta=0.0, seed=2).run()
    assert r1 == r2 and r1["n_found"] >= 1


def test_mobile_victims_move():
    b = S.make_building(4, mobile=0.1)
    s = S.Search(b, beta=0.0, seed=4)
    p0 = list(s.vpos)
    s.t = 60.0
    s._move_victims(60.0)
    assert any(a != c for a, c in zip(p0, s.vpos))
    assert all(b.grid[v] == S.FREE for v in s.vpos)
