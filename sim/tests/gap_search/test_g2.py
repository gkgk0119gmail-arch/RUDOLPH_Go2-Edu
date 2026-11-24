import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "go2_lab" / "gap_search"))
import g2_sim as g  # noqa: E402


@pytest.mark.parametrize("seed", range(6))
def test_worlds_are_stratified_and_deterministic(seed):
    w, v = g.make_world(seed), g.make_world(seed)
    assert (w.grid == v.grid).all() and w.victim == v.victim
    assert w.behind_gap == (seed % 2 == 0)
    for p in (w.start, w.witness, w.victim):
        assert w.grid[p] == g.FREE


def test_sight_does_not_leak_through_walls_or_gaps():
    w = g.make_world(1)
    w.grid[:] = 0
    w.grid[20:41, [20, 40]] = g.WALL
    w.grid[[20, 40], 20:41] = g.WALL
    w.grid[30, 40] = g.GAP
    out = np.ones(w.grid.shape, bool)
    out[20:41, 20:41] = False
    for p in [(30, 30), (21, 21), (25, 33), (39, 39)]:
        r = g.Run(w, "posture")
        r.pos = p
        assert not (r.visible() & out).any()


def test_relevance_never_uses_true_position():
    w = g.make_world(2)
    r = g.Run(w, "linked")
    r.sense()
    b = r.belief()
    w.victim = w.start
    assert (b == r.belief()).all()


def test_methods_identical_before_any_clue():
    """separate / linked_nop / posture take the same decisions until the witness is met."""
    for seed in range(8):
        w = g.make_world(seed)
        w.witness = (0, 0)  # inside a wall corner, never met
        ts = {m: g.Run(w, m).run()["path"] for m in ("posture", "separate", "linked_nop")}
        # witness cell may be glimpsed as a wall; skip those runs
        if not any(g.Run(w, m).clue for m in ts):
            runs = {m: g.Run(w, m) for m in ts}
            res = {m: r.run() for m, r in runs.items()}
            if not any(r.clue for r in runs.values()):
                assert len({round(v["path"], 6) for v in res.values()}) == 1
