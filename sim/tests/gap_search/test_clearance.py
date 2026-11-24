import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "go2_lab" / "gap_search"))
from clearance import BLOCKED, CROUCH, UNKNOWN, WALK, Body, classify, find_gaps  # noqa: E402

V = 0.05


def room(h=None, width=0.8, nx=60, ny=40):
    """Free 3 x 2 m room with a wall at x = 1.4-1.6 m; opening of `width`, lintel at height h (None: closed)."""
    g = np.ones((nx, ny, 20), np.int8)
    g[:, :, 0] = 2  # floor
    g[28:32] = 2
    if h is not None:
        w = int(round(width / V / 2))
        g[28:32, ny // 2 - w:ny // 2 + w, 1:int(round(h / V))] = 1
    return g


def gaps(g):
    c, clear = classify(g, 0.0, V)
    return find_gaps(c, clear, V)


@pytest.mark.parametrize("h,want", [(0.30, BLOCKED), (0.35, BLOCKED), (0.40, CROUCH), (0.45, CROUCH), (0.50, CROUCH),
                                    (0.55, WALK), (0.80, WALK)])
def test_height_classes(h, want):
    c, clear = classify(room(h), 0.0, V)
    assert c[30, 20] == want
    assert clear[30, 20] == pytest.approx(h)


def test_low_gap_found_and_joins_both_sides():
    g = gaps(room(0.45))
    assert len(g) == 1 and g[0].connects and g[0].clearance == pytest.approx(0.45)
    j = g[0].to_json((0, 0), V)
    assert j["center"] == pytest.approx([1.5, 1.0], abs=0.06) and j["status"] == "deferred"


def test_closed_high_and_narrow_openings_are_not_gaps():
    assert gaps(room(None)) == []
    assert gaps(room(0.8)) == []       # walk through standing
    assert gaps(room(0.45, 0.3)) == [] # narrower than the robot
    assert len(gaps(room(0.45, 0.6))) == 1


def test_unknown_is_never_passable():
    g = room(0.45)
    g[30, :, 1:10][g[30, :, 1:10] == 1] = 0
    c, _ = classify(g, 0.0, V)
    assert c[30, 20] == UNKNOWN
    # only one side can be reached: the gap is kept but not marked as connecting
    assert not any(k.connects for k in gaps(g))


def test_far_side_unseen_keeps_deferred_gap():
    g = room(0.45)
    g[32:] = 0
    found = gaps(g)
    assert len(found) == 1 and not found[0].connects and len(found[0].entries) == 1


def test_small_step_is_not_an_obstacle():
    g = room(0.8)
    g[10, 20, 1] = 2  # 5-10 cm high
    c, _ = classify(g, 0.0, V)
    assert c[10, 20] == WALK


def test_map_must_reach_standing_height():
    with pytest.raises(ValueError):
        classify(np.ones((4, 4, 6), np.int8), 0.0, V, Body())
