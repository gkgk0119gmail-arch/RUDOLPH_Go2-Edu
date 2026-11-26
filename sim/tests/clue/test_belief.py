import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clue import belief as B  # noqa: E402


def room_grid():
    g = np.full((20, 30), B.FREE)
    g[:, 15] = B.OCC          # wall with a door at row 10
    g[10, 15] = B.FREE
    return g


def test_spread_conserves_and_respects_walls():
    g = room_grid()
    b = np.zeros(g.shape)
    b[2, 14] = 1.0            # next to the wall, far from the door
    b2 = B.spread(b, g, 0.25, 30.0, 0.01)
    assert abs(b2.sum() - 1) < 1e-9
    assert b2[g == B.OCC].sum() == 0
    assert b2[:, 16:].sum() < 1e-3   # little has gone round through the door yet
    assert np.allclose(B.spread(b, g, 0.25, 600.0, 0.0), b)  # static: no spread


def test_observe_lowers_seen_mass():
    b = np.full((4, 4), 1 / 16)
    seen = np.zeros((4, 4), bool)
    seen[:2] = True
    b2, miss = B.observe(b, seen, 0.8)
    assert abs(b2.sum() - 1) < 1e-9 and b2[seen].sum() < 0.5 and abs(miss - 0.6) < 1e-9


def test_frontier_mass_is_geodesic():
    g = np.full((10, 10), B.UNK)
    g[:, :3] = B.FREE         # explored strip on the left
    g[:5, 3] = B.OCC          # wall along the top half of the boundary
    b = np.zeros(g.shape)
    b[1, 8] = 1.0             # victim mass behind the wall
    fr = B.frontiers(g)
    m = B.frontier_mass(b, g, fr)
    best = max(m, key=m.get)
    assert best[0] >= 5       # reached through the open lower half, not straight through the wall


def test_prior_mixture():
    g = np.full((40, 40), B.FREE)
    p = B.clue_prior(g, 0.5, (20, 20), 0.0, {"direction_deg": 90, "distance_m": 8.0, "certainty": "high"})
    assert abs(p.sum() - 1) < 1e-9
    r, c = np.unravel_index(p.argmax(), p.shape)
    assert r > 30 and abs(c - 20) <= 3   # facing +col (east), "right" = south = larger row
