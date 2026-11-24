"""Side / top view of the Go2 + XT16 + D435i collision model (assets/go2_xt16/go2_xt16.urdf)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from merge_fixed_links import origin_to_tf, rpy_to_mat  # noqa: E402

root = ET.parse(ROOT / "assets/go2_xt16/go2_xt16.urdf").getroot()
J = {j.find("child").get("link"): j for j in root.findall("joint")}


def q_of(name, thigh, calf):
    return thigh if "thigh" in name else calf if "calf_joint" in name else 0.0


def tf(link, q):
    T = np.eye(4)
    while link in J:
        j = J[link]
        A = origin_to_tf(j)
        if j.get("type") == "revolute":
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            R = np.eye(4)
            R[:3, :3] = rpy_to_mat(*(ax * q_of(j.get("name"), *q)))
            A = A @ R
        T = A @ T
        link = j.find("parent").get("link")
    return T


def points(g):
    if (b := g.find("box")) is not None:
        s = np.array([float(v) for v in b.get("size").split()]) / 2
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
    a = np.linspace(0, 2 * np.pi, 24)
    if (c := g.find("cylinder")) is not None:
        r, h = float(c.get("radius")), float(c.get("length")) / 2
        return np.concatenate([np.c_[r * np.cos(a), r * np.sin(a), np.full(24, z)] for z in (-h, h)])
    if (s := g.find("sphere")) is not None:
        r = float(s.get("radius"))
        return np.array([[r * np.cos(t) * np.sin(p), r * np.sin(t) * np.sin(p), r * np.cos(p)]
                         for t in a for p in np.linspace(0, np.pi, 12)])
    return None


def shapes(q):
    out = []
    for l in root.findall("link"):
        for c in l.findall("collision"):
            p = points(c.find("geometry"))
            if p is not None:
                T = tf(l.get("name"), q) @ origin_to_tf(c)
                out.append((l.get("name"), p @ T[:3, :3].T + T[:3, 3]))
    return out


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (title, q, view) in zip(ax, [("standing", (0.8, -1.5), "side"), ("crouched (example pose)", (1.25, -2.35), "side"),
                                    ("top view", (0.8, -1.5), "top")]):
    S = shapes(q)
    ground = min(p[:, 2].min() for _, p in S)
    i = [0, 2] if view == "side" else [0, 1]
    for name, p in S:
        xy = p[:, i] - ([0, ground] if view == "side" else [0, 0])
        h = ConvexHull(xy)
        col = {"xt16_body": "#c0392b", "d435i_body": "#1f6fb2", "xt16_mount_body": "#999999"}.get(name, "#d7d7d7")
        a.fill(*xy[h.vertices].T, facecolor=col, edgecolor="#444", lw=0.6, alpha=0.9)
    if view == "side":
        top = max(p[:, 2].max() for _, p in S) - ground
        print(f"{title}: base {-ground:.3f} m, highest point {top:.3f} m")
        a.axhline(top, color="#c0392b", ls="--", lw=1)
        a.axhline(0, color="k", lw=1)
        a.set_title(f"{title}: base {-ground:.2f} m, top {top:.2f} m")
        a.set_ylim(-0.03, 0.6)
        a.set_ylabel("z [m]")
    else:
        a.set_title("top view")
        a.set_ylabel("y [m]")
    a.set_xlabel("x forward [m]")
    a.set_aspect("equal")
fig.tight_layout()
fig.savefig(ROOT / "assets/go2_xt16/preview.png", dpi=110)
