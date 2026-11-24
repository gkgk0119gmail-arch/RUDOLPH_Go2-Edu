"""Go2 + XT16 + D435i model vs the real robot's calibrated TF."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_go2_xt16 as b  # noqa: E402
from merge_fixed_links import origin_to_tf  # noqa: E402

CFG = json.loads((ROOT / "configs/go2_xt16_sensors.json").read_text())
URDF = ET.parse(ROOT / "assets/go2_xt16/go2_xt16.urdf").getroot()
JOINT = {j.find("child").get("link"): j for j in URDF.findall("joint")}


def tf(link):
    T = np.eye(4)
    while link in JOINT:
        T = origin_to_tf(JOINT[link]) @ T
        link = JOINT[link].find("parent").get("link")
    assert link == "base"
    return T


def test_frames_match_real_tf():
    assert tf("hesai_lidar")[:3, 3] == pytest.approx([0.1454, 0, 0.1331])
    # yaw +90 deg of the real TF: LiDAR +x -> robot left, LiDAR +y -> robot rear
    assert tf("hesai_lidar")[:3, :3] @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-6)
    assert tf("hesai_lidar")[:3, :3] @ [0, 1, 0] == pytest.approx([-1, 0, 0], abs=1e-6)
    assert tf("camera_link")[:3, 3] == pytest.approx([0.32, 0, 0.055])
    fwd = tf("camera_depth_optical_frame")[:3, :3] @ [0, 0, 1]
    assert fwd[2] == pytest.approx(-np.sin(0.227), abs=1e-6)  # looks 13 deg down
    # FAST-LIO extrinsic (LiDAR in IMU) of the real robot
    imu = origin_to_tf(JOINT["imu"])[:3, 3] if "imu" in JOINT else np.array([-0.02557, 0, 0.04232])
    assert tf("hesai_lidar")[:3, 3] - imu == pytest.approx([0.17097, 0, 0.09078], abs=1e-5)


def test_geometry():
    p = b.layout(CFG)
    assert b.top_z(p["xt16"]) == pytest.approx(0.1331 - 0.0464 + 0.076)
    # mount fills exactly the space between trunk and LiDAR
    assert b.top_z(p["xt16_mount"]) == pytest.approx(p["xt16"]["center"][2] - 0.038)


def test_tree_mass_and_names():
    names = [l.get("name") for l in URDF.findall("link")]
    assert len(names) == len(set(names))
    assert len(JOINT) == len(names) - 1  # one root
    def mass(root):
        return sum(float(m.get("value")) for m in root.iter("mass"))
    M, _, I = b.payload(b.layout(CFG))
    assert mass(URDF) - mass(ET.parse(ROOT / "assets/go2/go2.urdf").getroot()) == pytest.approx(M)
    assert (np.linalg.eigvalsh(I) > 0).all()


def test_mesh_files_exist():
    for m in URDF.iter("mesh"):
        f = ROOT / "assets/go2_xt16" / m.get("filename")
        if "third_party" in str(f) and not (ROOT / "third_party").exists():
            pytest.skip("third_party not fetched")
        assert f.exists(), f
