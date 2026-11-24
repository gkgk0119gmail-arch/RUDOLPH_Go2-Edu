"""Go2 EDU Plus + Hesai XT16 + D435i: URDF, USD overlay and a geometry report from configs/go2_xt16_sensors.json.

    python tools/build_go2_xt16.py          # -> assets/go2_xt16/{go2_xt16.urdf,go2_xt16.usda,geometry.json}

The USD overlay adds the sensor shapes as colliders *inside* the stock base rigid body: our own URDF->USD conversion
nests rigid bodies and is unstable in training (docs/SIM_GUIDE.md). The payload mass is therefore not in the USD
(the base carries an explicit mass); geometry.json holds the combined mass / CoM / inertia for the env event.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "go2_xt16"
SRC = ROOT / "assets" / "go2" / "go2.urdf"
STOCK = ("../go2_stock/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/"
         "IsaacLab/Robots/Unitree/Go2/go2.usd")


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def box_inertia(m, s):
    x, y, z = s
    return m / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def layout(cfg):
    """Sensor bodies: name -> dict(kind, size, center (base), pitch, mass, inertia about its centre in base axes)."""
    x, m, d = cfg["xt16"], cfg["mount"], cfg["d435i"]
    o = np.array(x["optical_xyz"], float)
    bottom = o[2] - x["optical_above_base"]
    r, h = x["radius"], x["height"]
    it = x["mass"] / 12 * (3 * r * r + h * h)
    parts = {"xt16": dict(kind="cylinder", radius=r, length=h, center=o + [0, 0, -x["optical_above_base"] + h / 2],
                          pitch=0.0, mass=x["mass"], I=np.diag([it, it, x["mass"] * r * r / 2]))}
    mh = bottom - m["trunk_top_z"]
    assert mh > 0, f"XT16 bottom {bottom:.4f} lies inside the trunk"
    ms = [*m["size_xy"], mh]
    parts["xt16_mount"] = dict(kind="box", size=ms, center=np.array([o[0], o[1], m["trunk_top_z"] + mh / 2]), pitch=0.0,
                               mass=m["mass"], I=box_inertia(m["mass"], ms))
    R = rot_y(d["camera_link_pitch"])
    parts["d435i"] = dict(kind="box", size=d["box"], pitch=d["camera_link_pitch"], mass=d["mass"],
                          center=np.array(d["camera_link_xyz"]) + R @ np.array(d["box_center_in_camera_link"]),
                          I=R @ box_inertia(d["mass"], d["box"]) @ R.T)
    return parts


def top_z(p):
    if p["kind"] == "cylinder":
        return p["center"][2] + p["length"] / 2
    R, s = rot_y(p["pitch"]), np.array(p["size"]) / 2
    return p["center"][2] + float(np.abs(R[2]) @ s)


def payload(parts):
    M = sum(p["mass"] for p in parts.values())
    c = sum(p["mass"] * p["center"] for p in parts.values()) / M
    I = np.zeros((3, 3))
    for p in parts.values():
        d = p["center"] - c
        I += p["I"] + p["mass"] * (d @ d * np.eye(3) - np.outer(d, d))
    return M, c, I


def fmt(v):
    return " ".join(f"{float(a):.7g}" for a in v)


def write_urdf(cfg, parts):
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("name", "go2_xt16_d435i")
    for mesh in root.iter("mesh"):  # same depth below assets/ as assets/go2
        assert mesh.get("filename").startswith("../../third_party/"), mesh.get("filename")

    def geom(parent, p, rpy="0 0 0"):
        ET.SubElement(parent, "origin", xyz="0 0 0", rpy=rpy)
        g = ET.SubElement(parent, "geometry")
        if p["kind"] == "box":
            ET.SubElement(g, "box", size=fmt(p["size"]))
        else:
            ET.SubElement(g, "cylinder", radius=f"{p['radius']:.6g}", length=f"{p['length']:.6g}")

    def joint(name, parent, child, xyz, rpy):
        j = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(j, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)

    for name, p in parts.items():
        link = ET.SubElement(root, "link", name=f"{name}_body")
        R = rot_y(p["pitch"])
        I = R.T @ p["I"] @ R  # link axes
        ine = ET.SubElement(link, "inertial")
        ET.SubElement(ine, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ine, "mass", value=f"{p['mass']:.6g}")
        ET.SubElement(ine, "inertia", ixx=f"{I[0,0]:.6g}", ixy="0", ixz="0", iyy=f"{I[1,1]:.6g}", iyz="0",
                      izz=f"{I[2,2]:.6g}")
        vis = ET.SubElement(link, "visual")
        if name == "xt16":
            ET.SubElement(vis, "origin", xyz=fmt([0, 0, -p["length"] / 2]), rpy="0 0 0")
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename="meshes/xt32.dae")
        else:
            geom(vis, p)
        geom(ET.SubElement(link, "collision"), p)
        joint(f"{name}_body_joint", "base", f"{name}_body", p["center"], [0, p["pitch"], 0])

    x, d = cfg["xt16"], cfg["d435i"]
    for frame, xyz, rpy in (("hesai_lidar", x["optical_xyz"], [0, 0, x["optical_yaw"]]),
                            ("camera_link", d["camera_link_xyz"], [0, d["camera_link_pitch"], 0])):
        ET.SubElement(root, "link", name=frame)
        joint(f"{frame}_joint", "base", frame, xyz, rpy)
    ET.SubElement(root, "link", name="camera_depth_optical_frame")
    joint("camera_depth_optical_joint", "camera_link", "camera_depth_optical_frame", [0, 0, 0],
          [-np.pi / 2, 0, -np.pi / 2])
    ET.indent(tree)
    tree.write(OUT / "go2_xt16.urdf", xml_declaration=True, encoding="utf-8")


def write_usda(parts):
    prims = []
    for name, p in parts.items():
        c = ", ".join(f"{v:.7g}" for v in p["center"])
        if p["kind"] == "cylinder":
            body = f'def Cylinder "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI"])'
            shape = (f'uniform token axis = "Z"\n            double radius = {p["radius"]:.7g}\n'
                     f'            double height = {p["length"]:.7g}')
            sc = "(1, 1, 1)"
        else:
            body = f'def Cube "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI"])'
            shape = "double size = 1"
            sc = "(" + ", ".join(f"{v:.7g}" for v in p["size"]) + ")"
        prims.append(f"""        {body}
        {{
            {shape}
            color3f[] primvars:displayColor = [(0.12, 0.12, 0.13)]
            double3 xformOp:translate = ({c})
            float3 xformOp:rotateXYZ = (0, {np.degrees(p["pitch"]):.6g}, 0)
            float3 xformOp:scale = {sc}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
        }}""")
    (OUT / "go2_xt16.usda").write_text(f"""#usda 1.0
(
    defaultPrim = "go2_description"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Stock Isaac Lab Go2 + XT16 / mount / D435i colliders (tools/build_go2_xt16.py). Payload mass: env event."
)

def Xform "go2_description" (
    prepend references = @{STOCK}@
)
{{
    over "base"
    {{
{chr(10).join(prims)}
    }}
}}
""")


def main():
    cfg = json.loads((ROOT / "configs" / "go2_xt16_sensors.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    parts = layout(cfg)
    write_urdf(cfg, parts)
    write_usda(parts)
    M, c, I = payload(parts)
    rep = {"top_z_above_base": {k: round(top_z(p), 4) for k, p in parts.items()},
           "highest_known": round(max(top_z(p) for p in parts.values()), 4),
           "payload": {"mass": round(M, 4), "com": c.round(5).tolist(), "inertia_about_com": I.round(7).tolist()},
           "not_modelled": cfg["unmeasured"]}
    (OUT / "geometry.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
