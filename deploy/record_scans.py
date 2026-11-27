"""Record FAST-LIO scans + poses to one .npz for offline work on the 2D grid (no ROS needed afterwards).

    python3 record_scans.py --out /tmp/scans.npz        # Ctrl+C (or SIGINT) to stop and save
Each scan is voxel-downsampled (0.1 m) to keep the file small.
"""
import argparse
import math
import signal

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--voxel", type=float, default=0.1)
a = ap.parse_args()
rclpy.init()
node = rclpy.create_node("record_scans")
S = dict(pose=None, pts=[], idx=[0], poses=[])


def on_odom(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    S["pose"] = (p.x, p.y, p.z, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def on_cloud(m):
    if S["pose"] is None:
        return
    off = {f.name: f.offset for f in m.fields}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3, "offsets": [off["x"], off["y"], off["z"]],
                   "itemsize": m.point_step})
    a_ = np.frombuffer(bytes(m.data), dt, count=m.width * m.height)
    xyz = np.stack([a_["x"], a_["y"], a_["z"]], 1)
    xyz = xyz[np.isfinite(xyz).all(1)]
    _, keep = np.unique(np.floor(xyz / a.voxel).astype(np.int32), axis=0, return_index=True)
    S["pts"].append(xyz[keep].astype(np.float32))
    S["idx"].append(S["idx"][-1] + len(keep))
    S["poses"].append(S["pose"])


def save(*_):
    np.savez_compressed(a.out, pts=np.concatenate(S["pts"]) if S["pts"] else np.zeros((0, 3)),
                        idx=np.array(S["idx"]), poses=np.array(S["poses"]))
    print("saved", a.out, len(S["poses"]), "scans", flush=True)
    rclpy.shutdown()


node.create_subscription(Odometry, "/Odometry", on_odom, 20)
node.create_subscription(PointCloud2, "/cloud_registered", on_cloud, 20)
signal.signal(signal.SIGINT, save)
signal.signal(signal.SIGTERM, save)
try:
    rclpy.spin(node)
except Exception:  # noqa: BLE001  (shutdown from the signal handler)
    pass
