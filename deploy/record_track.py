"""Record FAST-LIO odometry + go2_hipan_low debug (+ optional depth frames) for sim-to-real checks.

    python3 record_track.py --seconds 90 --out logs/track_<name>.npz
    python3 record_track.py --seconds 5 --depth 10 --out logs/depth_<name>.npz   # also keep 10 depth frames
"""
import argparse
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--out", required=True)
ap.add_argument("--depth", type=int, default=0, help="number of depth frames to keep (evenly over the run)")
ap.add_argument("--depth_topic", default="/camera/depth/image_rect_raw")
args = ap.parse_args()

rclpy.init()
n = rclpy.create_node("record_track")
odom, dbg, depth = [], [], []
n.create_subscription(Odometry, "/Odometry", lambda m: odom.append(
    [time.time(), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
     m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z]), 50)
n.create_subscription(Float32MultiArray, "/hipan/debug", lambda m: dbg.append([time.time(), *m.data]), 100)
last_depth = [0.0]
period = args.seconds / max(args.depth, 1)


def on_depth(m):
    if len(depth) < args.depth and time.time() - last_depth[0] > period * 0.9:
        last_depth[0] = time.time()
        dt = np.uint16 if m.encoding in ("16UC1", "mono16") else np.float32
        depth.append(np.frombuffer(bytes(m.data), dt).reshape(m.height, m.width).copy())


if args.depth:
    n.create_subscription(Image, args.depth_topic, on_depth, 5)
t0 = time.time()
while time.time() - t0 < args.seconds:
    rclpy.spin_once(n, timeout_sec=0.01)
np.savez(args.out, odom=np.array(odom), dbg=np.array(dbg), depth=np.array(depth) if depth else np.zeros(0))
print("odom %d  debug %d  depth %d -> %s" % (len(odom), len(dbg), len(depth), args.out))
