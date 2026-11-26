"""HiPAN nav depth student on the real Go2 (10 Hz): D435i depth + low-level state + goal -> /hipan/cmd.

Needs: go2_hipan_low running (POLICY state; LIVE nav needs it started with cmd_source:=topic), FAST-LIO (/Odometry),
RealSense depth 848x480 (start_slam_sim.sh).

    python3 run_hipan_nav.py --goal 3 0              # SHADOW: publishes /hipan/cmd_shadow only (robot on the remote)
    python3 run_hipan_nav.py --goal 3 0 --live       # LIVE: publishes /hipan/cmd
    --goal X Y : metres in the robot's frame at start (x forward, y left). Or click "2D Goal Pose" in rviz (/goal_pose,
                 frame camera_init) at any time.
Stops (stand command) on arrival (0.25 m), stale odometry / depth / low-level state, or Ctrl+C.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

from hipan_nav_policy import DepthConverter, HipanNavPolicy, goal_feature

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(Path(__file__).parent / "weights/hipan_nav_student.npz"))
ap.add_argument("--goal", type=float, nargs=2, help="goal in the start frame (x forward, y left), metres")
ap.add_argument("--live", action="store_true")
ap.add_argument("--v_des", type=float, default=0.5, help="desired speed input (sim: 0.3-0.8 per episode)")
ap.add_argument("--real_pitch", type=float, default=11.6, help="measured D435i pitch down (deg)")
ap.add_argument("--depth_topic", default="/camera/depth/image_rect_raw")
# Optional safety envelope, OFF by default: the student is run exactly as trained. (First live run: it asked for
# 0.9 m/s + yaw / side-step / height / roll changes and the low-level policy lost balance; the fix belongs in training,
# not here.) If enabled, the clipped / smoothed command is what runs AND what the student sees as its previous command.
ap.add_argument("--vx", type=float, nargs=2, default=[-0.5, 1.0])
ap.add_argument("--vy", type=float, default=0.4)
ap.add_argument("--wz", type=float, default=1.2)
ap.add_argument("--h", type=float, nargs=2, default=[0.12, 0.36])
ap.add_argument("--roll", type=float, default=0.5, help="max |roll| command")
ap.add_argument("--smooth", type=float, default=1.0, help="per-step fraction toward the new command (1 = none)")
args = ap.parse_args()
LO = np.array([args.vx[0], -args.vy, -args.wz, args.h[0], -args.roll], np.float32)
HI = np.array([args.vx[1], args.vy, args.wz, args.h[1], args.roll], np.float32)

pol = HipanNavPolicy(args.weights)
rclpy.init()
node = rclpy.create_node("hipan_nav")
S = dict(odom=None, odom_t=0.0, depth=None, depth_t=0.0, dbg=None, dbg_t=0.0, K=None, goal=None)


def on_odom(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    S["odom"], S["odom_t"] = (p.x, p.y, yaw), time.time()


def on_depth(m):
    S["depth"], S["depth_t"] = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.width), time.time()


def on_dbg(m):
    S["dbg"], S["dbg_t"] = np.array(m.data, np.float32), time.time()


def on_info(m):
    if S["K"] is None:
        S["K"] = (list(m.k), m.width, m.height)


def on_goal(m):
    S["goal"] = (m.pose.position.x, m.pose.position.y)
    pol.reset()
    node.get_logger().info("new goal from rviz: (%.2f, %.2f)" % S["goal"])


node.create_subscription(Odometry, "/Odometry", on_odom, 10)
node.create_subscription(Image, args.depth_topic, on_depth, 2)
node.create_subscription(Float32MultiArray, "/hipan/debug", on_dbg, 10)
node.create_subscription(CameraInfo, "/camera/depth/camera_info", on_info, 2)
node.create_subscription(PoseStamped, "/goal_pose", on_goal, 2)
pub = node.create_publisher(Float32MultiArray, "/hipan/cmd" if args.live else "/hipan/cmd_shadow", 10)

t0 = time.time()
while (S["K"] is None or S["odom"] is None or S["dbg"] is None) and time.time() - t0 < 10:
    rclpy.spin_once(node, timeout_sec=0.05)
if S["K"] is None or S["odom"] is None or S["dbg"] is None:
    raise SystemExit("missing input: camera_info %s odometry %s /hipan/debug %s" % (
        S["K"] is not None, S["odom"] is not None, S["dbg"] is not None))
K, w, h = S["K"]
conv = DepthConverter(K, w, h, real_pitch_deg=args.real_pitch)
if args.goal:
    px, py, yaw = S["odom"]
    gx, gy = args.goal
    S["goal"] = (px + np.cos(yaw) * gx - np.sin(yaw) * gy, py + np.sin(yaw) * gx + np.cos(yaw) * gy)
node.get_logger().info("%s  depth %dx%d -> 64x112 (pad %d rows)  goal %s" % (
    "LIVE -> /hipan/cmd" if args.live else "SHADOW -> /hipan/cmd_shadow", w, h, conv.pad_rows, S["goal"]))

stand = pol.c_nom.copy()
c_prev = stand.copy()
log = {k: [] for k in ("t", "depth", "x", "a", "raw_cmd", "cmd", "pose", "dist", "state")}
next_t = time.time()
n = 0
try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.005)
        if time.time() < next_t:
            continue
        next_t += 0.1
        now = time.time()
        ok = (now - S["odom_t"] < 0.4) and (now - S["depth_t"] < 0.4) and (now - S["dbg_t"] < 0.2)
        low_state = int(S["dbg"][0]) if S["dbg"] is not None else -1
        if not ok or low_state != 3 or S["goal"] is None:
            # not walking / stale input / no goal: stand still, clear the memory
            if not ok and n % 10 == 0:
                node.get_logger().warn("stale input: odom %.2fs depth %.2fs low %.2fs" % (
                    now - S["odom_t"], now - S["depth_t"], now - S["dbg_t"]))
            pol.reset()
            c_prev = stand.copy()
            pub.publish(Float32MultiArray(data=stand.tolist()))
            n += 1
            continue
        px, py, yaw = S["odom"]
        gf, dist = goal_feature(S["goal"][0], S["goal"][1], px, py, yaw)
        if dist < 0.25:
            node.get_logger().info("ARRIVED (%.2f m)" % dist)
            S["goal"] = None
            pol.reset()
            pub.publish(Float32MultiArray(data=stand.tolist()))
            continue
        d = S["dbg"]
        low_obs = d[18:80].copy()  # debug: state 1, cmd 5, est 5, rpy 3, contact 4, then obs 62
        low_obs[57:62] = c_prev  # the command being executed (= the nav's own previous command)
        est = d[6:11]
        x = np.concatenate([low_obs, gf, [args.v_des], est]).astype(np.float32)
        depth = conv(S["depth"])
        a = pol.step(depth, x)
        raw_cmd = pol.to_command(a)
        cmd = np.clip(c_prev + args.smooth * (np.clip(raw_cmd, LO, HI) - c_prev), LO, HI)
        pub.publish(Float32MultiArray(data=cmd.tolist()))
        c_prev = cmd.astype(np.float32)
        for k, v in (("t", now), ("depth", (depth * 255).astype(np.uint8)), ("x", x), ("a", a), ("raw_cmd", raw_cmd), ("cmd", cmd),
                     ("pose", [px, py, yaw]), ("dist", dist), ("state", low_state)):
            log[k].append(v)
        n += 1
        if n % 10 == 0:
            node.get_logger().info("goal %.2f m (%.2f, %.2f)  cmd vx %.2f vy %.2f wz %.2f h %.2f roll %.2f  (student wanted vx %.2f wz %.2f h %.2f)" % (
                dist, gf[0] * 5, gf[1] * 5, *cmd, raw_cmd[0], raw_cmd[2], raw_cmd[3]))
except KeyboardInterrupt:
    pass
finally:
    pub.publish(Float32MultiArray(data=stand.tolist()))
    out = Path(__file__).parent / "logs" / time.strftime("nav_%Y%m%d_%H%M%S.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, live=args.live, goal=np.array(S["goal"] or [np.nan, np.nan]),
             **{k: np.array(v) for k, v in log.items()})
    print("log ->", out)
    node.destroy_node()
    rclpy.shutdown()
