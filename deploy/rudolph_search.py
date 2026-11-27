"""RUDOLPH search node (NX, ROS2 foxy, python 3.8): text clue + FAST-LIO map/pose -> next search goal.

    python3 rudolph_search.py                      # SHADOW: goals go to /rudolph/goal (rviz only)
    python3 rudolph_search.py --live               # LIVE: goals also go to /goal_pose (run_hipan_nav.py walks there)
    ros2 topic pub --once /rudolph/clue_text std_msgs/String "data: '로봇 오른쪽으로 5미터쯤 한 명 있어요'"
    ros2 topic pub --once /rudolph/victim_here std_msgs/Empty "{}"          # a person found at the robot
    rviz "Publish Point" -> /clicked_point                                     # a person found where clicked

Direction clues are always in the ROBOT's frame (x forward, y left) at the moment the text arrives.
Pipeline: /cloud_registered (map frame) -> LiveGrid (0.2 m: floor / obstacle band) -> LiveSearch (camera footprint =
searched, clue probability maps, per-reporter Bayesian trust, frontier rate score) -> goal (PoseStamped, camera_init).
Clue text is parsed in a background thread by the fine-tuned Qwen3-VL-8B Q4 GGUF on the NX's llama-server (default,
prompt ft1), or by sim/clue/serve_parse.py on a GPU PC (--parser http://HOST:8765).
Publishes for rviz: /rudolph/map, /rudolph/searched, /rudolph/belief (OccupancyGrid), /rudolph/goal, /rudolph/status.
"""
import argparse
import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, String

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
from clue import schema as SCH  # noqa: E402
from clue.live import LiveGrid, LiveSearch  # noqa: E402
from clue.prompts import PROMPTS  # noqa: E402
from clue.search2d import CELL, FREE, OCC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true", help="also publish /goal_pose (the nav student walks there)")
ap.add_argument("--parser", default="llama:http://127.0.0.1:8080",
                help="clue parser: 'llama:URL' = llama-server with the fine-tuned GGUF (prompt ft1, NX), "
                     "or URL of sim/clue/serve_parse.py (POST /parse)")
ap.add_argument("--size", type=float, default=120.0, help="map square [m] around the start pose")
ap.add_argument("--n_unclued", type=int, default=1, help="people expected besides the ones clues mention")
ap.add_argument("--arrive", type=float, default=0.5, help="goal reached within this distance [m]")
ap.add_argument("--replan", type=float, default=10.0, help="replan at least this often [s]")
ap.add_argument("--frame", default="camera_init")
ap.add_argument("--log", default=str(Path(__file__).parent / "logs"))
args = ap.parse_args()

rclpy.init()
node = rclpy.create_node("rudolph_search")
log = node.get_logger()
S = dict(pose=None, pose_t=0.0, grid=None, search=None, goal=None, goal_t=0.0, dirty=True, n_scans=0, msgs=[],
         trail=[], goals=[])
lock = threading.Lock()
Path(args.log).mkdir(parents=True, exist_ok=True)
event_log = open(Path(args.log) / time.strftime("rudolph_%Y%m%d_%H%M%S.jsonl"), "a")


def record(kind, **kw):
    event_log.write(json.dumps({"t": time.time(), "kind": kind, **kw}, ensure_ascii=False) + "\n")
    event_log.flush()


def status(msg):
    log.info(msg)
    S["msgs"] = (S["msgs"] + [time.strftime("%H:%M:%S ") + msg])[-8:]
    pub_status.publish(String(data="\n".join(S["msgs"])))


# -- inputs ----------------------------------------------------------------------------------------------------
def on_odom(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    with lock:
        S["pose"], S["pose_t"] = (p.x, p.y, p.z, yaw), time.time()
        if not S["trail"] or math.hypot(p.x - S["trail"][-1][0], p.y - S["trail"][-1][1]) > 0.1:
            S["trail"].append((p.x, p.y))
        if S["grid"] is None:
            S["grid"] = LiveGrid(size_m=args.size, center_xy=(p.x, p.y))
            S["search"] = LiveSearch((S["grid"].n, S["grid"].n), n_unclued=args.n_unclued)
            status("map started at (%.1f, %.1f), %.0f m square" % (p.x, p.y, args.size))


def cloud_xyz(m):
    off = {f.name: f.offset for f in m.fields}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [off["x"], off["y"], off["z"]], "itemsize": m.point_step})
    a = np.frombuffer(bytes(m.data), dt, count=m.width * m.height)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1)
    return xyz[np.isfinite(xyz).all(1)]


def on_cloud(m):
    with lock:
        if S["grid"] is None or S["pose"] is None or not S["grid"].inside(*S["pose"][:2]):
            return
        S["grid"].add_scan(cloud_xyz(m), S["pose"][:3])
        S["n_scans"] += 1


def ask_parser(text):
    """-> parsed clue dict (schema-normalized)."""
    if args.parser.startswith("llama:"):     # llama-server (OpenAI chat API), fine-tuned parser, short reply
        body = json.dumps({"messages": [{"role": "user", "content": PROMPTS["ft1"].format(u=text)}],
                           "temperature": 0, "max_tokens": 200}).encode()
        req = urllib.request.Request(args.parser[6:] + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        raw = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
        parsed, _ = SCH.normalize(SCH.extract_json(raw))
        if parsed is None:
            raise ValueError("no JSON in reply: %r" % raw[:80])
        return parsed
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(args.parser + "/parse", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["parsed"]


def parse_and_add(text, pose_rc, head):
    t = time.time()
    try:
        parsed = ask_parser(text)
    except Exception as e:  # noqa: BLE001  (network / service down: keep searching without the clue)
        status("parser failed (%s): clue ignored: %s" % (e, text))
        record("parse_fail", text=text, error=str(e))
        return
    with lock:
        msg = S["search"].add_clue(parsed, text=text, anchor=(pose_rc, head))
        S["dirty"] = True
    record("clue", text=text, parsed=parsed, latency=time.time() - t, result=msg)
    status("clue (%.1fs): %s -> %s" % (time.time() - t, text, msg))


def on_clue(m):
    with lock:
        if S["search"] is None:
            status("no map yet: clue ignored")
            return
        pose_rc, head = S["search"].pos, S["search"].head      # robot pose when the sentence came in
    status("heard: %s (parsing...)" % m.data)
    threading.Thread(target=parse_and_add, args=(m.data, pose_rc, head), daemon=True).start()


def found_at(x, y):
    with lock:
        if S["search"] is None:
            return
        r, c = S["grid"].cell(x, y)
        msg = S["search"].victim_found((int(r), int(c)))
        S["dirty"] = True
    record("victim", x=x, y=y, result=msg)
    status(msg)


node.create_subscription(Odometry, "/Odometry", on_odom, 10)
node.create_subscription(PointCloud2, "/cloud_registered", on_cloud, 5)
node.create_subscription(String, "/rudolph/clue_text", on_clue, 10)
node.create_subscription(Empty, "/rudolph/victim_here", lambda m: found_at(*S["pose"][:2]) if S["pose"] else None, 10)
node.create_subscription(PointStamped, "/clicked_point", lambda m: found_at(m.point.x, m.point.y), 10)
pub_goal = node.create_publisher(PoseStamped, "/rudolph/goal", 2)
pub_nav = node.create_publisher(PoseStamped, "/goal_pose", 2) if args.live else None
pub_map = node.create_publisher(OccupancyGrid, "/rudolph/map", 1)
pub_seen = node.create_publisher(OccupancyGrid, "/rudolph/searched", 1)
pub_bel = node.create_publisher(OccupancyGrid, "/rudolph/belief", 1)
pub_status = node.create_publisher(String, "/rudolph/status", 2)


def grid_msg(data_rows_up):
    """Our grid has row 0 at the top (+y); OccupancyGrid row 0 is at origin (lowest y)."""
    g = S["grid"]
    m = OccupancyGrid()
    m.header.frame_id = args.frame
    m.header.stamp = node.get_clock().now().to_msg()
    m.info.resolution = CELL
    m.info.width = m.info.height = g.n
    m.info.origin.position.x, m.info.origin.position.y = g.x0, g.y1 - g.n * CELL
    m.data = np.flipud(data_rows_up).astype(np.int8).ravel().tolist()
    return m


def publish_goal(rc):
    g, (px, py, _, _) = S["grid"], S["pose"]
    x, y = g.world(*rc)
    msg = PoseStamped()
    msg.header.frame_id = args.frame
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x, msg.pose.position.y = float(x), float(y)
    yaw = math.atan2(y - py, x - px)
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    pub_goal.publish(msg)
    if pub_nav is not None:
        pub_nav.publish(msg)
    return float(x), float(y)


def cycle():
    with lock:
        if S["search"] is None or time.time() - S["pose_t"] > 1.0:
            return
        g, s = S["grid"], S["search"]
        px, py, _, yaw = S["pose"]
        if not g.inside(px, py):
            status("robot outside the %.0f m map square: restart with a larger --size" % args.size)
            return
        s.set_map(g.known())
        s.set_pose(g.cell(px, py), yaw)
        s.look()
        goal = S["goal"]
        reached = goal is not None and math.hypot(goal[0] - px, goal[1] - py) < args.arrive
        stale = goal is not None and time.time() - S["goal_t"] > args.replan
        gone = goal is not None and s.searched[g.cell(*goal)]          # the goal cell has been looked at already
        if not (S["dirty"] or goal is None or reached or stale or gone):
            return
        tgt, _ = s.next_goal()
        S["dirty"] = False
        if tgt is None:
            if goal is not None:
                status("nothing reachable left unsearched: search done (%d/%d found)" % (
                    sum(f is not None for f in s.found), len(s.found)))
            S["goal"] = None
            return
        S["goal"], S["goal_t"] = publish_goal(tgt), time.time()
        S["goals"].append(S["goal"])
        record("goal", x=S["goal"][0], y=S["goal"][1], why="reached" if reached else "stale" if stale else
               "looked" if gone else "event")


def publish_maps():
    with lock:
        if S["search"] is None:
            return
        k = S["search"].known
        occ = np.where(k == OCC, 100, np.where(k == FREE, 0, -1))
        pub_map.publish(grid_msg(occ))
        pub_seen.publish(grid_msg(np.where(S["search"].searched, 100, -1)))
        h = S["search"].heat()
        pub_bel.publish(grid_msg(np.where(h > 0, np.clip(h / max(h.max(), 1e-12) * 100, 1, 100), -1)))
        g = S["grid"]
        np.savez_compressed(Path(args.log) / "rudolph_state.npz", known=k, searched=S["search"].searched, heat=h,
                            trail=np.array(S["trail"]), goals=np.array(S["goals"]), x0=g.x0, y1=g.y1, cell=CELL,
                            clues=json.dumps([dict(c, anchor_rc=[int(v) for v in c["anchor_rc"]])
                                              for c in S["search"].clues], ensure_ascii=False))


node.create_timer(0.5, cycle)
node.create_timer(2.0, publish_maps)
status("RUDOLPH search %s, parser %s" % ("LIVE -> /goal_pose" if args.live else "SHADOW", args.parser))
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    event_log.close()
    node.destroy_node()
    rclpy.shutdown()
