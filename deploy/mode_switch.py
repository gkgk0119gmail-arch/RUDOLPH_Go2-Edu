"""Go2 motion switcher: check / release / restore Unitree's built-in controller (sport mode).

    python3 mode_switch.py check                  # which controller is active (safe, read only)
    python3 mode_switch.py release --yes          # turn sport mode OFF -> the robot goes LIMP (hang it or lay it down!)
    python3 mode_switch.py select mcf             # turn sport mode back ON (name from `check`; robot lying down)

/lowcmd from go2_hipan_low (mode:=live) only takes effect after `release`.
"""
import argparse
import json
import random
import time

import rclpy
from unitree_api.msg import Request, Response

API = {"check": 1001, "select": 1002, "release": 1003}

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=list(API))
ap.add_argument("name", nargs="?", default="mcf", help="controller name as printed by check (this Go2: mcf)")
ap.add_argument("--yes", action="store_true", help="required for release")
args = ap.parse_args()
if args.action == "release" and not args.yes:
    raise SystemExit("release makes the robot go limp. Hang it up or lay it down, then rerun with --yes")

rclpy.init()
node = rclpy.create_node("mode_switch")
got = {}
rid = random.randint(1, 2**31)
node.create_subscription(Response, "/api/motion_switcher/response",
                         lambda m: got.__setitem__("r", m) if m.header.identity.id == rid else None, 10)
pub = node.create_publisher(Request, "/api/motion_switcher/request", 10)
time.sleep(0.5)
req = Request()
req.header.identity.id = rid
req.header.identity.api_id = API[args.action]
if args.action == "select":
    req.parameter = json.dumps({"name": args.name})
t0 = time.time()
sent = 0
while "r" not in got and time.time() - t0 < 5:
    if time.time() - t0 > sent * 1.0:
        pub.publish(req)
        sent += 1
    rclpy.spin_once(node, timeout_sec=0.05)
if "r" not in got:
    raise SystemExit("no response from /api/motion_switcher (is the Go2 on and DDS visible?)")
r = got["r"]
print("code", r.header.status.code, "data", r.data or "(none)")
if args.action == "check" and r.data:
    d = json.loads(r.data)
    print("active controller:", repr(d.get("name")), "(empty = released, low-level control possible)")
node.destroy_node()
rclpy.shutdown()
