"""Publish one clue / victim event to rudolph_search.py (NX, ROS2 foxy). Called by remote.sh, not by hand.

    printf '%s' "로봇 오른쪽 5미터에 한 명" | python3 send_clue.py clue     # -> /rudolph/clue_text
    python3 send_clue.py victim                                           # -> /rudolph/victim_here

Waits up to 3 s for the search node to subscribe, so the message is not lost when it is the only publisher.
"""
import sys
import time

import rclpy
from std_msgs.msg import Empty, String

kind = sys.argv[1] if len(sys.argv) > 1 else "clue"
rclpy.init()
node = rclpy.create_node("rudolph_panel_send")
if kind == "clue":
    text = sys.stdin.read().strip()
    pub, msg = node.create_publisher(String, "/rudolph/clue_text", 10), String(data=text)
else:
    pub, msg = node.create_publisher(Empty, "/rudolph/victim_here", 10), Empty()
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)
if pub.get_subscription_count() == 0:
    print("✗ 수색 노드(rudolph_search)가 없습니다 — 보내지 않았습니다")
    sys.exit(1)
pub.publish(msg)
time.sleep(0.3)  # let DDS deliver before the node goes away
print("✓ 보냄: " + (text if kind == "clue" else "요구조자 발견 (현재 로봇 위치)"))
node.destroy_node()
rclpy.shutdown()
