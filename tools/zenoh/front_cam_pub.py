"""Go2 built-in head camera -> small ROS image for the laptop RViz (NX, ROS2 foxy, python 3.8).

    python3 front_cam_pub.py [--hz 2] [--width 480]        # -> /go2/front/image (sensor_msgs/Image, bgr8)

The main board multicasts the head camera as RTP/H264 on 230.1.1.1:1720 (1280x720, ~14.5 fps). Jetson hardware
decoder (nvv4l2decoder) + scaler (nvvidconv) do the work, so the CPU only copies a 480x270 frame; we publish
--hz frames per second. remote.sh bridge start/stop runs it next to the Zenoh bridge (cores 0-3, nice 10).
Other readers of the same multicast (tools/panel/rec.sh recording) are not affected.
"""
import argparse
import subprocess
import time

import gi
import numpy as np
import rclpy
from sensor_msgs.msg import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hz", type=float, default=2.0)
ap.add_argument("--width", type=int, default=480)
ap.add_argument("--topic", default="/go2/front/image")
ap.add_argument("--keyframes", type=int, default=0,
                help="1: decode key frames only (NX CPU 5.5 %% of a core, ~0.5 Hz); 0: all frames (22 %%, ~1.4 Hz)")
args = ap.parse_args()
W = args.width // 2 * 2
H = int(round(W * 9 / 16)) // 2 * 2


def robot_iface():
    out = subprocess.run(["ip", "-4", "-o", "addr"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    for line in out.splitlines():
        if " 192.168.123." in line:
            return line.split()[1]
    return "eth0"


rclpy.init()
node = rclpy.create_node("go2_front_cam")
pub = node.create_publisher(Image, args.topic, 2)
Gst.init(None)
pipe = Gst.parse_launch(
    "udpsrc address=230.1.1.1 port=1720 multicast-iface=%s "
    "caps=\"application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96\" "
    "! rtph264depay ! h264parse ! nvv4l2decoder skip-frames=%s ! videorate drop-only=true max-rate=%d ! nvvidconv "
    "! video/x-raw,format=BGRx,width=%d,height=%d "
    "! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    % (robot_iface(), "2" if args.keyframes else "0", max(1, int(round(args.hz))), W, H))
sink = pipe.get_by_name("sink")
pipe.set_state(Gst.State.PLAYING)
node.get_logger().info("Go2 head camera -> %s %dx%d @ %.1f Hz" % (args.topic, W, H, args.hz))

period, last, n = 1.0 / args.hz, 0.0, 0
try:
    while rclpy.ok():
        sample = sink.emit("try-pull-sample", int(0.5 * Gst.SECOND))
        if sample is None:
            if n == 0 and time.time() - last > 5:
                node.get_logger().warn("no camera frames yet (Go2 on? multicast on %s)" % robot_iface())
                last = time.time()
            continue
        now = time.time()
        if now - last < 0.7 * period:     # videorate already thins to ~hz; this only guards bursts
            continue
        last = now
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            continue
        try:
            raw = np.frombuffer(info.data, np.uint8)
            stride = raw.size // H                      # nvvidconv may pad rows
            bgr = raw[:H * stride].reshape(H, stride)[:, :W * 4].reshape(H, W, 4)[:, :, :3].copy()
        finally:
            buf.unmap(info)
        msg = Image()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "go2_front_camera"
        msg.height, msg.width, msg.encoding, msg.step = H, W, "bgr8", W * 3
        msg.data = bgr.tobytes()
        pub.publish(msg)
        n += 1
except KeyboardInterrupt:
    pass
finally:
    pipe.set_state(Gst.State.NULL)
    node.destroy_node()
    rclpy.shutdown()
