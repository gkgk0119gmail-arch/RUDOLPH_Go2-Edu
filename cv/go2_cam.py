"""Go2 built-in head camera reader (no ROS).

The Go2 main board multicasts the head camera as RTP/H264 on 230.1.1.1:1720 (1280x720, ~14.5 fps).
GStreamer only depayloads RTP -> H264 byte stream on stdout (plugins-good is enough), PyAV decodes it.
Works on the desktop (iface enx68e43b307a52, 192.168.123.99) and on the NX (iface eth0).

    cam = Go2Camera()            # starts a reader thread
    t, frame = cam.read()        # latest BGR frame (np.uint8 HxWx3) and its receive time
    cam.close()
"""
from __future__ import annotations

import subprocess
import threading
import time

import av
import numpy as np

GROUP, PORT = "230.1.1.1", 1720


def default_iface() -> str:
    """The interface on the robot LAN (192.168.123.x)."""
    out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if " 192.168.123." in line:
            return line.split()[1]
    return "eth0"


class Go2Camera:
    def __init__(self, iface: str | None = None):
        self.iface = iface or default_iface()
        cmd = ["gst-launch-1.0", "-q", "udpsrc", f"address={GROUP}", f"port={PORT}", f"multicast-iface={self.iface}",
               "!", "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96",
               "!", "rtph264depay", "!", "video/x-h264,stream-format=byte-stream,alignment=au", "!", "fdsink", "fd=1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.codec = av.CodecContext.create("h264", "r")
        self.lock = threading.Lock()
        self.frame, self.t, self.n = None, 0.0, 0
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while self.running:
            data = self.proc.stdout.read(65536)
            if not data:
                break
            try:
                packets = self.codec.parse(data)
            except (av.FFmpegError, ValueError):
                continue
            for p in packets:
                try:
                    for f in self.codec.decode(p):
                        img = f.to_ndarray(format="bgr24")
                        with self.lock:
                            self.frame, self.t, self.n = img, time.time(), self.n + 1
                except (av.FFmpegError, ValueError):          # missing reference frames right after start
                    pass

    def read(self, timeout: float = 5.0, newer_than: int = -1):
        """Latest frame as (recv_time, bgr). Waits until a frame with index > newer_than exists."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.n > newer_than:
                    return self.t, self.frame
            time.sleep(0.005)
        raise TimeoutError(f"no camera frame on {self.iface} (is the Go2 on and the LAN cable in?)")

    def read_new(self, last_n: int, timeout: float = 5.0):
        """Like read(), but also returns the frame index so callers can skip frames already seen."""
        t, img = self.read(timeout, newer_than=last_n)
        return self.n, t, img

    def close(self):
        self.running = False
        self.proc.terminate()
        try:
            self.proc.wait(2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


if __name__ == "__main__":
    cam = Go2Camera()
    n0, t0 = None, time.time()
    _, img = cam.read()
    n0 = cam.n
    time.sleep(5)
    print(f"iface {cam.iface}  frame {img.shape}  {(cam.n - n0) / (time.time() - t0):.1f} fps")
    cam.close()
