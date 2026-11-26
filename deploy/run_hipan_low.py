"""Run the HiPAN low-level student on the real Go2 (ROS2 Foxy, NX).

Default is SHADOW mode (--live not given): reads /lowstate, builds the observation, runs the policy at 50 Hz and logs
everything, but publishes NOTHING. The robot stays under Unitree's own controller. Use it to check inputs (joint
order, IMU, contacts) and outputs (targets vs the current pose) before any live run.

    source ~/Desktop/3DOG/env.sh
    python3 run_hipan_low.py --seconds 10 --cmd 0 0 0 0.30 0          # shadow, logs to logs/shadow_<time>.npz

Live mode (joint commands on /lowcmd) is deliberately not implemented yet: it needs the sport-mode release, the
LowCmd CRC and a damping fallback, and should be tested with the robot hung up.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from unitree_go.msg import LowState

from hipan_low_policy import ISAAC_TO_SDK, HipanLowPolicy, roll_pitch_from_quat

FOOT_SDK_TO_POLICY = [1, 0, 3, 2]  # foot_force order FR FL RR RL -> policy FL FR RL RR

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(Path(__file__).parent / "weights/hipan_low_student.npz"))
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--cmd", type=float, nargs=5, default=[0.0, 0.0, 0.0, 0.30, 0.0], help="vx vy wz height roll")
ap.add_argument("--contact_thr", type=float, default=20.0, help="raw foot_force above this = contact (to calibrate)")
ap.add_argument("--consistent", action="store_true",
                help="shadow check: feed last_action = the action that holds the current pose (robot not following)")
ap.add_argument("--live", action="store_true")
args = ap.parse_args()
if args.live:
    raise SystemExit("live mode not implemented yet (shadow only)")

pol = HipanLowPolicy(args.weights)
cmd = np.array(args.cmd, np.float32)
rclpy.init()
node = rclpy.create_node("hipan_low_shadow")
latest = {}
node.create_subscription(LowState, "/lowstate", lambda m: latest.__setitem__("m", m), 10)

log = {k: [] for k in ("t", "obs", "action", "target", "q", "est", "foot_raw")}
t0 = time.time()
next_t = t0
n = 0
while time.time() - t0 < args.seconds:
    rclpy.spin_once(node, timeout_sec=0.002)
    if "m" not in latest or time.time() < next_t:
        continue
    next_t += 0.02
    m = latest["m"]
    q_sdk = np.array([x.q for x in m.motor_state[:12]], np.float32)
    dq_sdk = np.array([x.dq for x in m.motor_state[:12]], np.float32)
    q, dq = q_sdk[ISAAC_TO_SDK], dq_sdk[ISAAC_TO_SDK]
    raw = np.array(m.foot_force, np.float32)[FOOT_SDK_TO_POLICY]
    contact = (raw > args.contact_thr).astype(np.float32)
    w, x, y, z = m.imu_state.quaternion
    roll, pitch = roll_pitch_from_quat(w, x, y, z)
    gyro = np.array(m.imu_state.gyroscope, np.float32)
    if args.consistent:
        pol.last_action = ((q - pol.offset) / pol.scale).astype(np.float32)
    obs = pol.build_obs(q, dq, contact, roll, pitch, gyro, cmd)
    target, est = pol.step(obs)
    for k, v in (("t", time.time() - t0), ("obs", obs), ("action", pol.last_action.copy()), ("target", target),
                 ("q", q), ("est", est), ("foot_raw", raw)):
        log[k].append(v)
    n += 1
    if n % 50 == 0:
        print("t %.1fs  |target-q| max %.3f mean %.3f rad  est v=(%.2f %.2f) h=%.3f roll=%.3f  contact %s  raw %s" % (
            time.time() - t0, np.abs(target - q).max(), np.abs(target - q).mean(), est[0], est[1], est[3], est[4],
            contact.astype(int).tolist(), raw.astype(int).tolist()), flush=True)

out = Path(__file__).parent / "logs" / time.strftime("shadow_%Y%m%d_%H%M%S.npz")
out.parent.mkdir(exist_ok=True)
np.savez(out, cmd=cmd, joint_order="policy (Isaac): hips FL FR RL RR, thighs, calves",
         **{k: np.array(v) for k, v in log.items()})
dt = np.diff(np.array(log["t"]))
print("steps %d  rate %.1f Hz  dt max %.1f ms  -> %s" % (n, 1 / dt.mean(), dt.max() * 1e3, out))
node.destroy_node()
rclpy.shutdown()
