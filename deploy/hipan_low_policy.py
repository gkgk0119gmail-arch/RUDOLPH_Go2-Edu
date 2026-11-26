"""HiPAN low-level student on the real Go2: numpy-only inference (Python 3.8 / numpy 1.17 on the Jetson NX).

Weights + sim settings come from sim/scripts/export_hipan_low.py (hipan_low_student.npz). This module only does:
  * the policy forward pass (MLP: 50-step history -> 12 joint actions + 5 motion-state estimates)
  * the 62-dim observation, built exactly like the sim ``student`` group
  * joint order mapping between the policy (Isaac: hips, thighs, calves; FL FR RL RR) and the Unitree SDK
    motor order (FR, FL, RR, RL legs; hip, thigh, calf in each)

    python3 hipan_low_policy.py hipan_low_student.npz     # self-test: numpy vs torch actions, FK vs sim feet
"""
import json
import sys

import numpy as np

# policy (Isaac) joint i  <->  Unitree SDK motor index
LEGS_SDK = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
ISAAC_LEGS = ["FL", "FR", "RL", "RR"]
ISAAC_TO_SDK = np.array([LEGS_SDK[ISAAC_LEGS[i % 4]] * 3 + i // 4 for i in range(12)])  # policy i -> sdk motor
SDK_TO_ISAAC = np.argsort(ISAAC_TO_SDK)

# leg geometry (unitree go2 URDF)
HIP_X, HIP_Y, THIGH_Y, L1, L2 = 0.1934, 0.0465, 0.0955, 0.213, 0.213
SIGN = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def feet_pos_body(q):
    """q: 12 joint angles in policy order -> [12] foot positions in the body frame, policy leg order (x, y, z each)."""
    out = []
    for k, leg in enumerate(ISAAC_LEGS):
        sx, sy = SIGN[leg]
        qh, qt, qc = q[k], q[4 + k], q[8 + k]
        # sagittal chain in the thigh frame (rotation about y): thigh then calf
        x = -L1 * np.sin(qt) - L2 * np.sin(qt + qc)
        z = -L1 * np.cos(qt) - L2 * np.cos(qt + qc)
        y = sy * THIGH_Y
        # hip roll about x
        c, s = np.cos(qh), np.sin(qh)
        out += [sx * HIP_X + x, sy * HIP_Y + c * y - s * z, s * y + c * z]
    return np.array(out, np.float32)


def roll_pitch_from_quat(qw, qx, qy, qz):
    """IMU quaternion (w, x, y, z) -> (roll, pitch), same convention as the sim roll_pitch term (ZYX Euler)."""
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return roll, pitch


class HipanLowPolicy:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.meta = json.loads(str(z["meta"]))
        self.W = {k: z[k].astype(np.float32) for k in z.files if k.endswith("_weight") or k.endswith("_bias")}
        self.h, self.d = self.meta["history"], self.meta["obs_dim"]
        self.default_q = np.array(self.meta["default_joint_pos"], np.float32)
        self.scale = np.array(self.meta["action_scale"], np.float32)
        self.offset = np.array(self.meta["action_offset"], np.float32)
        self.kp = np.array(self.meta["kp"], np.float32)
        self.kd = np.array(self.meta["kd"], np.float32)
        self.buf = None
        self.last_action = np.zeros(12, np.float32)
        self._test = {k: z[k] for k in ("test_obs", "test_act", "test_est") if k in z.files}

    def _lin(self, x, name):
        return x @ self.W[name + "_weight"].T + self.W[name + "_bias"]

    def forward(self, hist):
        """hist: [H, D] (newest last) -> action [12], estimate [5]."""
        x = hist.reshape(-1)
        e = elu(self._lin(x, "enc_0"))
        e = elu(self._lin(e, "enc_2"))
        zz = self._lin(e, "enc_4")
        m = self._lin(zz, "motion")
        p = np.concatenate([zz, m, hist[-1]])
        p = elu(self._lin(p, "pi_0"))
        p = elu(self._lin(p, "pi_2"))
        p = elu(self._lin(p, "pi_4"))
        return self._lin(p, "pi_6"), m

    def reset(self):
        self.buf = None
        self.last_action[:] = 0

    def build_obs(self, q, dq, contact, roll, pitch, gyro, cmd):
        """All in policy (Isaac) joint order. q, dq: [12] rad, rad/s; contact: [4] 0/1 (FL FR RL RR);
        gyro: [3] body rad/s; cmd: [vx, vy, wz, h, roll]."""
        return np.concatenate([
            q - self.default_q, dq * 0.05, feet_pos_body(q), contact, [roll, pitch], np.asarray(gyro) * 0.25,
            self.last_action, cmd]).astype(np.float32)

    def step(self, obs):
        """obs [62] -> joint position targets [12] (policy order), estimate [5]."""
        if self.buf is None:
            self.buf = np.repeat(obs[None], self.h, 0)
        else:
            self.buf = np.concatenate([self.buf[1:], obs[None]], 0)
        a, m = self.forward(self.buf)
        self.last_action = a.astype(np.float32)
        return self.offset + self.scale * a, m

    # ---- self-test against the sim replay stored in the npz
    def self_test(self):
        obs, act, est = self._test["test_obs"], self._test["test_act"], self._test["test_est"]
        self.reset()
        da, dm = [], []
        for t in range(len(obs)):
            self.last_action = obs[t, 45:57]  # use the sim's own previous action so inputs are identical
            if self.buf is None:
                self.buf = np.repeat(obs[t][None], self.h, 0)
            else:
                self.buf = np.concatenate([self.buf[1:], obs[t][None]], 0)
            a, m = self.forward(self.buf)
            da.append(np.abs(a - act[t]).max())
            dm.append(np.abs(m - est[t]).max())
        q = obs[:, 0:12] + self.default_q
        fk = np.array([feet_pos_body(x) for x in q])
        dfk = np.abs(fk - obs[:, 24:36])
        print("numpy vs torch  action max err %.2e   estimate max err %.2e" % (max(da), max(dm)))
        print("FK vs sim feet  mean err %.4f m   max err %.4f m (sim obs has +-0.01 noise)" % (dfk.mean(), dfk.max()))
        print("joint map policy->sdk", ISAAC_TO_SDK.tolist())
        return max(da), dfk.mean()


if __name__ == "__main__":
    import time

    p = HipanLowPolicy(sys.argv[1] if len(sys.argv) > 1 else "hipan_low_student.npz")
    p.self_test()
    hist = np.random.randn(p.h, p.d).astype(np.float32)
    t0 = time.time()
    for _ in range(200):
        p.forward(hist)
    print("forward %.2f ms" % ((time.time() - t0) / 200 * 1e3))
