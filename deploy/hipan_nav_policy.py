"""HiPAN nav depth student (CNN + GRU) in numpy, plus the real D435i -> sim depth-image conversion.

Weights: sim/scripts/export_hipan_nav.py -> weights/hipan_nav_student.npz

    python3 hipan_nav_policy.py weights/hipan_nav_student.npz        # numpy vs torch replay + timing
"""
import json
import sys

import numpy as np


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def conv2d(x, w, b, stride, pad):
    """x [C, H, W], w [O, C, k, k] -> [O, H', W'] (torch Conv2d semantics, cross-correlation)."""
    C, H, W = x.shape
    O, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    s0, s1, s2 = xp.strides
    cols = np.lib.stride_tricks.as_strided(xp, shape=(C, k, k, Ho, Wo), strides=(s0, s1, s2, s1 * stride, s2 * stride))
    out = w.reshape(O, -1) @ cols.reshape(C * k * k, Ho * Wo)
    return (out + b[:, None]).reshape(O, Ho, Wo)


class HipanNavPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.W = {k: z[k].astype(np.float32) for k in z.files if not k.startswith(("test_", "meta"))}
        self._test = {k: z[k] for k in z.files if k.startswith("test_")}
        self.c_lo = np.array(self.meta["c_lo"], np.float32)
        self.c_hi = np.array(self.meta["c_hi"], np.float32)
        self.c_nom = np.array(self.meta["c_nom"], np.float32)
        self.hid = self.W["gru_weight_hh"].shape[1]
        self.reset()

    def reset(self):
        self.h = np.zeros(self.hid, np.float32)

    def _lin(self, x, n):
        return self.W[n + "_weight"] @ x + self.W[n + "_bias"]

    def step(self, depth, x):
        """depth [64, 112] in [0, 1]; x [71] -> action [5] (normalised, clip to [-1, 1] -> command)."""
        W = self.W
        xn = np.clip((x - W["norm_mean"]) / np.sqrt(W["norm_var"] + 1e-2), -5, 5)
        f = depth[None].astype(np.float32)
        f = elu(conv2d(f, W["cnn_0_weight"], W["cnn_0_bias"], 2, 2))
        f = elu(conv2d(f, W["cnn_2_weight"], W["cnn_2_bias"], 2, 1))
        f = elu(conv2d(f, W["cnn_4_weight"], W["cnn_4_bias"], 2, 1))
        f = elu(conv2d(f, W["cnn_6_weight"], W["cnn_6_bias"], 2, 1))
        f = elu(self._lin(f.reshape(-1), "cnn_9"))
        g = elu(self._lin(xn, "fc_0"))
        inp = np.concatenate([f, g])
        gi = W["gru_weight_ih"] @ inp + W["gru_bias_ih"]
        gh = W["gru_weight_hh"] @ self.h + W["gru_bias_hh"]
        H = self.hid
        r = sigmoid(gi[:H] + gh[:H])
        u = sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = ((1 - u) * n + u * self.h).astype(np.float32)
        zz = self._lin(self.h, "z")
        p = elu(self._lin(np.concatenate([self.h, zz, xn]), "pi_0"))
        p = elu(self._lin(p, "pi_2"))
        return self._lin(p, "pi_4")

    def to_command(self, a):
        a = np.clip(a, -1.0, 1.0)
        return (self.c_lo + self.c_hi) / 2 + a * (self.c_hi - self.c_lo) / 2

    def self_test(self):
        self.reset()
        err = max(np.abs(self.step(d, x) - a).max() for d, x, a in
                  zip(self._test["test_depth"], self._test["test_x"], self._test["test_a"]))
        print("numpy vs torch (30 GRU steps): action max err %.2e" % err)
        return err


class DepthConverter:
    """Real D435i depth (uint16 mm, any resolution) -> the sim student image: 64 x 112, 87 x 58 deg pinhole,
    camera pitched ``sim_pitch`` down. The real camera is pitched ``real_pitch`` down, so the sim image centre sits
    tan(sim - real) * fy pixels lower in the real image. Each output pixel = median of the valid real pixels in its
    block (4 x 4 samples); a block with no valid pixel = DEPTH_MAX (the sim maps no-hit rays to DEPTH_MAX)."""

    def __init__(self, K, width, height, hw=(64, 112), fov_deg=(87.0, 58.0), sim_pitch_deg=13.0,
                 real_pitch_deg=11.6, depth_max=4.0, samples=4):
        fx, cx, fy, cy = K[0], K[2], K[4], K[5]
        Ho, Wo = hw
        tx, ty = np.tan(np.radians(fov_deg[0] / 2)), np.tan(np.radians(fov_deg[1] / 2))
        dy = fy * np.tan(np.radians(sim_pitch_deg - real_pitch_deg))
        # sub-sample positions inside each output pixel
        o = (np.arange(samples) + 0.5) / samples
        uc = ((np.arange(Wo)[:, None] + o[None]) / Wo * 2 - 1).reshape(-1)  # [-1, 1]
        vc = ((np.arange(Ho)[:, None] + o[None]) / Ho * 2 - 1).reshape(-1)
        cols = np.clip(np.round(cx + uc * tx * fx), 0, width - 1).astype(int).reshape(Wo, samples)
        rows = np.clip(np.round(cy + dy + vc * ty * fy), 0, height - 1).astype(int).reshape(Ho, samples)
        self.rows, self.cols = rows, cols
        self.hw, self.s, self.dmax = hw, samples, depth_max
        self.pad_rows = int(((cy + dy + ty * fy) - (height - 1)).clip(min=0))

    def __call__(self, depth_mm):
        d = depth_mm.astype(np.float32) * 1e-3
        d[d <= 0.05] = np.nan
        Ho, Wo = self.hw
        blk = d[self.rows[:, :, None, None], self.cols[None, None, :, :]]  # [Ho, s, Wo, s]
        blk = blk.transpose(0, 2, 1, 3).reshape(Ho, Wo, -1)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(blk, axis=2)
        m = np.nan_to_num(m, nan=self.dmax)
        return (np.clip(m, 0, self.dmax) / self.dmax).astype(np.float32)


def goal_feature(gx, gy, px, py, yaw):
    """Goal (world) relative to the robot in its yaw frame: (x, y) clipped to 5 m / 5, distance / 10 (sim goal_feat)."""
    dx, dy = gx - px, gy - py
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = c * dx + s * dy, -s * dx + c * dy
    dist = max(np.hypot(x, y), 1e-6)
    k = min(dist, 5.0) / dist / 5.0
    return np.array([x * k, y * k, min(dist, 10.0) / 10.0], np.float32), dist


if __name__ == "__main__":
    import time

    p = HipanNavPolicy(sys.argv[1] if len(sys.argv) > 1 else "weights/hipan_nav_student.npz")
    p.self_test()
    d, x = np.random.rand(64, 112).astype(np.float32), np.random.randn(71).astype(np.float32)
    t0 = time.time()
    for _ in range(20):
        p.step(d, x)
    print("step %.1f ms" % ((time.time() - t0) / 20 * 1e3))
    conv = DepthConverter([427.3, 0, 423.4, 0, 427.3, 242.8, 0, 0, 1], 848, 480)
    img = (np.random.rand(480, 848) * 3000 + 500).astype(np.uint16)
    t0 = time.time()
    for _ in range(10):
        conv(img)
    print("depth convert %.1f ms, bottom rows padded by edge: %d px" % ((time.time() - t0) / 10 * 1e3, conv.pad_rows))
