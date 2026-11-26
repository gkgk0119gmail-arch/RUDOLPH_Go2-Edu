"""Command vs FAST-LIO measured velocity vs the policy's own estimate, per constant-command segment.
    python3 analyze_track.py logs/track_run1.npz"""
import sys

import numpy as np

z = np.load(sys.argv[1])
o, d = z["odom"], z["dbg"]
t_o, p, q = o[:, 0], o[:, 1:4], o[:, 4:8]  # q = w x y z


def rot(qw, qx, qy, qz):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


# central differences over +-2 samples (0.4 s at 10 Hz)
k = 2
vb, wz, tm = [], [], []
yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
for i in range(k, len(o) - k):
    dt = t_o[i + k] - t_o[i - k]
    vw = (p[i + k] - p[i - k]) / dt
    vb.append(rot(*q[i]).T @ vw)
    wz.append((yaw[i + k] - yaw[i - k]) / dt)
    tm.append(t_o[i])
vb, wz, tm = np.array(vb), np.array(wz), np.array(tm)
td = d[:, 0]
cmd, est = d[:, 2:7], d[:, 7:12]
m_vx, m_vy, m_wz = (np.interp(td, tm, vb[:, j]) for j in (0, 1)), None, None
mvx = np.interp(td, tm, vb[:, 0]); mvy = np.interp(td, tm, vb[:, 1]); mwz = np.interp(td, tm, wz)
mz = np.interp(td, t_o, p[:, 2])

# segments of constant non-zero command
key = np.round(cmd[:, :3] / 0.05).astype(int)
seg, start = [], 0
for i in range(1, len(d) + 1):
    if i == len(d) or (key[i] != key[start]).any():
        if i - start > 60 and np.abs(cmd[start, :3]).max() > 0.05:
            seg.append((start + 35, i))  # skip 0.7 s of acceleration
        start = i
print("%-24s %22s %22s %22s" % ("command [vx vy wz]", "measured (FAST-LIO)", "policy estimate", "measured/command"))
for a, b in seg:
    c = cmd[a:b, :3].mean(0)
    m = np.array([mvx[a:b].mean(), mvy[a:b].mean(), mwz[a:b].mean()])
    e = est[a:b, :3].mean(0)
    e[2] = np.nan
    j = np.argmax(np.abs(c))
    print("[%5.2f %5.2f %5.2f] %6.1fs   [%5.2f %5.2f %5.2f]   [%5.2f %5.2f   - ]      %4.0f %%" % (
        *c, (b - a) * 0.02, *m, e[0], e[1], 100 * m[j] / c[j]))
st = np.abs(cmd[:, :3]).max(1) < 0.01
print("standing: measured |v| %.3f m/s, |wz| %.3f rad/s, policy est v=(%.3f %.3f)" % (
    np.hypot(mvx[st], mvy[st]).mean(), np.abs(mwz[st]).mean(), est[st, 0].mean(), est[st, 1].mean()))
# estimate quality over the whole run
ev = np.c_[est[:, 0] - mvx, est[:, 1] - mvy]
print("policy velocity estimate error: vx bias %.3f rms %.3f | vy bias %.3f rms %.3f m/s" % (
    ev[:, 0].mean(), np.sqrt((ev[:, 0] ** 2).mean()), ev[:, 1].mean(), np.sqrt((ev[:, 1] ** 2).mean())))
print("roll: IMU mean %.3f  policy est mean %.3f  rms err %.3f rad" % (
    d[:, 12].mean(), est[:, 4].mean(), np.sqrt(((est[:, 4] - d[:, 12]) ** 2).mean())))
print("height: policy est mean %.3f (cmd %.2f)  FAST-LIO z range %.3f m (should be ~0 on flat floor)" % (
    est[:, 3].mean(), cmd[:, 3].mean(), mz.max() - mz.min()))

# ---- per-axis fit over all POLICY samples (analog stick: few constant segments)
P = d[:, 1] == 3
print("\nPOLICY samples %d (%.0f s)" % (P.sum(), P.sum() * 0.02))
meas = {0: mvx, 1: mvy, 2: mwz}
for j, name in enumerate(["vx", "vy", "wz"]):
    best = None
    for lag in range(0, 40, 2):  # measured lags the command (accel + odom smoothing)
        c = cmd[:-lag or None, j] if lag else cmd[:, j]
        mm = meas[j][lag:]
        sel = P[lag:] & (np.abs(c) > 0.12) & (np.abs(np.delete(cmd[:len(c), :3], j, 1)).max(1) < 0.08)
        if sel.sum() < 30:
            continue
        g = (c[sel] * mm[sel]).sum() / (c[sel] ** 2).sum()
        r = np.corrcoef(c[sel], mm[sel])[0, 1]
        if best is None or r > best[2]:
            best = (lag, g, r, sel.sum(), c[sel], mm[sel])
    if best is None:
        print("%s: not enough samples" % name)
        continue
    lag, g, r, n, c, mm = best
    bins = [(-1.3, -0.5), (-0.5, -0.12), (0.12, 0.5), (0.5, 1.3)]
    txt = []
    for lo, hi in bins:
        s = (c > lo) & (c <= hi)
        if s.sum() > 15:
            txt.append("cmd %+.2f -> %+.2f" % (c[s].mean(), mm[s].mean()))
    print("%s: measured = %.2f x command (corr %.2f, lag %.2f s, n %d)   %s" % (name, g, r, lag * 0.02, n, " | ".join(txt)))
# estimate vs measured only in POLICY
ev = np.c_[est[P, 0] - mvx[P], est[P, 1] - mvy[P]]
print("POLICY velocity estimate error: vx bias %+.3f rms %.3f | vy bias %+.3f rms %.3f m/s" % (
    ev[:, 0].mean(), np.sqrt((ev[:, 0] ** 2).mean()), ev[:, 1].mean(), np.sqrt((ev[:, 1] ** 2).mean())))
sv = P & (np.abs(cmd[:, 0]) > 0.2)
if sv.sum() > 20:
    print("  while walking fwd/back: est vx %.2f vs measured %.2f (ratio %.2f)" % (
        est[sv, 0].mean(), mvx[sv].mean(), (est[sv, 0] * mvx[sv]).sum() / (mvx[sv] ** 2).sum()))
print("POLICY height est %.3f (cmd %.2f), roll est %.3f vs IMU %.3f (rms err %.3f)" % (
    est[P, 3].mean(), cmd[P, 3].mean(), est[P, 4].mean(), d[P, 12].mean(), np.sqrt(((est[P, 4] - d[P, 12]) ** 2).mean())))
zP = mz[P]
print("FAST-LIO z while walking: std %.3f m, range %.3f m" % (zP.std(), zP.max() - zP.min()))
