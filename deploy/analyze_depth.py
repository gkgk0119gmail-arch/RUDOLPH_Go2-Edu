"""Real D435i depth vs the sim depth camera: floor-plane pose, invalid pixels, student-input rendering.
    python3 analyze_depth.py logs/depth_A_floor.npz [--png out.png]"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1])
ci = json.load(open("logs/depth_camera_info.json"))
fx, cx, fy, cy = ci["k"][0], ci["k"][2], ci["k"][4], ci["k"][5]
D = z["depth"].astype(np.float32) / 1000.0  # [n, 480, 848] m
dbg = z["dbg"]
H, W = D.shape[1:]
u, v = np.meshgrid(np.arange(W), np.arange(H))
med = np.median(np.where(D > 0, D, np.nan), axis=0)
valid = np.isfinite(med) & (med > 0.1)
print("frames %d  %dx%d  invalid pixels %.1f %% (median over frames)" % (len(D), W, H, 100 * (1 - valid.mean())))
inv_rows = 1 - valid.mean(1)
print("invalid by image band (top->bottom):", " ".join("%.0f%%" % (100 * inv_rows[i:i + H // 6].mean()) for i in range(0, H, H // 6)))
# temporal noise
st = np.nanstd(np.where(D > 0, D, np.nan), axis=0)
print("temporal noise (std over frames): median %.1f mm" % (1000 * np.nanmedian(st)))

# floor plane from the lower half (camera optical frame: x right, y down, z forward)
X = (u - cx) / fx * med
Y = (v - cy) / fy * med
P = np.stack([X, Y, med], -1)[valid & (v > H * 0.45)]
best, rng = None, np.random.default_rng(0)
for _ in range(300):
    s = P[rng.choice(len(P), 3, replace=False)]
    nrm = np.cross(s[1] - s[0], s[2] - s[0])
    if np.linalg.norm(nrm) < 1e-6:
        continue
    nrm /= np.linalg.norm(nrm)
    d0 = -nrm @ s[0]
    inl = np.abs(P @ nrm + d0) < 0.015
    if best is None or inl.sum() > best[2].sum():
        best = (nrm, d0, inl)
nrm, d0, inl = best
A = np.c_[P[inl], np.ones(inl.sum())]
_, _, vt = np.linalg.svd(A, full_matrices=False)
pl = vt[-1] / np.linalg.norm(vt[-1][:3])
nrm, d0 = pl[:3], pl[3]
if nrm[1] > 0:  # normal pointing up = -y
    nrm, d0 = -nrm, -d0
cam_h = abs(d0)
pitch_down = np.arctan2(-nrm[2], -nrm[1])  # up vector has a -z (backward) part when the camera looks down
roll = np.arctan2(nrm[0], -nrm[1])
res = np.abs(P[inl] @ nrm + d0)
body_pitch = dbg[:, 13].mean() if len(dbg) else 0.0
est_h = dbg[dbg[:, 1] == 3, 10].mean() if (dbg[:, 1] == 3).any() else np.nan
print("floor inliers %d (%.0f %% of lower half), plane residual %.1f mm" % (inl.sum(), 100 * inl.mean(), 1000 * res.mean()))
print("REAL : camera height %.3f m, pitch down %.3f rad (%.1f deg), roll %.3f rad" % (cam_h, pitch_down, np.degrees(pitch_down), roll))
exp_h = est_h + 0.055 * np.cos(body_pitch) - 0.32 * np.sin(body_pitch)
print("SIM  : camera pitch down 0.227 rad (13.0 deg) + body pitch %.3f = %.3f rad; camera height = body h %.3f "
      "-> %.3f m" % (body_pitch, 0.227 + body_pitch, est_h, exp_h))
np.save(sys.argv[1].replace(".npz", "_median.npy"), med)

# student input: 64x112, clip 4 m, invalid -> 4 m (sim: nan/inf -> DEPTH_MAX), normalized
def student(img):
    ys = (np.arange(64) + 0.5) * img.shape[0] / 64
    xs = (np.arange(112) + 0.5) * img.shape[1] / 112
    s = img[ys.astype(int)][:, xs.astype(int)]
    return np.clip(np.nan_to_num(s, nan=4.0), 0, 4.0) / 4.0
real_in = student(np.where(valid, med, np.nan))
# analytic sim floor at the nominal pose (87x58 deg, 64x112, pitch 0.227 + body pitch, height exp_h)
sfx = 56 / np.tan(np.radians(87 / 2))
sfy = 32 / np.tan(np.radians(58 / 2))
uu, vv = np.meshgrid(np.arange(112) + 0.5 - 56, np.arange(64) + 0.5 - 32)
th = 0.227 + body_pitch
dirz = np.cos(th) - (vv / sfy) * (-np.sin(th))  # ray forward component (unused, documentation)
down = (vv / sfy) * np.cos(th) + np.sin(th)       # ray component toward the floor per unit optical z
sim_d = np.where(down > 1e-4, exp_h / np.maximum(down, 1e-4), np.inf)
sim_in = np.clip(np.nan_to_num(sim_d, posinf=4.0), 0, 4.0) / 4.0
diff = np.abs(real_in - sim_in)
m = real_in < 0.999
print("student input (64x112, /4 m): real vs sim floor, mean |diff| %.3f (= %.2f m) on pixels both see the floor" % (
    diff[m & (sim_in < 0.999)].mean(), 4 * diff[m & (sim_in < 0.999)].mean()))
print("horizon row (first floor row from top): real %d, sim %d of 64" % (
    np.argmax((real_in < 0.999).mean(1) > 0.5), np.argmax((sim_in < 0.999).mean(1) > 0.5)))
if "--png" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a, img, t in ((ax[0], real_in, "real D435i -> student input"), (ax[1], sim_in, "sim (nominal pose)"),
                      (ax[2], real_in - sim_in, "real - sim")):
        im = a.imshow(img, cmap="inferno" if a is not ax[2] else "coolwarm", vmin=0 if a is not ax[2] else -0.3,
                      vmax=1 if a is not ax[2] else 0.3)
        a.set_title(t)
        plt.colorbar(im, ax=a, fraction=0.03)
    plt.tight_layout()
    plt.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=90)
