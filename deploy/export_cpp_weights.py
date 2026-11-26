"""npz (sim/scripts/export_hipan_low.py) -> flat binary for the C++ node (go2_hipan_low), plus the sim replay test.

    python3 export_cpp_weights.py weights/hipan_low_student.npz      # -> weights/hipan_low_student.bin, *_test.bin

Format (little-endian): b"HPL1", int32 H, D, EST, float32 default_q[12], offset[12], scale[12], kp[12], kd[12],
int32 n_layers, then per layer in forward order: int32 rows, int32 cols, float32 W[rows*cols] (row-major), b[rows].
Test file: b"HPT1", int32 T, D, 12, EST, float32 obs[T*D], act[T*12], est[T*EST].
"""
import json
import struct
import sys

import numpy as np

LAYERS = ["enc_0", "enc_2", "enc_4", "motion", "pi_0", "pi_2", "pi_4", "pi_6"]

src = sys.argv[1] if len(sys.argv) > 1 else "weights/hipan_low_student.npz"
z = np.load(src)
meta = json.loads(str(z["meta"]))
out = src.replace(".npz", ".bin")
with open(out, "wb") as f:
    f.write(b"HPL1")
    f.write(struct.pack("<3i", meta["history"], meta["obs_dim"], meta["est_dim"]))
    for k in ("default_joint_pos", "action_offset", "action_scale", "kp", "kd"):
        f.write(np.asarray(meta[k], "<f4").tobytes())
    f.write(struct.pack("<i", len(LAYERS)))
    for name in LAYERS:
        W, b = z[name + "_weight"].astype("<f4"), z[name + "_bias"].astype("<f4")
        f.write(struct.pack("<2i", *W.shape))
        f.write(W.tobytes())
        f.write(b.tobytes())
test = src.replace(".npz", "_test.bin")
with open(test, "wb") as f:
    obs, act, est = z["test_obs"], z["test_act"], z["test_est"]
    f.write(b"HPT1")
    f.write(struct.pack("<4i", obs.shape[0], obs.shape[1], act.shape[1], est.shape[1]))
    for a in (obs, act, est):
        f.write(a.astype("<f4").tobytes())
print("wrote", out, test)
