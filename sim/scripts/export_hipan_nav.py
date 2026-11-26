"""Export the HiPAN nav depth student (CNN + GRU) to npz for numpy inference on the NX, with a torch replay test.

    python scripts/export_hipan_nav.py --student weights/go2_hipan_nav_student_v2.pt --out ../deploy/weights/hipan_nav_student.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from go2_lab.hipan_nav import C_HI, C_LO, C_NOM, DEPTH_MAX, R_ARRIVE  # noqa: E402
from go2_lab.hipan_nav_student import NavStudent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--student", default=str(ROOT / "weights/go2_hipan_nav_student_v2.pt"))
ap.add_argument("--out", default=str(ROOT.parent / "deploy/weights/hipan_nav_student.npz"))
args = ap.parse_args()
ck = torch.load(args.student, map_location="cpu", weights_only=True)
net = NavStudent(z_dim=ck["z_dim"]).eval()
net.load_state_dict(ck["student"])
W = {k.replace(".", "_"): v.float().numpy() for k, v in net.state_dict().items()}
# replay test: 30 steps of random-but-plausible inputs through the GRU
g = torch.Generator().manual_seed(0)
T = 30
depth = torch.rand(T, 1, 1, 64, 112, generator=g)
x = torch.randn(T, 1, 71, generator=g) * 0.5
h = net.init_hidden(1, "cpu")
A = []
with torch.no_grad():
    for t in range(T):
        a, _, h = net.step(depth[t], x[t], h)
        A.append(a[0].numpy())
meta = dict(source=args.student, teacher=ck.get("teacher"), z_dim=ck["z_dim"], depth_hw=[64, 112], depth_max=DEPTH_MAX,
            depth_fov_deg=[87.0, 58.0], depth_pitch_deg=13.0, c_lo=C_LO, c_hi=C_HI, c_nom=C_NOM, r_arrive=R_ARRIVE,
            x_layout="low obs 62 (c_prev at 57:62) + goal (x/5, y/5 clipped at 5 m, dist/10) + v_des + est 5")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, meta=json.dumps(meta), test_depth=depth.numpy()[:, 0, 0], test_x=x.numpy()[:, 0],
         test_a=np.array(A), **W)
print("wrote", args.out, {k: v.shape for k, v in W.items()})
