"""Does the D435i (0.32 m forward, 13 deg down, 87 x 58 deg) see an overhead beam? Robot standing (0.32 m) or crouched
(0.15 m) at several distances in front of beams with bottoms at 0.35 / 0.42 / 0.50 m.

    python scripts/check_depth_overhead.py --out results/hipan_nav/depth_overhead.json
"""
import argparse
import json
import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import go2_lab  # noqa: E402,F401
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/hipan_nav/depth_overhead.json"))
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
BOTTOMS, BEAM_X, BEAM_W, BEAM_T = (0.35, 0.42, 0.50), 7.0, 0.3, 0.3
DISTS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)  # camera lens -> beam front face [m]
HEIGHTS = (0.32, 0.15)
cfg, _ = resolve_task_config("Go2-HiPAN-Nav-Depth", "rsl_rl_cfg_entry_point")
cfg.scene.terrain.terrain_generator.scene = "beam_test"
cfg.scene.num_envs = len(BOTTOMS) * len(DISTS) * len(HEIGHTS)
cfg.events.base_com = cfg.events.payload = None
cfg.terminations.nav_timeout = None
cfg.terminations.goal = None

with launch_simulation(cfg, args):
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    from go2_lab.crouch import body_height
    from go2_lab.hipan_nav import D435I_POS, NavVecEnv

    env = gym.make("Go2-HiPAN-Nav-Depth", cfg=cfg)
    venv = NavVecEnv(env, str(ROOT / "weights/go2_hipan_low_student.pt"), depth=True)
    u, dev, n = env.unwrapped, venv.device, venv.num_envs
    cases = [(b, d, h) for b in BOTTOMS for d in DISTS for h in HEIGHTS]
    hc = torch.tensor([c[2] for c in cases], device=dev)
    tile = torch.tensor([BOTTOMS.index(c[0]) for c in cases], device=dev)
    face = BEAM_X - BEAM_W / 2
    x = torch.tensor([face - c[1] - D435I_POS[0] for c in cases], device=dev)
    pos = torch.stack([x, tile * 12.0 + 6.0, torch.full_like(x, 0.40)], 1)
    z = torch.zeros(n, device=dev)
    robot = u.scene["robot"]

    def place():
        robot.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat_from_euler_xyz(z, z, z)], 1))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))

    place()
    cmd = torch.stack([z, z, z, hc, z], 1)
    a = (cmd - (venv.c_lo + venv.c_hi) / 2) / ((venv.c_hi - venv.c_lo) / 2)
    for i in range(40):  # settle at the commanded height
        if i % 10 == 0:  # keep the robots at their distance (they drift a little while crouching)
            p = robot.data.root_pos_w.torch.clone()
            p[:, 0] = pos[:, 0]
            p[:, 1] = pos[:, 1]
            q = robot.data.root_quat_w.torch.clone()
            robot.write_root_pose_to_sim_index(root_pose=torch.cat([p, q], 1))
        o, _, done, _ = venv.step(a)
    cam = u.scene["depth"]
    hits = cam.ray_hits_w.torch  # [n, rays, 3]
    bot = torch.tensor([c[0] for c in cases], device=dev)
    on_beam = ((hits[..., 0] > face - 0.02) & (hits[..., 0] < BEAM_X + BEAM_W / 2 + 0.02)
               & (hits[..., 2] > bot[:, None] - 0.02) & (hits[..., 2] < bot[:, None] + BEAM_T + 0.02))
    under = on_beam & (hits[..., 2] < bot[:, None] + 0.02)  # rays on the beam's underside (bottom edge region)
    H, W = o["depth"].shape[-2:]
    rows_beam = on_beam.reshape(n, H, W).any(2)
    cam_z = cam.data.pos_w.torch[:, 2]
    bh = body_height(u)
    rx = robot.data.root_pos_w.torch[:, 0]
    out = []
    for i, (b, d, h) in enumerate(cases):
        r = rows_beam[i].nonzero().flatten()
        out.append(dict(beam_bottom=b, cam_to_face=d, height_cmd=h, height=round(float(bh[i]), 3),
                        cam_z=round(float(cam_z[i]), 3), dist_actual=round(float(face - rx[i] - D435I_POS[0]), 2),
                        beam_px=round(float(on_beam[i].float().mean()), 3), underside_px=int(under[i].sum()),
                        beam_rows=[int(r.min()), int(r.max())] if len(r) else None, fell=bool(done[i])))
        print(out[-1], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(image_hw=[H, W], rows=out), indent=1))
    np.save(Path(args.out).with_suffix(".npy"), o["depth"][:, 0].cpu().numpy())
    env.close()
