"""Posture vs overhead clearance: does the nav policy stand back up after passing under an obstacle?
Bins every high-level step by the overhead clearance over the body footprint and 1.5 m ahead (privileged map),
reports commanded / actual body height per bin.

    python scripts/posture_hipan_nav.py --teacher <model.pt> [--student <student.pt>]
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
ap.add_argument("--teacher", required=True)
ap.add_argument("--student")
ap.add_argument("--low", default=str(ROOT / "weights/go2_hipan_low_student.pt"))
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
TASK = "Go2-HiPAN-Nav-Depth" if args.student else "Go2-HiPAN-Nav"
cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
g = cfg.scene.terrain.terrain_generator
g.num_rows, g.num_cols, g.seed = 4, 6, args.seed

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.crouch import body_height
    from go2_lab.hipan_nav import NavVecEnv

    env = gym.make(TASK, cfg=cfg)
    venv = NavVecEnv(env, args.low, depth=bool(args.student))
    ns, dev, n = venv.ns, venv.device, venv.num_envs
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    runner = create_rsl_rl_runner(venv, agent)
    runner.load(args.teacher)
    actor = runner.alg.actor.eval()
    ns.final_only = True
    env.reset()
    obs = venv._observe()
    if args.student:
        from go2_lab.hipan_nav_student import NavStudent

        ck = torch.load(args.student, map_location=dev, weights_only=True)
        net = NavStudent(z_dim=ck["z_dim"]).to(dev).eval()
        net.load_state_dict(ck["student"])
        h = net.init_hidden(n, dev)
    # map grid: x -1..3 (41, 0.1 m), y -1.5..1.5 (31); footprint x -0.4..0.4, |y|<=0.2 ; ahead x 0.4..1.5
    rec = []
    for _ in range(args.steps):
        m = obs["map"][:, 1]
        foot = m[:, 6:15, 13:18].amin((1, 2))
        ahead = m[:, 14:26, 13:18].amin((1, 2))
        behind = m[:, 0:6, 13:18].amin((1, 2))
        with torch.no_grad():
            if args.student:
                a, _, h = net.step(obs["depth"], torch.cat([obs["nav"], obs["est"]], 1), h)
            else:
                a = actor(obs)
        obs, _, done, _ = venv.step(a)
        rec.append(torch.stack([foot, ahead, behind, venv.cmd[:, 3], body_height(venv.unwrapped)], 1)[~done.bool()])
        if args.student:
            h = h * (~done.bool()).float()[:, None]
    R = torch.cat(rec)
    foot, ahead, behind, hc, ha = R.T
    OPEN = 0.6  # >= 0.51 standing clearance + margin
    bins = {
        "open (no overhang around)": (foot >= OPEN) & (ahead >= OPEN) & (behind >= OPEN),
        "overhang ahead (approach)": (foot >= OPEN) & (ahead < OPEN),
        "under overhang": foot < OPEN,
        "just passed (overhang behind only)": (foot >= OPEN) & (ahead >= OPEN) & (behind < OPEN),
    }
    res = {}
    for k, msk in bins.items():
        if msk.sum() == 0:
            continue
        res[k] = dict(share=round(float(msk.float().mean()), 3), cmd_h=round(float(hc[msk].mean()), 3),
                      act_h=round(float(ha[msk].mean()), 3), low_share=round(float((hc[msk] < 0.26).float().mean()), 3))
        print(f"{k:38s}", res[k], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    env.close()
