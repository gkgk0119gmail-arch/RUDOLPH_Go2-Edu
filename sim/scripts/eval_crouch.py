"""Measure a crouch-walk checkpoint on flat ground: speed x height grid, falls, tracking, real top height.

    python scripts/eval_crouch.py --ckpt crouch_init.pt --out results/gap_search/eval_x.json
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
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out")
ap.add_argument("--per", type=int, default=24, help="robots per (speed, height) cell")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--no_payload", action="store_true")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
SPEEDS, HEIGHTS = (0.0, 0.3, 0.6), (0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16)
cfg, _ = resolve_task_config("Go2-Crouch-Walk", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.per * len(SPEEDS) * len(HEIGHTS)
cfg.scene.env_spacing = 3.0
cfg.scene.terrain.terrain_type, cfg.scene.terrain.terrain_generator = "plane", None
cfg.curriculum.terrain_levels = None
cfg.commands.base_height.curriculum = False
cfg.observations.policy.enable_corruption = False
cfg.events.push_robot = cfg.events.add_base_mass = cfg.events.base_com = None
if args.no_payload:
    cfg.events.payload = None
cfg.episode_length_s = args.seconds + 5
cfg.seed = 7

with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from go2_lab.crouch import body_height

    env = ManagerBasedRLEnv(cfg=cfg)
    sd = torch.load(args.ckpt, map_location=env.device, weights_only=True)["actor_state_dict"]
    nin = sd["mlp.0.weight"].shape[1]

    def act(x):
        for i in (0, 2, 4):
            x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
        return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]

    n = env.num_envs
    idx = torch.arange(n, device=env.device)
    sp = torch.tensor(SPEEDS, device=env.device)[idx // (args.per * len(HEIGHTS))]
    hc = torch.tensor(HEIGHTS, device=env.device)[(idx // args.per) % len(HEIGHTS)]
    obs, _ = env.reset()
    robot = env.scene["robot"]
    b = robot.body_names.index("base")
    mass = robot.data.body_mass.torch[:, b]
    com = robot.data.body_com_pose_b.torch[0, b, :3].tolist()
    print(f"base mass {float(mass.min()):.3f}-{float(mass.max()):.3f} kg, com {[round(c, 4) for c in com]}, obs {obs['policy'].shape}")
    vt, pt, ht = (env.command_manager.get_term(k) for k in ("base_velocity", "base_posture", "base_height"))
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    acc = {k: torch.zeros(n, device=env.device) for k in ("h", "h_err", "v", "tilt", "cnt")}
    hmax = torch.zeros(n, device=env.device)
    steps, warm = int(args.seconds / env.step_dt), int(3.0 / env.step_dt)

    def force(o):
        vt.command[:] = 0
        vt.command[:, 0] = sp
        pt.effective[:] = 0
        ht.target[:, 0] = hc
        o = o.clone()
        o[:, 9:12] = vt.command       # velocity command
        o[:, 48:50] = 0               # posture command
        o[:, 50] = hc
        return o

    for t in range(steps):
        with torch.inference_mode():
            obs, _, term, trunc, _ = env.step(act(force(obs["policy"])[:, :nin]))
        alive &= ~term
        if t >= warm:
            h = body_height(env)
            m = alive.float()
            acc["h"] += h * m
            acc["h_err"] += (h - hc).abs() * m
            acc["v"] += robot.data.root_lin_vel_b.torch[:, 0] * m
            acc["tilt"] += torch.rad2deg(torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1, 1))) * m
            acc["cnt"] += m
            hmax = torch.where(alive, torch.maximum(hmax, h), hmax)
    rows = []
    for s in SPEEDS:
        for h in HEIGHTS:
            q = (sp == s) & (hc == h) if False else ((sp - s).abs() < 1e-6) & ((hc - h).abs() < 1e-6)
            c = acc["cnt"][q].sum().clamp(min=1)
            ok = q & alive
            rows.append(dict(speed_cmd=s, height_cmd=h, n=int(q.sum()), fell=round(float((~alive[q]).float().mean()), 3),
                             height=round(float(acc["h"][q].sum() / c), 3), height_err=round(float(acc["h_err"][q].sum() / c), 3),
                             speed=round(float(acc["v"][q].sum() / c), 3), tilt_deg=round(float(acc["tilt"][q].sum() / c), 1),
                             base_max=round(float(hmax[ok].mean()), 3) if ok.any() else None))
            print(rows[-1], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(ckpt=args.ckpt, payload=not args.no_payload, base_mass=float(mass[0]),
                                                  rows=rows), indent=1))
    env.close()
