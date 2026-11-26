"""Evaluate the HiPAN high-level teacher or depth student on unseen worlds: final goal only, first episode of every
robot. Success / fall (trunk or head contact) / time-out, time to goal, by tile difficulty.

    python scripts/eval_hipan_nav.py --teacher logs/rsl_rl/hipan_nav/<run>/model_3999.pt --out results/hipan_nav/eval_teacher.json
    python scripts/eval_hipan_nav.py --teacher <same> --student weights/go2_hipan_nav_student.pt --out ...
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
ap.add_argument("--out")
ap.add_argument("--envs", type=int, default=480)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--seed", type=int, default=11, help="world seed (teacher trained on 1, student on 3)")
ap.add_argument("--pgcl", action="store_true", help="keep the sub-goals (teacher only; default: final goal)")
ap.add_argument("--switch", type=float, default=0.0, help="per-step probability of a goal change")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
TASK = "Go2-HiPAN-Nav-Depth" if args.student else "Go2-HiPAN-Nav"
cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
g = cfg.scene.terrain.terrain_generator
g.num_rows, g.num_cols, g.seed = args.rows, args.cols, args.seed

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.hipan_nav import NavVecEnv

    env = gym.make(TASK, cfg=cfg)
    venv = NavVecEnv(env, args.low, depth=bool(args.student))
    ns, dev, n = venv.ns, venv.device, venv.num_envs
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    runner = create_rsl_rl_runner(venv, agent)
    runner.load(args.teacher)
    actor = runner.alg.actor.eval()
    ns.final_only = not args.pgcl
    env.reset()
    obs = venv._observe()
    if args.student:
        from go2_lab.hipan_nav_student import NavStudent

        ck = torch.load(args.student, map_location=dev, weights_only=True)
        net = NavStudent(z_dim=ck["z_dim"]).to(dev).eval()
        net.load_state_dict(ck["student"])
        h = net.init_hidden(n, dev)

    task0 = ns.task.clone()
    L = ns.len[task0].clone()
    rows = (ns.tile // args.cols).float() / max(args.rows - 1, 1)
    out = torch.full((n,), -1, dtype=torch.long, device=dev)  # 0 success, 1 fall, 2 time-out
    t_end = torch.zeros(n, device=dev)
    crouch = torch.zeros(n, device=dev)
    steps = 0
    while (out < 0).any() and steps < 700:
        with torch.no_grad():
            if args.student:
                a, _, h = net.step(obs["depth"], torch.cat([obs["nav"], obs["est"]], 1), h)
            else:
                a = actor(obs)
        obs, _, done, ex = venv.step(a)
        steps += 1
        live = out < 0
        crouch += (live & (obs["priv"][:, 3] < 0.25)).float()  # body height below 0.25 m
        d = done.bool() & live
        res = torch.where(ex["success"], 0, torch.where(ex["timeout"], 2, 1))
        out = torch.where(d, res, out)
        t_end = torch.where(d, torch.full_like(t_end, steps * 0.1), t_end)
        if args.student:
            h = h * (~done.bool()).float()[:, None]
        if args.switch:
            sw = (torch.rand(n, device=dev) < args.switch) & ~done.bool() & (out < 0)
            if sw.any():
                ns.switch_goal(sw.nonzero().flatten())
                obs = venv._observe()

    def summ(m):
        k = max(int(m.sum()), 1)
        s = (out == 0) & m
        return dict(n=int(m.sum()), success=round(float(s.sum()) / k, 3), fall=round(float(((out == 1) & m).sum()) / k, 3),
                    timeout=round(float(((out == 2) & m).sum()) / k, 3), unfinished=round(float(((out < 0) & m).sum()) / k, 3),
                    time_s=round(float(t_end[s].mean()), 1) if s.any() else None,
                    speed=round(float((L[s] / t_end[s].clamp(min=0.1)).mean()), 2) if s.any() else None,
                    crouch_share=round(float((crouch[m] / (t_end[m] * 10).clamp(min=1)).mean()), 3))

    res = dict(all=summ(torch.ones(n, dtype=torch.bool, device=dev)))
    for name, lo, hi in (("easy", -0.1, 0.34), ("mid", 0.34, 0.67), ("hard", 0.67, 1.1)):
        res[name] = summ((rows > lo) & (rows <= hi))
    for name, lo, hi in (("path<6m", 0, 6), ("path6-10m", 6, 10), ("path>10m", 10, 99)):
        res[name] = summ((L >= lo) & (L < hi))
    for k, v in res.items():
        print(k, v, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(teacher=args.teacher, student=args.student, seed=args.seed,
                                                  switch=args.switch, pgcl=args.pgcl, results=res), indent=1))
    env.close()
