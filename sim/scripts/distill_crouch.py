"""DAgger: distil a crouch-walk teacher into a student without base linear velocity.

    python scripts/distill_crouch.py --teacher logs/.../model_2998.pt --out weights/go2_crouch_student.pt
"""
import argparse
import sys
import time
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
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="Go2-Crouch-Robust")
ap.add_argument("--envs", type=int, default=4096)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--beta_iters", type=int, default=100, help="teacher share of the actions goes 1 -> 0 over these")
ap.add_argument("--seed", type=int, default=1)
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
cfg.commands.base_height.curriculum = False

with launch_simulation(cfg, args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from go2_lab.student import History, Student, mlp_actor

    torch.manual_seed(args.seed)
    env = ManagerBasedRLEnv(cfg=cfg)
    dev = env.device
    teach = mlp_actor(torch.load(args.teacher, map_location=dev, weights_only=True)["actor_state_dict"])
    net = Student().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, 1e-4)
    hist = History(args.envs, dev)
    robot = env.scene["robot"]
    obs = env.reset()[0]["policy"]
    t0 = time.time()
    for it in range(args.iters):
        beta = max(0.0, 1 - it / args.beta_iters)
        X, A, V, fell = [], [], [], 0
        with torch.no_grad():
            for _ in range(args.steps):
                h = hist.push(obs)
                a_t, v = teach(obs), robot.data.root_lin_vel_b.torch.clone()
                a_s, _ = net(h)
                X.append(h); A.append(a_t); V.append(v)
                use_t = torch.rand(args.envs, 1, device=dev) < beta
                o, _, term, trunc, _ = env.step(torch.where(use_t, a_t, a_s))
                obs = o["policy"].clone()
                hist.reset(term | trunc)
                fell += int(term.sum())
        la = lv = 0.0
        for t in torch.randperm(args.steps).tolist():
            a, v = net(X[t])
            l1, l2 = (a - A[t]).pow(2).mean(), (v - V[t]).pow(2).mean()
            opt.zero_grad(); (l1 + l2).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            la += l1.item() / args.steps; lv += l2.item() / args.steps
        sched.step()
        if it % 25 == 0 or it == args.iters - 1:
            print(f"it {it} beta {beta:.2f} action_mse {la:.4f} vel_rmse {lv ** .5:.3f} falls/env/10s "
                  f"{fell / args.envs / (args.steps * env.step_dt) * 10:.3f} {time.time() - t0:.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": net.state_dict(), "teacher": args.teacher, "iters": args.iters}, args.out)
    env.close()
