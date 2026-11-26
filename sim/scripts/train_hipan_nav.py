"""PPO for the HiPAN high-level navigation teacher (frozen low-level student inside the environment).

    python scripts/train_hipan_nav.py --low weights/go2_hipan_low_student.pt --iters 4000 --run teacher
    python scripts/train_hipan_nav.py --low weights/go2_hipan_low_student.pt --smoke 50 --envs 64
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import warp as wp

wp.config.enable_backward = False
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import go2_lab  # noqa: E402,F401
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Go2-HiPAN-Nav")
ap.add_argument("--low", default=str(ROOT / "weights/go2_hipan_low_student.pt"))
ap.add_argument("--envs", type=int, default=1024)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--run", default="teacher")
ap.add_argument("--resume")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--smoke", type=int, default=0, help="only step N times with random actions and report timing")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]

cfg, agent = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
g = cfg.scene.terrain.terrain_generator
g.num_rows, g.num_cols, g.seed = args.rows, args.cols, args.seed

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.hipan_nav import NavVecEnv

    env = gym.make(args.task, cfg=cfg)
    venv = NavVecEnv(env, args.low, depth="Depth" in args.task)
    if args.smoke:
        o = venv.get_observations()
        print({k: tuple(v.shape) for k, v in o.items()}, flush=True)
        t0, R, D, S = time.time(), 0.0, 0, 0
        for i in range(args.smoke):
            # pure pursuit towards the (sub)goal at nominal posture: checks arrival / PGCL bookkeeping
            gf = o["nav"][:, 62:65]
            bearing = torch.atan2(gf[:, 1], gf[:, 0])
            c = torch.stack([0.5 * (bearing.abs() < 0.5).float(), torch.zeros_like(bearing),
                             (1.5 * bearing).clamp(-1.0, 1.0), torch.full_like(bearing, 0.30),
                             torch.zeros_like(bearing)], 1)
            a = (c - (venv.c_lo + venv.c_hi) / 2) / ((venv.c_hi - venv.c_lo) / 2)
            o, r, d, ex = venv.step(a)
            R += float(r.mean()); D += int(d.sum()); S += int(ex["success"].sum())
            if i % 10 == 0:
                print(i, {k: round(v, 3) for k, v in ex["log"].items()}, flush=True)
        dt = time.time() - t0
        print(f"{args.smoke} high steps x {venv.num_envs} envs in {dt:.1f}s -> {args.smoke * venv.num_envs / dt:.0f} "
              f"high-steps/s; mean r {R / args.smoke:.3f}, dones {D}, successes {S}", flush=True)
        for k in ("nav", "priv", "map", "est"):
            v = o[k]
            print(k, float(v.min()), float(v.max()), float(v.mean()))
        if "depth" in o.keys():
            torch.save({k: o[k][:16].cpu() for k in o.keys()}, ROOT / "results/hipan_nav/smoke_obs.pt")
        env.close()
        raise SystemExit
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    agent.max_iterations, agent.seed = args.iters, args.seed
    log_dir = ROOT / "logs/rsl_rl" / agent.experiment_name / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run}"
    runner = create_rsl_rl_runner(venv, agent, log_dir=str(log_dir))
    if args.resume:
        runner.load(args.resume)
    print("log dir", log_dir, flush=True)
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=False)
    env.close()
