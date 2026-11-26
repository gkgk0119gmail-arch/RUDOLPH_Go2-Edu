"""DAgger: distil the HiPAN high-level teacher (privileged map, PGCL) into a depth + GRU student (final goal only).

    python scripts/distill_hipan_nav.py --teacher logs/rsl_rl/hipan_nav/<run>/model_3999.pt \
        --out weights/go2_hipan_nav_student.pt

Loss = |c - c_hat|^2 + w_z |z_s - z_hat|^2 (z_s: the teacher's map latent). Truncated BPTT over chunks of --steps.
Our extension: the goal may change mid-episode (--switch per-step probability), as the goal selector will do.
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
ap.add_argument("--low", default=str(ROOT / "weights/go2_hipan_low_student.pt"))
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--beta_iters", type=int, default=300)
ap.add_argument("--wz", type=float, default=0.1)
ap.add_argument("--switch", type=float, default=0.003)
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=3)
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]

TASK = "Go2-HiPAN-Nav-Depth"
cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
g = cfg.scene.terrain.terrain_generator
g.seed = args.seed  # other layouts than the teacher's training world

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.hipan_nav import NavVecEnv
    from go2_lab.hipan_nav_student import NavStudent

    torch.manual_seed(args.seed)
    env = gym.make(TASK, cfg=cfg)
    venv = NavVecEnv(env, args.low, depth=True)
    ns, dev, n = venv.ns, venv.device, venv.num_envs
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    runner = create_rsl_rl_runner(venv, agent)
    runner.load(args.teacher)
    actor = runner.alg.actor.eval()
    ns.final_only = True
    env.reset()  # restart every robot with the final goal only
    obs = venv._observe()

    def teacher(o):
        with torch.no_grad():
            return actor(o), actor.cnns["map"](o["map"])

    def x_of(o):
        return torch.cat([o["nav"], o["est"]], 1)

    net = NavStudent(z_dim=actor.cnns["map"](obs["map"]).shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, 5e-5)
    h = net.init_hidden(n, dev)
    reset = torch.ones(n, dtype=torch.bool, device=dev)

    def save(path, it):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"student": net.state_dict(), "teacher": args.teacher, "iters": it, "z_dim": net.z.out_features,
                    "depth_hw": list(obs["depth"].shape[-2:])}, path)

    t0 = time.time()
    stat = dict(succ=0, done=0, fall=0)
    for it in range(args.iters):
        beta = max(0.0, 1 - it / args.beta_iters)
        h0, r0 = h.detach(), reset.clone()
        Dp, X, A, Z, R = [], [], [], [], []
        with torch.no_grad():
            for _ in range(args.steps):
                x = x_of(obs)
                net.norm.update(x)
                a_t, z_t = teacher(obs)
                h = h * (~reset).float()[:, None]
                a_s, _, h = net.step(obs["depth"], x, h)
                Dp.append(obs["depth"]); X.append(x); A.append(a_t); Z.append(z_t); R.append(reset)
                use_t = torch.rand(n, 1, device=dev) < beta
                obs, _, done, ex = venv.step(torch.where(use_t, a_t, a_s))
                reset = done.bool()
                stat["succ"] += int(ex["success"].sum()); stat["done"] += int(reset.sum())
                stat["fall"] += int((reset & ~ex["success"] & ~ex["timeout"]).sum())
                sw = (torch.rand(n, device=dev) < args.switch) & ~reset
                if sw.any():
                    ns.switch_goal(sw.nonzero().flatten())
                    obs = venv._observe()
        R[0] = r0
        a_hat, z_hat, _ = net.unroll(torch.stack(Dp), torch.stack(X), h0, torch.stack(R))
        la = (a_hat - torch.stack(A)).square().mean()
        lz = (z_hat - torch.stack(Z)).square().mean()
        opt.zero_grad(); (la + args.wz * lz).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if it % 25 == 0 or it == args.iters - 1:
            el = time.time() - t0
            dn = max(stat["done"], 1)
            print(f"it {it}/{args.iters} beta {beta:.2f} action_mse {la.item():.4f} z_mse {lz.item():.4f} "
                  f"episodes {stat['done']} success {stat['succ'] / dn:.2f} fall {stat['fall'] / dn:.2f} "
                  f"{el:.0f}s ETA {el / (it + 1) * (args.iters - it - 1) / 60:.0f}min", flush=True)
            stat = dict(succ=0, done=0, fall=0)
        if args.save_every and it and it % args.save_every == 0:
            save(args.out.replace(".pt", f"_it{it}.pt"), it)
    save(args.out, args.iters)
    env.close()
