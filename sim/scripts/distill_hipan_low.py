"""DAgger: distil the HiPAN low-level teacher into a proprioceptive student (no motion state, domain or heightmap).

    python scripts/distill_hipan_low.py --teacher logs/rsl_rl/hipan_low/<run>/model_9998.pt \
        --out weights/go2_hipan_low_student.pt

The student sees the noisy ``student`` group (50-step history) and learns the teacher's action plus the motion state
x_m = [v_B, h, roll]. Training uses the teacher's final command ranges (curriculum level 1) and a 0-40 ms control delay
per robot and episode (our crouch student collapsed at >= 30 ms without it).
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
ap.add_argument("--envs", type=int, default=4096)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--beta_iters", type=int, default=100, help="teacher share of the actions goes 1 -> 0 over these")
ap.add_argument("--delay_max", type=int, default=8, help="max control delay in 5 ms physics steps")
ap.add_argument("--save_every", type=int, default=250)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--task", default="Go2-HiPAN-Low", help="env of the teacher (e.g. Go2-HiPAN-Low-V3 for its randomization)")
ap.add_argument("--no_contact", action="store_true",
                help="zero the 4 foot-contact entries (36:40) of the student input: the real foot_force zero drifts")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]

cfg, agent = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
# the teacher finished its command curriculum (level 1.0): train on the full ranges, penalties do not matter here
v = cfg.commands.base_velocity.ranges
v.lin_vel_x, v.lin_vel_y, v.ang_vel_z = (-1.5, 1.5), (-1.0, 1.0), (-1.5, 1.5)
cfg.commands.base_roll.start = cfg.commands.base_roll.full
if getattr(cfg.curriculum, "command_levels", None) is not None:
    cfg.curriculum.command_levels = None
cfg.curriculum.penalty_scale = None
if args.delay_max > 0:
    from go2_lab.crouch import delayed_action

    delayed_action(cfg, 0, args.delay_max)

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.hipan_low import motion_state
    from go2_lab.hipan_student import EST, H, HipanStudent, History

    torch.manual_seed(args.seed)
    env = gym.make(args.task, cfg=cfg)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    runner = create_rsl_rl_runner(venv, agent)
    runner.load(args.teacher)
    teach = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    dev, n = u.device, u.num_envs

    obs = venv.get_observations()
    D = obs["student"].shape[1]
    net = HipanStudent(d=D).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, 1e-4)
    hist = History(n, dev, d=D)
    print(f"student obs {D}, history {H}, estimate {EST}, envs {n}, delay 0-{args.delay_max * 5} ms", flush=True)

    def save(path, it):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"student": net.state_dict(), "teacher": args.teacher, "iters": it, "obs_dim": D, "history": H, "no_contact": bool(args.no_contact),
                    "est": EST, "delay_max_ms": args.delay_max * 5}, path)

    t0 = time.time()
    for it in range(args.iters):
        beta = max(0.0, 1 - it / args.beta_iters)
        X, A, M, fell = [], [], [], 0
        with torch.no_grad():
            for _ in range(args.steps):
                so = obs["student"].clone()
                if args.no_contact:
                    so[:, 36:40] = 0.0
                h = hist.push(so)
                a_t = teach(obs).clone()
                m_t = motion_state(u).clone()
                a_s, _ = net(h)
                X.append(h); A.append(a_t); M.append(m_t)
                use_t = torch.rand(n, 1, device=dev) < beta
                obs, _, done, _ = venv.step(torch.where(use_t, a_t, a_s))
                hist.reset(done)
                fell += int(u.termination_manager.terminated.sum())
        la = lm = 0.0
        for t in torch.randperm(args.steps).tolist():
            a, m = net(X[t])
            l1, l2 = (a - A[t]).pow(2).mean(), (m - M[t]).pow(2).mean()
            opt.zero_grad(); (l1 + l2).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            la += l1.item() / args.steps; lm += l2.item() / args.steps
        sched.step()
        if it % 25 == 0 or it == args.iters - 1:
            el = time.time() - t0
            print(f"it {it}/{args.iters} beta {beta:.2f} action_mse {la:.4f} motion_rmse {lm ** .5:.3f} "
                  f"falls/env/10s {fell / n / (args.steps * u.step_dt) * 10:.3f} {el:.0f}s "
                  f"ETA {el / (it + 1) * (args.iters - it - 1) / 60:.0f}min", flush=True)
        if args.save_every and it and it % args.save_every == 0:
            save(args.out.replace(".pt", f"_it{it}.pt"), it)
    save(args.out, args.iters)
    env.close()
