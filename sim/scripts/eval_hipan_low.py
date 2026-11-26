"""Measure a HiPAN low-level teacher on flat ground: commanded vs real body height / roll / speed, falls.

    python scripts/eval_hipan_low.py --ckpt logs/rsl_rl/hipan_low/<run>/model_5999.pt --out results/hipan_low/eval.json
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
ap.add_argument("--per", type=int, default=16)
ap.add_argument("--seconds", type=float, default=8.0)
ap.add_argument("--rough", action="store_true", help="training terrain mix instead of a plane")
ap.add_argument("--student", help="HiPAN student weights; --ckpt is then only used to build the runner")
ap.add_argument("--delay_ms", type=int, default=0, help="fixed control delay (multiple of 5 ms)")
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
HEIGHTS = (0.36, 0.32, 0.28, 0.24, 0.20, 0.16, 0.12)
ROLLS = (0.0, 0.5, 1.0)
SPEED = 0.4
cfg, agent = resolve_task_config("Go2-HiPAN-Low", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.per * len(HEIGHTS) * len(ROLLS)
cfg.scene.env_spacing = 3.0
cfg.curriculum.terrain_levels = cfg.curriculum.command_levels = cfg.curriculum.penalty_scale = None
if not args.rough:
    cfg.scene.terrain.terrain_type, cfg.scene.terrain.terrain_generator = "plane", None
cfg.events.push_robot = None
for k in ("base_velocity", "base_height", "base_roll"):
    getattr(cfg.commands, k).resampling_time_range = (1e6, 1e6)
cfg.episode_length_s = args.seconds + 2
cfg.seed = 7
if args.delay_ms:
    from go2_lab.crouch import delayed_action

    delayed_action(cfg, args.delay_ms // 5, args.delay_ms // 5)

with launch_simulation(cfg, args):
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    from go2_lab.crouch import body_height
    from go2_lab.hipan_low import roll_world

    env = gym.make("Go2-HiPAN-Low", cfg=cfg)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    runner = create_rsl_rl_runner(venv, agent)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    hist = None
    if args.student:
        from go2_lab.hipan_student import HipanStudent, History

        ck = torch.load(args.student, map_location=dev, weights_only=True)
        net = HipanStudent(d=ck["obs_dim"]).to(dev).eval()
        net.load_state_dict(ck["student"])
        hist = History(n, dev, d=ck["obs_dim"])

        def policy(o):  # noqa: F811
            return net(hist.push(o["student"]))[0]
    idx = torch.arange(n, device=dev)
    hc = torch.tensor(HEIGHTS, device=dev)[(idx // args.per) % len(HEIGHTS)]
    rc = torch.tensor(ROLLS, device=dev)[idx // (args.per * len(HEIGHTS))]
    vt, ht, rt = (u.command_manager.get_term(k) for k in ("base_velocity", "base_height", "base_roll"))

    def fix():
        vt.vel_command_b[:] = 0
        vt.vel_command_b[:, 0] = SPEED
        ht.target[:, 0] = hc
        rt.target[:, 0] = rc

    fix()
    obs = venv.get_observations()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    acc = {k: torch.zeros(n, device=dev) for k in ("h", "r", "v", "cnt")}
    steps, warm = int(args.seconds / u.step_dt), int(3.0 / u.step_dt)
    for t in range(steps):
        fix()
        with torch.inference_mode():
            obs, _, done, _ = venv.step(policy(obs))
            if hist is not None:
                hist.reset(done)
        alive &= ~u.termination_manager.terminated
        if t >= warm:
            m = alive.float()
            acc["h"] += body_height(u) * m
            acc["r"] += roll_world(u) * m
            acc["v"] += u.scene["robot"].data.root_lin_vel_b.torch[:, 0] * m
            acc["cnt"] += m
    rows = []
    for r in ROLLS:
        for h in HEIGHTS:
            q = ((hc - h).abs() < 1e-6) & ((rc - r).abs() < 1e-6)
            c = acc["cnt"][q].sum().clamp(min=1)
            rows.append(dict(height_cmd=h, roll_cmd=r, speed_cmd=SPEED, n=int(q.sum()),
                             fell=round(float((~alive[q]).float().mean()), 3),
                             height=round(float(acc["h"][q].sum() / c), 3), roll=round(float(acc["r"][q].sum() / c), 3),
                             speed=round(float(acc["v"][q].sum() / c), 3)))
            print(rows[-1], flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(ckpt=args.ckpt, student=args.student, delay_ms=args.delay_ms,
                                                  rough=args.rough, rows=rows), indent=1))
    env.close()
