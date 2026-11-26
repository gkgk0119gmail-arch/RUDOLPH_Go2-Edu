"""Export the HiPAN low-level student for the real robot (numpy-only inference on the NX).

Writes weights/deploy/hipan_low_student.npz:
  * every Linear layer of HipanStudent (enc.*, motion, pi.*) as float32
  * meta read from a live Isaac env (no hand copying): joint names in policy order, default joint positions,
    action scale / offset, actuator stiffness / damping / effort limit, control dt, history length, obs layout + scales
  * a replay test: a sim rollout on flat ground (observation per step + torch action + motion estimate), so the NX
    runner can check that numpy gives the same actions on the same inputs

    python scripts/export_hipan_low.py [--student weights/go2_hipan_low_student.pt] [--steps 300]
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
ap.add_argument("--student", default=str(ROOT / "weights/go2_hipan_low_student.pt"))
ap.add_argument("--out", default=str(ROOT / "weights/deploy/hipan_low_student.npz"))
ap.add_argument("--task", default="Go2-HiPAN-Low-Play")
ap.add_argument("--steps", type=int, default=300)
add_launcher_args(ap)
args = ap.parse_args()
sys.argv = sys.argv[:1]
cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = 1
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
if getattr(cfg.curriculum, "terrain_levels", None) is not None:
    cfg.curriculum.terrain_levels = None
if getattr(cfg.events, "push_robot", None) is not None:
    cfg.events.push_robot = None

with launch_simulation(cfg, args):
    import gymnasium as gym
    import numpy as np
    import torch

    from go2_lab.hipan_student import History, HipanStudent

    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    dev = u.device
    ck = torch.load(args.student, map_location=dev, weights_only=True)
    net = HipanStudent(d=ck["obs_dim"]).to(dev).eval()
    net.load_state_dict(ck["student"])
    hist = History(1, dev, d=ck["obs_dim"])

    robot = u.scene["robot"]
    term = u.action_manager.get_term("joint_pos")
    import re
    kp = torch.zeros(len(robot.joint_names))
    kd = torch.zeros_like(kp)
    eff = torch.zeros_like(kp)
    act_cfgs = {}
    for name, a in cfg.scene.robot.actuators.items():
        act_cfgs[name] = {k: (v if isinstance(v, (int, float, str, dict, list, tuple)) or v is None else str(v))
                          for k, v in a.to_dict().items() if k != "class_type"}
        for j, jn in enumerate(robot.joint_names):
            if any(re.fullmatch(e, jn) for e in a.joint_names_expr):
                pick = lambda v: float(v if not isinstance(v, dict) else next(x for e, x in v.items() if re.fullmatch(e, jn)))  # noqa: E731
                kp[j], kd[j] = pick(a.stiffness), pick(a.damping)
                eff[j] = pick(a.saturation_effort if getattr(a, "saturation_effort", None) is not None else a.effort_limit)
    scale = term._scale if torch.is_tensor(term._scale) else torch.full((12,), float(term._scale))
    offset = term._offset if torch.is_tensor(term._offset) else torch.full((12,), float(term._offset))
    feet = u.observation_manager._group_obs_term_cfgs["student"][3].params["sensor_cfg"]
    meta = dict(
        source=args.student, task=args.task,
        joint_names=list(robot.joint_names),
        default_joint_pos=robot.data.default_joint_pos.torch[0].tolist(),
        action_scale=scale.reshape(-1)[:12].tolist() if scale.numel() > 1 else [float(scale)] * 12,
        action_offset=offset.reshape(-1)[:12].tolist() if offset.numel() > 1 else [float(offset)] * 12,
        kp=kp.tolist(), kd=kd.tolist(), effort_limit=eff.tolist(), actuator_cfgs=act_cfgs,
        control_dt=float(u.step_dt), sim_dt=float(u.physics_dt), decimation=int(u.cfg.decimation),
        history=net.h, obs_dim=net.d, est_dim=net.est,
        obs_terms=u.observation_manager.active_terms["student"],
        obs_term_dims=[list(d) for d in u.observation_manager.group_obs_term_dim["student"]],
        obs_scales=dict(joint_vel=0.05, ang_vel=0.25),
        foot_contact="net contact force > 1 N (binary)", feet_bodies=feet.body_names,
        body_names=list(robot.body_names),
        command_layout="[vx, vy, wz] (m/s, rad/s), base height above ground (m), world roll (rad)",
        estimate_layout="[v_B x, y, z (m/s), base height (m), world roll (rad)]",
    )

    # flat-ground replay: walk, then crouch + walk, then turn
    plan = [(100, [0.5, 0.0, 0.0, 0.30, 0.0]), (100, [0.4, 0.0, 0.0, 0.18, 0.0]), (100, [0.0, 0.0, 0.8, 0.30, 0.2])]
    env.reset()
    obs_log, act_log, est_log = [], [], []
    from go2_lab.hipan_nav import CMD_SLICE
    for n_steps, c in plan:
        for _ in range(n_steps):
            o = u.obs_buf["student"].clone()
            o[:, CMD_SLICE] = torch.tensor(c, device=dev)
            with torch.no_grad():
                a, m = net(hist.push(o))
            obs_log.append(o[0].cpu().numpy())
            act_log.append(a[0].cpu().numpy())
            est_log.append(m[0].cpu().numpy())
            env.step(a)
    W = {k.replace(".", "_"): v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, meta=json.dumps(meta), test_obs=np.array(obs_log, np.float32),
             test_act=np.array(act_log, np.float32), test_est=np.array(est_log, np.float32), **W)
    print(json.dumps(meta, indent=1), flush=True)
    print("weights:", {k: v.shape for k, v in W.items()}, flush=True)
    print(f"wrote {args.out}", flush=True)
    env.close()
