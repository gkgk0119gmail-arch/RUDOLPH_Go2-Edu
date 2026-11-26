"""Record a video of the HiPAN high-level policy (teacher or depth student) driving to random goals.
Camera follows robot 0; overlay: policy, episode result, commanded height / speed; student: depth inset.

    python scripts/record_hipan_nav.py --teacher <model.pt> [--student <student.pt>] --out results/hipan_nav/teacher.mp4
"""
import argparse
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
ap.add_argument("--out", required=True)
ap.add_argument("--envs", type=int, default=1)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--max_s", type=float, default=90.0)
ap.add_argument("--tiles", default="", help="comma list of tiles for robot 0, one per episode (default: rows 1,2,3,3)")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
sys.argv = sys.argv[:1]
TASK = "Go2-HiPAN-Nav-Depth" if args.student else "Go2-HiPAN-Nav"
cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs, cfg.seed = args.envs, args.seed
g = cfg.scene.terrain.terrain_generator
g.num_rows, g.num_cols, g.seed = args.rows, args.cols, args.seed
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

for c in ("base_velocity", "base_height", "base_roll"):
    if hasattr(cfg.commands, c) and getattr(cfg.commands, c) is not None:
        getattr(cfg.commands, c).debug_vis = False

cfg.sim.visualizer_cfgs = [KitVisualizerCfg(origin_type="asset", origin_track_path="robot", origin_env_index=0,
                                            eye=(-2.4, -2.4, 3.4), lookat=(0.6, 0.6, 0.0), headless=True,
                                            window_width=1280, window_height=720)]

with launch_simulation(cfg, args):
    import cv2
    import gymnasium as gym
    import imageio
    import numpy as np
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
    ns.final_only = True
    if args.student:
        from go2_lab.hipan_nav_student import NavStudent

        ck = torch.load(args.student, map_location=dev, weights_only=True)
        net = NavStudent(z_dim=ck["z_dim"]).to(dev).eval()
        net.load_state_dict(ck["student"])
        h = net.init_hidden(n, dev)

    def diff0():
        return float(ns.tile[0] // args.cols) / max(args.rows - 1, 1)

    tiles = [int(t) for t in args.tiles.split(",")] if args.tiles else [r * args.cols + c for r, c in ((1, 2), (2, 4), (3, 1), (3, 3))]

    def reroll(i):  # put robot 0 (the one on camera) on tile tiles[i] with a fresh task
        ids = torch.tensor([0], device=dev)
        ns.tile[0] = tiles[i % len(tiles)]
        ns.new_task(ids)
        ns.fresh[0] = True
        env.unwrapped._reset_idx(ids)
        venv.hist.reset(torch.arange(n, device=dev) == 0)
        venv.cmd[0], venv.a1[0], venv.a2[0] = venv.c_nom, venv.a_nom, venv.a_nom
        venv.prev_dist = venv._goal_dist()

    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    goal_mk = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/nav_goal", markers={
        "ball": sim_utils.SphereCfg(radius=0.15, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))),
        "pole": sim_utils.CylinderCfg(radius=0.03, height=0.6, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.3)))}))

    def show_goal():
        gp = ns.goal_w()[:1]
        z = torch.zeros(1, 1, device=dev)
        goal_mk.visualize(translations=torch.cat([torch.cat([gp, z + 0.75], 1), torch.cat([gp, z + 0.3], 1)]),
                          marker_indices=torch.tensor([0, 1], device=dev))

    reroll(0)
    obs = venv._observe()
    label = "student (depth + GRU)" if args.student else "teacher (privileged map)"
    hud = dict(ep=1, msg="", msg_t=0, h=0.3, v=0.0, depth=None)
    frames = []
    orig_step = env.step
    k = [0]

    def rec_step(a):
        out = orig_step(a)
        k[0] += 1
        if k[0] % 2 == 0:  # 25 fps of the 50 Hz low level = real time
            show_goal()
            f = env.unwrapped.sim.visualizers[0].render_rgb_array()
            if f is not None:
                f = np.ascontiguousarray(f[..., :3])
                put = lambda s, y, c=(255, 255, 255), sc=0.8: cv2.putText(f, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 2, cv2.LINE_AA)  # noqa: E731
                put(f"HiPAN nav {label}", 40, sc=0.9)
                put(f"episode {hud['ep']}  room difficulty {diff0():.2f}  goal dist {float(venv._goal_dist()[0]):.1f} m", 75)
                put(f"cmd height {hud['h']:.2f} m   speed {hud['v']:.2f} m/s", 108)
                if hud["msg_t"] > 0:
                    col = (80, 255, 80) if hud["msg"] == "SUCCESS" else (255, 80, 80)
                    put(hud["msg"], 160, col, 1.6)
                    hud["msg_t"] -= 1
                if hud["depth"] is not None:
                    dimg = cv2.applyColorMap((255 * (1 - hud["depth"])).astype(np.uint8), cv2.COLORMAP_INFERNO)
                    dimg = cv2.resize(dimg, (336, 192), interpolation=cv2.INTER_NEAREST)[..., ::-1]
                    H, W = f.shape[:2]
                    f[H - 212:H - 20, W - 356:W - 20] = dimg
                    cv2.putText(f, "D435i depth (input)", (W - 356, H - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(f)
        return out

    env.step = rec_step
    steps, done_eps = 0, 0
    while done_eps < args.episodes and steps < args.max_s * 10 * args.episodes:
        with torch.no_grad():
            if args.student:
                a, _, h = net.step(obs["depth"], torch.cat([obs["nav"], obs["est"]], 1), h)
                hud["depth"] = obs["depth"][0, 0].cpu().numpy()
            else:
                a = actor(obs)
        obs, _, done, ex = venv.step(a)
        hud["h"], hud["v"] = float(venv.cmd[0, 3]), float(obs["priv"][0, :2].norm())
        steps += 1
        if args.student:
            h = h * (~done.bool()).float()[:, None]
        if bool(done[0]):
            r = "SUCCESS" if bool(ex["success"][0]) else ("TIMEOUT" if bool(ex["timeout"][0]) else "FALL")
            print(f"episode {hud['ep']}: {r}", flush=True)
            hud.update(msg=r, msg_t=40, ep=hud["ep"] + 1)
            done_eps += 1
            for _ in range(20):  # hold result banner on the last frame
                if frames:
                    frames.append(frames[-1])
            if done_eps < args.episodes:
                reroll(done_eps)
                obs = venv._observe()
                if args.student:
                    h[0] = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {args.out}: {len(frames)} frames", flush=True)
    env.close()
