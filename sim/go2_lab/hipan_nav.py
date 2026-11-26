"""HiPAN high-level navigation (arXiv 2604.26504, Sec. IV-C) on top of our frozen low-level student.

High level at 10 Hz outputs c = [vx, vy, wz, h, roll] to the low-level student (50 Hz, weights fixed, 0-20 ms control
delay), which runs inside the environment, so tracking errors and delays are part of the dynamics.

Environment: procedural rooms (``nav_world.py``) with partition walls, doorways under low or tilted lintels, tables,
floating beams / low slabs (collapsed debris) and floor clutter. The goal is random for now (a start/goal pair
of the robot's tile); the goal selector will replace this later.

Teacher (PPO, privileged):
  * ``nav``  (1D): low-level proprioception o_p incl. c_{t-1} (62, the low student's input), relative (sub)goal in
    the yaw frame (3), desired speed (1)
  * ``priv`` (1D): motion state x_m = [v_B, h, roll] (5)
  * ``map``  (2D): elevation and clearance around the robot, x in [-1, 3] m, y in [-1.5, 1.5] m at 0.1 m (2 x 41 x 31)
Student (``scripts/distill_hipan_nav.py``): ``nav`` + estimated x_m (from the low student) + ``depth`` (D435i pose)
with a GRU; only the final goal.

PGCL (privileged guided curriculum): sub-goals every d m along the reference path; d starts at 1 m and grows by 1 m
each time all sub-goals are reached; once d exceeds the path, only the final goal is given; after 2 final-goal
successes in a row the robot gets a new task.

Rewards (Table IV of the paper; shaping details not given there are ours): arrival 5.0, progress 3.0 x distance
gained towards the current (sub)goal (ours: with the paper's 0.5 visitation weight the first teacher farmed the
visitation bonus and reached 1 % of the sub-goals), state visitation 0.2/sqrt(count) on a 0.5 m grid, speed 0.25*exp(-|v_des - |v||/0.3), command rate / smoothness -0.1,
tracking error -0.2, body velocity -0.1, nominal posture -0.04, command limit -2.5, collision -2.5.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from . import nav_world as nw
from .hipan_low import HipanLowEnvCfg, motion_state

# --------------------------------------------------------------------------------------------- terrain
_WORLD: dict = {}


class NavTerrainGenerator:
    """Drop-in for Isaac Lab's TerrainGenerator: builds a nav_world and exposes its mesh and tile origins."""

    def __init__(self, cfg: NavTerrainGeneratorCfg, device: str = "cpu"):
        self.cfg = cfg
        if cfg.scene == "beam_test":
            w = nw.build_beam_test()
            cfg.num_rows, cfg.num_cols = w["rows"], w["cols"]
        else:
            w = nw.build_world(cfg.num_rows, cfg.num_cols, seed=cfg.seed or 0, n_tasks=cfg.n_tasks,
                               difficulty=cfg.difficulty)
        self.terrain_mesh = nw.world_mesh(w)
        o = np.zeros((cfg.num_rows, cfg.num_cols, 3), np.float32)
        for r in range(cfg.num_rows):
            for c in range(cfg.num_cols):
                o[r, c, :2] = ((r + 0.5) * nw.TILE, (c + 0.5) * nw.TILE)
        self.terrain_origins = o
        self.flat_patches = {}
        _WORLD.clear()
        _WORLD.update(w)


@configclass
class NavTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = NavTerrainGenerator
    size: tuple[float, float] = (nw.TILE, nw.TILE)
    num_rows: int = 6
    num_cols: int = 8
    sub_terrains: dict = {}
    seed: int = 0
    n_tasks: int = 64
    difficulty: float | None = None  # None: grows with the row index 0 -> 1
    scene: str = "random"            # or "beam_test" (scripts/check_depth_overhead.py)


# --------------------------------------------------------------------------------------------- task state
MAP_X, MAP_Y = (-1.0, 3.0), (-1.5, 1.5)
N_MX, N_MY = 41, 31
VISIT = 0.5
R_ARRIVE = 0.25
V_DES = (0.3, 0.8)


class NavState:
    """Tasks, PGCL sub-goals, visitation counts and arrival bookkeeping for every robot (lives on the env)."""

    def __init__(self, env):
        w = _WORLD
        self.env, dev = env, env.device
        self.n = n = env.num_envs
        self.dev = dev
        self.elev = torch.tensor(w["elev"], device=dev)
        self.clear = torch.tensor(w["clear"], device=dev)
        tasks = w["tasks"]
        self.n_tiles, self.per = len(tasks), len(tasks[0]["start"])
        self.start = torch.tensor(np.concatenate([t["start"] for t in tasks]), device=dev, dtype=torch.float32)
        paths = [p for t in tasks for p in t["paths"]]
        pts, lens = nw.resample_paths(paths, 0.1, 200)
        self.pts, self.len = torch.tensor(pts, device=dev), torch.tensor(lens, device=dev)
        self.tile_origin = torch.tensor([[(t["tile"] // w["cols"]) * nw.TILE, (t["tile"] % w["cols"]) * nw.TILE]
                                         for t in tasks], device=dev, dtype=torch.float32)
        self.tile = torch.arange(n, device=dev) % self.n_tiles
        self.task = torch.zeros(n, dtype=torch.long, device=dev)
        self.d = torch.ones(n, device=dev)
        self.arc = torch.ones(n, device=dev)
        self.streak = torch.zeros(n, dtype=torch.long, device=dev)
        self.fails = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.limit = torch.full((n,), 10**9, dtype=torch.long, device=dev)
        self.v_des = torch.full((n,), 0.5, device=dev)
        self.cells = int(nw.TILE / VISIT)
        self.visits = torch.zeros(n, self.cells * self.cells, device=dev)
        self.arrivals = torch.zeros(n, device=dev)   # sub-goal/goal arrivals since the last high-level step
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ended_success = torch.zeros(n, dtype=torch.bool, device=dev)  # outcome of the episode reset last
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.final_only = False                      # evaluation / student: no sub-goals
        self.goal_override = None                    # [n, 2] world goal set from outside (goal selector)
        self.stats = dict(success=0.0, final=0.0, d=1.0)

    # ---- tasks
    def new_task(self, ids):
        k = torch.randint(0, self.per, (len(ids),), device=self.dev)
        self.task[ids] = self.tile[ids] * self.per + k
        self.d[ids] = 1.0
        self.streak[ids] = 0
        self.fails[ids] = 0

    def on_reset(self, ids):
        """PGCL update from the episode that just ended, then the start pose of the next one."""
        self.ended_success[ids] = self.success[ids]
        old = ids[~self.fresh[ids]]
        if len(old):
            succ = self.success[old]
            L = self.len[self.task[old]]
            final = self.d[old] >= L
            self.stats["success"] = 0.99 * self.stats["success"] + 0.01 * float(succ.float().mean())
            self.stats["final"] = 0.99 * self.stats["final"] + 0.01 * float(final.float().mean())
            self.streak[old] = torch.where(succ & final, self.streak[old] + 1, torch.where(succ, 0, self.streak[old]))
            self.d[old] = torch.where(succ & ~final, self.d[old] + 1.0, self.d[old])
            self.fails[old] = torch.where(succ, 0, self.fails[old] + 1)
            renew = old[(self.streak[old] >= 2) | (self.fails[old] >= 3)]
            if len(renew):
                self.new_task(renew)
        new = ids[self.fresh[ids]]
        if len(new):
            self.new_task(new)
            self.fresh[new] = False
        if self.final_only:
            self.d[ids] = 1e3
        self.arc[ids] = torch.minimum(self.d[ids], self.len[self.task[ids]])
        self.success[ids] = False
        self.t[ids] = 0
        L = self.len[self.task[ids]]
        self.limit[ids] = ((10.0 + 3.0 * L) / self.env.step_dt).long()
        self.v_des[ids] = torch.empty(len(ids), device=self.dev).uniform_(*V_DES)
        self.visits[ids] = 0.0
        self.stats["d"] = float(self.d.mean())
        return self.start[self.task[ids]]

    def switch_goal(self, ids):
        """Our extension: the goal changes mid-episode (as the goal selector will do). New goal = the goal of another
        task on the same tile, given directly (final stage); the time limit restarts."""
        k = torch.randint(0, self.per, (len(ids),), device=self.dev)
        self.task[ids] = self.tile[ids] * self.per + k
        self.arc[ids] = self.len[self.task[ids]]
        L = (self.goal_w()[ids] - self.env.scene["robot"].data.root_pos_w.torch[ids, :2]).norm(dim=1) * 1.5
        self.limit[ids] = self.t[ids] + ((10.0 + 3.0 * L) / self.env.step_dt).long()

    def goal_w(self):
        if self.goal_override is not None:
            return self.goal_override
        i = (self.arc / 0.1).round().long().clamp(0, self.pts.shape[1] - 1)
        return self.pts[self.task, i]

    def is_final(self):
        return self.goal_override is not None or self.arc >= self.len[self.task] - 1e-4

    # ---- called every low-level step (termination term, before resets)
    def step_low(self):
        pos = self.env.scene["robot"].data.root_pos_w.torch[:, :2]
        near = (self.goal_w() - pos).norm(dim=1) < R_ARRIVE
        final = self.is_final()
        sub = near & ~final
        self.arrivals += near.float()
        self.arc = torch.where(sub, torch.minimum(self.arc + self.d, self.len[self.task]), self.arc)
        self.success |= near & final
        self.t += 1
        return near & final

    # ---- observations / rewards helpers
    def goal_feat(self):
        r = self.env.scene["robot"].data
        g = self.goal_w() - r.root_pos_w.torch[:, :2]
        yaw = r.heading_w.torch
        c, s = torch.cos(yaw), torch.sin(yaw)
        x, y = c * g[:, 0] + s * g[:, 1], -s * g[:, 0] + c * g[:, 1]
        dist = torch.sqrt(x * x + y * y).clamp(min=1e-6)
        k = dist.clamp(max=5.0) / dist / 5.0
        return torch.stack([x * k, y * k, dist.clamp(max=10.0) / 10.0], 1)

    def visit(self, active):
        pos = self.env.scene["robot"].data.root_pos_w.torch[:, :2] - self.tile_origin[self.tile]
        ij = (pos / VISIT).long().clamp(0, self.cells - 1)
        cell = ij[:, 0] * self.cells + ij[:, 1]
        rows = torch.arange(self.n, device=self.dev)
        self.visits[rows, cell] += active.float()
        return self.visits[rows, cell].clamp(min=1.0)

    def local_map(self):
        """[N, 2, 41, 31]: elevation and clearance (m, capped at 1) in the robot's yaw frame."""
        r = self.env.scene["robot"].data
        p, yaw = r.root_pos_w.torch[:, :2], r.heading_w.torch
        gx = torch.linspace(*MAP_X, N_MX, device=self.dev)
        gy = torch.linspace(*MAP_Y, N_MY, device=self.dev)
        X, Y = torch.meshgrid(gx, gy, indexing="ij")
        c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
        wx = p[:, 0, None, None] + c * X - s * Y
        wy = p[:, 1, None, None] + s * X + c * Y
        ix, iy = (wx / nw.RES).floor().long(), (wy / nw.RES).floor().long()
        H, W = self.elev.shape
        inside = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
        ix, iy = ix.clamp(0, H - 1), iy.clamp(0, W - 1)
        e = torch.where(inside, self.elev[ix, iy], torch.ones_like(wx)).clamp(max=1.0)
        cl = torch.where(inside, self.clear[ix, iy], torch.zeros_like(wx)).clamp(max=1.0)
        return torch.stack([e, cl], 1)


def nav_state(env) -> NavState:
    if not hasattr(env, "_nav"):
        env._nav = NavState(env)
    return env._nav


# --------------------------------------------------------------------------------------------- MDP terms
def reset_to_task(env, env_ids, z: float = 0.40):
    ns = nav_state(env)
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    xy = ns.on_reset(ids)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-math.pi, math.pi)
    zero = torch.zeros_like(yaw)
    pose = torch.cat([xy, torch.full_like(yaw, z)[:, None], quat_from_euler_xyz(zero, zero, yaw)], 1)
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def goal_reached(env):
    return nav_state(env).step_low()


def nav_timeout(env):
    ns = nav_state(env)
    return ns.t >= ns.limit


# --------------------------------------------------------------------------------------------- env cfg
D435I_POS = (0.32, 0.0, 0.055)   # camera_link in assets/go2_xt16/go2_xt16.urdf
D435I_PITCH = 0.227              # rad, looking down
DEPTH_HW = (64, 112)
DEPTH_MAX = 4.0


def depth_camera_cfg(hw=DEPTH_HW) -> RayCasterCameraCfg:
    """D435i depth (87 x 58 deg) as a ray-cast camera against the static world mesh, 10 Hz."""
    f = 1.0
    ha = 2 * f * math.tan(math.radians(87 / 2))
    va = 2 * f * math.tan(math.radians(58 / 2))
    s = math.sin(D435I_PITCH / 2)
    return RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", mesh_prim_paths=["/World/ground"], update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=D435I_POS, rot=(0.0, s, 0.0, math.cos(D435I_PITCH / 2)),
                                            convention="world"),
        data_types=["distance_to_image_plane"], depth_clipping_behavior="max", max_distance=DEPTH_MAX,
        pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=f, horizontal_aperture=ha, vertical_aperture=va,
                                                     width=hw[1], height=hw[0]))


@configclass
class HipanNavEnvCfg(HipanLowEnvCfg):
    """Low-level scene and robot (same randomisation the low student was trained with) on navigation worlds."""

    def __post_init__(self):
        super().__post_init__()
        from .crouch import delayed_action

        self.scene.num_envs = 1024
        self.scene.terrain.terrain_generator = NavTerrainGeneratorCfg()
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        # ground under the trunk, cast from the base (not from above: tables and beams would be hit first)
        self.scene.ground_scan.offset.pos = (0.0, 0.0, 0.0)
        self.observations.policy = None
        self.observations.heightmap = None
        self.curriculum.terrain_levels = self.curriculum.command_levels = self.curriculum.penalty_scale = None
        for k in ("base_velocity", "base_height", "base_roll"):
            getattr(self.commands, k).resampling_time_range = (1e6, 1e6)
        self.events.reset_base = EventTerm(func=reset_to_task, mode="reset")
        self.events.push_robot = None
        delayed_action(self, 0, 4)  # 0-20 ms
        self.rewards = NavLowRewardsCfg()
        self.terminations.goal = DoneTerm(func=goal_reached)
        self.terminations.nav_timeout = DoneTerm(func=nav_timeout, time_out=True)
        self.episode_length_s = 60.0


@configclass
class NavLowRewardsCfg:
    alive = None  # rewards are computed by NavVecEnv at 10 Hz


@configclass
class HipanNavEnvCfg_DEPTH(HipanNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.depth = depth_camera_cfg()


# --------------------------------------------------------------------------------------------- 10 Hz wrapper
C_LO = (-0.5, -0.4, -1.2, 0.12, -0.5)
C_HI = (1.0, 0.4, 1.2, 0.36, 0.5)
C_NOM = (0.0, 0.0, 0.0, 0.30, 0.0)
CMD_SLICE = slice(57, 62)  # [vx, vy, wz, h, roll] inside the low student's 62-dim observation
COLLIDE = ["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"]

REWARD_W = dict(arrive=5.0, progress=3.0, visit=0.2, speed=0.25, rate=-0.1, smooth=-0.1, track=-0.2, body=-0.1, nominal=-0.04, upright=-0.3,
                limit=-2.5, collision=-2.5)


class NavVecEnv:
    """rsl_rl VecEnv: one high-level step (0.1 s) = ``decim`` low-level student steps."""

    def __init__(self, env, low_student: str, decim: int = 5, depth: bool = False):
        from rsl_rl.env import VecEnv  # noqa: F401  (interface reference)

        from .hipan_student import HipanStudent, History

        self.env = env
        u = self.unwrapped = env.unwrapped
        self.num_envs, self.device, self.cfg = u.num_envs, u.device, u.cfg
        self.num_actions = 5
        self.decim = decim
        self.max_episode_length = int(u.max_episode_length // decim)
        self.depth = depth
        ck = torch.load(low_student, map_location=self.device, weights_only=True)
        self.low = HipanStudent(d=ck["obs_dim"]).to(self.device).eval()
        self.low.load_state_dict(ck["student"])
        for p in self.low.parameters():
            p.requires_grad_(False)
        self.hist = History(self.num_envs, self.device, d=ck["obs_dim"])
        t = lambda v: torch.tensor(v, device=self.device)  # noqa: E731
        self.c_lo, self.c_hi, self.c_nom = t(C_LO), t(C_HI), t(C_NOM)
        self.a_nom = (self.c_nom - (self.c_lo + self.c_hi) / 2) / ((self.c_hi - self.c_lo) / 2)
        self.cmd = self.c_nom.repeat(self.num_envs, 1)
        self.a1 = self.a_nom.repeat(self.num_envs, 1)
        self.a2 = self.a1.clone()
        cs = u.scene["contact_forces"]
        self.coll_ids = cs.find_bodies(COLLIDE)[0]
        self.m_est = torch.zeros(self.num_envs, 5, device=self.device)
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        env.reset()
        self.ns = nav_state(u)
        self.prev_dist = self._goal_dist()
        self._obs = self._observe()

    # rsl_rl uses this for random episode starts; our episodes are bookkept by NavState, so it is a plain counter
    @property
    def episode_length_buf(self):
        return self.ep_len

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self.ep_len = v

    def get_observations(self):
        return self._obs

    def _goal_dist(self):
        return (self.ns.goal_w() - self.unwrapped.scene["robot"].data.root_pos_w.torch[:, :2]).norm(dim=1)

    def _open_overhead(self):
        """1 where the clearance over x -0.4..1.0 m, |y| <= 0.3 m (robot frame) is >= 0.6 m (stand + margin)."""
        cl = self.ns.local_map()[:, 1, 6:21, 12:19]
        return (cl.amin((1, 2)) >= 0.6).float()

    def _low_obs(self):
        o = self.unwrapped.obs_buf["student"].clone()
        o[:, CMD_SLICE] = self.cmd
        return o

    def _observe(self):
        from tensordict import TensorDict

        u, ns = self.unwrapped, self.ns
        nav = torch.cat([self._low_obs(), ns.goal_feat(), ns.v_des[:, None]], 1)
        d = dict(nav=nav, priv=motion_state(u), map=ns.local_map(), est=self.m_est.clone())
        if self.depth:
            z = u.scene["depth"].data.output["distance_to_image_plane"].torch
            d["depth"] = (torch.nan_to_num(z, nan=DEPTH_MAX, posinf=DEPTH_MAX).clamp(0, DEPTH_MAX) / DEPTH_MAX
                          ).permute(0, 3, 1, 2) if z.dim() == 4 else z
        return TensorDict(d, batch_size=[self.num_envs])

    def step(self, actions):
        u, ns, W = self.unwrapped, self.ns, REWARD_W
        a = actions.to(self.device)
        ac = a.clamp(-1.0, 1.0)
        self.cmd = (self.c_lo + self.c_hi) / 2 + ac * (self.c_hi - self.c_lo) / 2
        ns.arrivals.zero_()
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        tout = torch.zeros_like(done)
        coll = torch.zeros_like(done)
        cs = u.scene["contact_forces"]
        for _ in range(self.decim):
            with torch.no_grad():
                act, m = self.low(self.hist.push(self._low_obs()))
            self.m_est = m
            _, _, term, trunc, _ = self.env.step(act)
            f = cs.data.net_forces_w_history.torch[:, :, self.coll_ids].norm(dim=-1).amax(1)
            coll |= (f > 1.0).any(1) & ~done
            new = (term | trunc) & ~done
            tout |= new & trunc.bool() & ~term.bool()
            done |= term.bool() | trunc.bool()
            self.hist.reset(term | trunc)
        succ = done & ns.ended_success
        active = ~done
        r = u.scene["robot"].data
        v = r.root_lin_vel_b.torch
        w = r.root_ang_vel_b.torch
        from .crouch import body_height
        from .hipan_low import roll_world

        h, roll = body_height(u), roll_world(u)
        track = ((self.cmd[:, :2] - v[:, :2]).norm(dim=1) + 0.3 * (self.cmd[:, 2] - w[:, 2]).abs()
                 + 5.0 * (self.cmd[:, 3] - h).abs() + (self.cmd[:, 4] - roll).abs())
        dist = self._goal_dist()
        same = active & (ns.arrivals == 0)  # same (sub)goal as at the previous step
        progress = torch.where(same, (self.prev_dist - dist).clamp(-0.2, 0.2), torch.zeros_like(dist))
        self.prev_dist = dist
        terms = dict(
            arrive=ns.arrivals.clone(),
            progress=progress,
            visit=active.float() / ns.visit(active).sqrt(),
            speed=active.float() * torch.exp(-(ns.v_des - v[:, :2].norm(dim=1)).abs() / 0.3),
            rate=(ac - self.a1).square().sum(1),
            smooth=(ac - 2 * self.a1 + self.a2).square().sum(1),
            track=active.float() * track,
            body=active.float() * (v[:, 2].square() + 0.1 * w[:, :2].square().sum(1)),
            nominal=((self.cmd[:, 3] - C_NOM[3]) / 0.1).square() + (self.cmd[:, 4] / 0.3).square(),
            # v2: stand back up when nothing overhangs the body footprint or the next 1 m (privileged map, teacher only)
            upright=active.float() * self._open_overhead() * ((C_NOM[3] - self.cmd[:, 3]).clamp(min=0) / 0.1).square(),
            limit=(a.abs() - 1.0).clamp(min=0).sum(1),
            collision=coll.float(),
        )
        rew = sum(W[k] * x for k, x in terms.items())
        self.a2 = torch.where(done[:, None], self.a_nom, self.a1)
        self.a1 = torch.where(done[:, None], self.a_nom, ac)
        self.cmd = torch.where(done[:, None], self.c_nom, self.cmd)
        self.ep_len = torch.where(done, 0, self.ep_len + 1)
        self._obs = self._observe()
        ended = done.sum().clamp(min=1)
        log = {f"/nav_rew/{k}": float((W[k] * x).mean()) for k, x in terms.items()}
        log.update({"/nav/success_ema": ns.stats["success"], "/nav/final_stage_ema": ns.stats["final"],
                    "/nav/pgcl_d": ns.stats["d"]})
        if done.any():
            log["/nav/success_step"] = float(succ.sum() / ended)
            log["/nav/fall_step"] = float((done & ~succ & ~tout).sum() / ended)
        # success ends the episode without a terminal state: bootstrap it like a time-out, otherwise the positive
        # visitation/speed terms would teach the robot to avoid the goal
        extras = {"time_outs": tout | succ, "log": log, "success": succ, "done": done, "timeout": tout}
        return self._obs, rew, done.long(), extras

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------------------------- PPO teacher
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg  # noqa: E402

_CNN = dict(output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu",
            global_pool="none", flatten=True)


@configclass
class HipanNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hipan_nav"
    obs_groups = {"actor": ["nav", "priv", "map"], "critic": ["nav", "priv", "map"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.8),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(**_CNN))
    critic = RslRlCNNModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
                              cnn_cfg=RslRlCNNModelCfg.CNNCfg(**_CNN))
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5,
        num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.995, lam=0.95, desired_kl=0.01,
        max_grad_norm=1.0)
