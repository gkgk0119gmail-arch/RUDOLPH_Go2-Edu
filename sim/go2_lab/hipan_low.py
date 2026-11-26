"""HiPAN low-level locomotion teacher (arXiv 2604.26504, Sec. IV-B, Tables I and III) on our Go2 + XT16 + D435i.

Command c = [vx, vy, wz, h, roll]: h = base height above the ground under the trunk, roll in the world frame.
Teacher input (privileged, PPO):
  * ``policy`` (1D): proprioception o_p = [q, qd, foot positions (body frame), foot contacts, roll/pitch, w, a_{t-1}],
    command c, motion state x_m = [v_B, h, roll], domain parameters (payload, CoM offset, friction, motor gain scale)
  * ``heightmap`` (2D, CNN encoder): terrain heights on x, y in [-2, 2] m at 0.1 m (41 x 41) around the robot
Student (scripts/distill_hipan_low.py) sees only the proprioceptive part (``student`` group, 50-step history).

Deviations from the paper (not specified there, or differences of our robot):
  * height range 0.12-0.36 m (paper 0.1-0.4 on a Go1); roll and velocity ranges grow from a small start to the paper's
    range with a simple grid-adaptive curriculum (Margolis et al.) driven by posture tracking
  * friction is sampled per shape (Isaac Lab buckets); the teacher sees the mean of its robot
  * motor torque scale is approximated by scaling PD stiffness and damping
  * total reward clipped at 0 per step (legged_gym convention), penalty weights ramp from 10 % to 100 % over the
    first 1,500 iterations, initial action std 0.5: without these the penalty-heavy Table III gives no learning signal
  * the payload is our measured sensor payload (1.025 kg) + random extra 0-3 kg (paper: payload 0-3 kg)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from .crouch import GO2_XT16_USD, PAYLOAD, BodyHeightCommandCfg, body_height

HMAP = dict(size=4.0, res=0.1)  # x, y in [-2, 2] m
N_HMAP = int(round(HMAP["size"] / HMAP["res"])) + 1
FEET = ".*_foot"
KP_NOMINAL = 25.0  # UNITREE_GO2_CFG DC motor stiffness


# --------------------------------------------------------------------------------------------- helpers
def roll_world(env, asset_name: str = "robot") -> torch.Tensor:
    r, _, _ = euler_xyz_from_quat(env.scene[asset_name].data.root_quat_w.torch)
    return r


def roll_pitch(env, asset_name: str = "robot") -> torch.Tensor:
    r, p, _ = euler_xyz_from_quat(env.scene[asset_name].data.root_quat_w.torch)
    return torch.stack([r, p], 1)


def feet_pos_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    d = env.scene[asset_cfg.name].data
    rel = d.body_pos_w.torch[:, asset_cfg.body_ids] - d.root_pos_w.torch[:, None]
    q = d.root_quat_w.torch[:, None].expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape[0], -1)


def feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    f = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (f.norm(dim=-1) > 1.0).float()


def motion_state(env) -> torch.Tensor:
    """x_m = [v_B (3), h, roll]: privileged for the teacher, estimated by the student."""
    v = env.scene["robot"].data.root_lin_vel_b.torch
    return torch.cat([v, body_height(env)[:, None], roll_world(env)[:, None]], 1)


def domain_params(env) -> torch.Tensor:
    """[base mass, base CoM (3), friction, motor gain scale (12)] as seen by the simulator."""
    r = env.scene["robot"]
    b = r.body_names.index("base")
    from isaaclab.actuators.newton import read_group_parameter

    coll = r.actuators
    native = getattr(coll, "_native_group_names", set())
    k = torch.cat([(read_group_parameter(coll, n, "controller", "kp") if n in native else a.stiffness)
                   .reshape(env.num_envs, -1).to(env.device) for n, a in coll.items()], 1) / KP_NOMINAL
    fr = getattr(env, "_hipan_friction", torch.zeros(env.num_envs, device=env.device))
    return torch.cat([r.data.body_mass.torch[:, b:b + 1], r.data.body_com_pose_b.torch[:, b, :3], fr[:, None], k], 1)


def heightmap(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N, 1, 41, 41] terrain height relative to the base, clipped to [-1, 1] m."""
    s = env.scene.sensors[sensor_cfg.name]
    h = s.data.ray_hits_w.torch[..., 2] - env.scene["robot"].data.root_pos_w.torch[:, 2:3]
    h = torch.nan_to_num(h, nan=-1.0, posinf=-1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return h.reshape(-1, 1, N_HMAP, N_HMAP)


# --------------------------------------------------------------------------------------------- commands
class RollCommand(CommandTerm):
    """World-frame roll target, uniform in [-r, r]; r is set by the curriculum."""

    cfg: RollCommandCfg

    def __init__(self, cfg: RollCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.r = cfg.start
        self.target = torch.zeros(self.num_envs, 1, device=self.device)
        self.err = 1.0
        self.metrics["error_rad"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _resample_command(self, env_ids):
        env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        t = torch.empty(len(env_ids), device=self.device).uniform_(-self.r, self.r)
        level = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_level_envs
        self.target[env_ids, 0] = torch.where(level, torch.zeros_like(t), t)

    def _update_command(self):
        pass

    def _update_metrics(self):
        e = (roll_world(self._env) - self.target[:, 0]).abs()
        self.metrics["error_rad"] = e
        ok = self._env.episode_length_buf > 50
        if ok.any():
            m = float(e[ok].mean())
            self.err = m if self.err >= 1.0 else 0.99 * self.err + 0.01 * m


@configclass
class RollCommandCfg(CommandTermCfg):
    class_type: type = RollCommand
    start: float = 0.3
    full: float = 1.0
    rel_level_envs: float = 0.3


def grid_adaptive(env, env_ids, v_start, v_full, h_err: float, r_err: float, step: float, min_steps: int):
    """Grow velocity and roll ranges from ``start`` to ``full`` while height and roll are tracked well."""
    st = env.__dict__.setdefault("_hipan_cur", {"level": 0.0, "since": 0})
    st["since"] += 1
    h = env.command_manager.get_term("base_height")
    r = env.command_manager.get_term("base_roll")
    if st["since"] >= min_steps and h._err < h_err and r.err < r_err and st["level"] < 1.0:
        st["level"] = min(1.0, st["level"] + step)
        st["since"] = 0
        h._err = r.err = 1.0
    a = st["level"]
    v = env.command_manager.get_term("base_velocity").cfg.ranges
    v.lin_vel_x, v.lin_vel_y, v.ang_vel_z = (
        tuple(s + a * (f - s) for s, f in zip(v_start[i], v_full[i])) for i in range(3))
    r.r = r.cfg.start + a * (r.cfg.full - r.cfg.start)
    return torch.tensor(a)


def penalty_ramp(env, env_ids, start: float, steps: int):
    """Scale every negative reward weight from ``start`` to 1 over ``steps`` environment steps."""
    rm = env.reward_manager
    base = env.__dict__.setdefault("_hipan_w0", {n: rm.get_term_cfg(n).weight for n in rm.active_terms})
    f = min(1.0, start + (1.0 - start) * env.common_step_counter / steps)
    for n, w in base.items():
        if w < 0:
            rm.get_term_cfg(n).weight = w * f
    return torch.tensor(f)


# --------------------------------------------------------------------------------------------- events
def add_payload_random(env, env_ids, asset_cfg: SceneEntityCfg, mass, com, inertia, extra_range):
    """Exact sensor payload + a random extra point mass at the base CoM (paper: payload 0-3 kg)."""
    a = env.scene[asset_cfg.name]
    b = asset_cfg.body_ids
    ids = torch.arange(env.num_envs, device=env.device)
    m0 = a.data.body_mass.torch[:, b].clone()
    pose0 = a.data.body_com_pose_b.torch[:, b].clone()
    I0 = a.data.body_inertia.torch[:, b].clone().reshape(-1, 1, 3, 3)
    c0 = pose0[..., :3]
    cp = torch.tensor(com, device=m0.device, dtype=m0.dtype).expand_as(c0)
    me = torch.empty_like(m0).uniform_(*extra_range)
    ms = torch.full_like(m0, mass)
    m = m0 + ms + me
    c = (m0[..., None] * c0 + ms[..., None] * cp + me[..., None] * c0) / m[..., None]

    def shift(mm, d):
        eye = torch.eye(3, device=d.device)
        return mm[..., None, None] * ((d * d).sum(-1)[..., None, None] * eye - d[..., :, None] * d[..., None, :])

    Ip = torch.tensor(inertia, device=m0.device, dtype=m0.dtype)
    I = I0 + shift(m0, c0 - c) + Ip + shift(ms, cp - c) + shift(me, c0 - c)
    pose0[..., :3] = c
    a.set_masses_index(masses=m, body_ids=b, env_ids=ids)
    a.set_coms_index(coms=pose0, body_ids=b, env_ids=ids)
    a.set_inertias_index(inertias=I.reshape(-1, 1, 9), body_ids=b, env_ids=ids)


def record_friction(env, env_ids, asset_cfg: SceneEntityCfg):
    """Mean static friction of each robot's shapes after ``physics_material`` (PhysX tensor view), for the teacher."""
    a = env.scene[asset_cfg.name]
    try:
        import warp as wp

        m = wp.to_torch(a.root_view.get_material_properties())
        env._hipan_friction = m[..., 0].float().mean(1).to(env.device)
    except Exception as e:  # other backends: friction stays unobserved (zeros)
        print("record_friction: not available,", e)


# --------------------------------------------------------------------------------------------- rewards (Table III)
# Tracking terms use the squared error, exp(-e^2/s): Table III writes ||.||^2 for velocity; the other exponents are
# lost in the PDF text, and |e|/0.0025 m gave no height gradient (teacher 1 ignored the height command, 0.35 m always).
def _t(x):
    import warp as wp

    return x if torch.is_tensor(x) else (x.torch if hasattr(x, "torch") else wp.to_torch(x))


def feet_air_time_moving(env, command_name, sensor_cfg: SceneEntityCfg, threshold: float):
    """Gait term (not in HiPAN Table III): sum over feet of (air time - threshold) at touchdown, only while a
    velocity OR yaw command is given (Isaac Lab's version ignores yaw, so turning on the spot could still shuffle)."""
    cs = env.scene.sensors[sensor_cfg.name]
    first = _t(cs.compute_first_contact(env.step_dt))[:, sensor_cfg.body_ids].float()
    air = _t(cs.data.last_air_time)[:, sensor_cfg.body_ids]
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd[:, :2].norm(dim=1) > 0.1) | (cmd[:, 2].abs() > 0.2)
    return ((air - threshold) * first).sum(1) * moving.float()


def feet_slide_l1(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Gait term: horizontal speed of feet that are in contact (dragging the feet instead of stepping)."""
    f = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w_history)[:, :, sensor_cfg.body_ids].norm(dim=-1)
    contact = f.amax(1) > 1.0
    v = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (v * contact.float()).sum(1)


def track_lin_vel(env, command_name, scale):
    v = env.scene["robot"].data.root_lin_vel_b.torch[:, :2]
    return torch.exp(-(env.command_manager.get_command(command_name)[:, :2] - v).square().sum(1) / scale)


def track_yaw(env, command_name, scale):
    w = env.scene["robot"].data.root_ang_vel_b.torch[:, 2]
    return torch.exp(-(env.command_manager.get_command(command_name)[:, 2] - w).square() / scale)


def track_height(env, command_name, scale):
    return torch.exp(-(env.command_manager.get_command(command_name)[:, 0] - body_height(env)).square() / scale)


def track_roll(env, command_name, scale):
    return torch.exp(-(env.command_manager.get_command(command_name)[:, 0] - roll_world(env)).square() / scale)


def smooth_action(env):
    a = env.action_manager.action
    h = env.__dict__.setdefault("_hipan_a2", torch.zeros_like(a))
    out = (a - 2 * env.action_manager.prev_action + h).square().sum(1)
    env._hipan_a2 = env.action_manager.prev_action.clone()
    return out


def pitch_abs(env):
    _, p, _ = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w.torch)
    return p.abs()


def joint_limit_any(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return (mdp.joint_pos_limits(env, asset_cfg) > 0).float()


def collision_any(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    f = env.scene[sensor_cfg.name].data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids].norm(dim=-1)
    return (f.max(1)[0] > threshold).any(1).float()


# --------------------------------------------------------------------------------------------- task
@configclass
class TeacherObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    feet_pos = ObsTerm(func=feet_pos_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    feet_contact = ObsTerm(func=feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    roll_pitch = ObsTerm(func=roll_pitch)
    ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    actions = ObsTerm(func=mdp.last_action)
    vel_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    height_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
    roll_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_roll"})
    motion = ObsTerm(func=motion_state)
    domain = ObsTerm(func=domain_params)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StudentObsCfg(ObsGroup):
    """What the real robot has (same order as the first part of the teacher's group), with sensor noise."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
    feet_pos = ObsTerm(func=feet_pos_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
                       noise=Unoise(n_min=-0.01, n_max=0.01))
    feet_contact = ObsTerm(func=feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    roll_pitch = ObsTerm(func=roll_pitch, noise=Unoise(n_min=-0.05, n_max=0.05))
    ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
    actions = ObsTerm(func=mdp.last_action)
    vel_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    height_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
    roll_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_roll"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class HeightmapObsCfg(ObsGroup):
    hmap = ObsTerm(func=heightmap, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HipanLowEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        self.scene.robot.spawn.usd_path = GO2_XT16_USD
        # terrain: flat, stairs, holes, slopes (paper), sized for the Go2
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(proportion=0.2, step_height_range=(0.02, 0.14),
                                                              step_width=0.3, platform_width=3.0, border_width=1.0),
            "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.02, 0.14), step_width=0.3, platform_width=3.0, border_width=1.0),
            "holes": terrain_gen.MeshGapTerrainCfg(proportion=0.1, gap_width_range=(0.05, 0.2), platform_width=2.5),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.0, 0.4),
                                                           platform_width=2.0, border_width=0.25),
            "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.0, 0.4),
                                                                       platform_width=2.0, border_width=0.25),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.15, noise_range=(0.01, 0.06),
                                                           noise_step=0.01, border_width=0.25),
        }
        # sensors
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=HMAP["res"],
                                                                        size=[HMAP["size"], HMAP["size"]])
        self.scene.ground_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base", offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw", pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[0.3, 0.15]),
            debug_vis=False, mesh_prim_paths=["/World/ground"])
        # commands
        v = self.commands.base_velocity
        v.heading_command, v.rel_heading_envs, v.rel_standing_envs = False, 0.0, 0.1
        v.ranges.lin_vel_x, v.ranges.lin_vel_y, v.ranges.ang_vel_z = (-0.5, 0.8), (-0.3, 0.3), (-1.0, 1.0)
        self.commands.base_height = BodyHeightCommandCfg(resampling_time_range=(4.0, 8.0), range=(0.12, 0.36),
                                                         curriculum=False, rel_edge=0.0)
        self.commands.base_roll = RollCommandCfg(resampling_time_range=(4.0, 8.0))
        self.curriculum.command_levels = CurrTerm(func=grid_adaptive, params={
            "v_start": ((-0.5, 0.8), (-0.3, 0.3), (-1.0, 1.0)), "v_full": ((-1.5, 1.5), (-1.0, 1.0), (-1.5, 1.5)),
            "h_err": 0.03, "r_err": 0.08, "step": 0.1, "min_steps": 480})
        self.curriculum.penalty_scale = CurrTerm(func=penalty_ramp, params={"start": 0.1, "steps": 24 * 1500})
        # observations
        self.observations.policy = TeacherObsCfg()
        self.observations.heightmap = HeightmapObsCfg()
        self.observations.student = StudentObsCfg()
        # domain randomisation (Table I, training distribution)
        ev = self.events
        f = ev.physics_material.params
        f["static_friction_range"], f["dynamic_friction_range"], f["restitution_range"] = (0.7, 1.2), (0.7, 1.2), (0.0, 0.1)
        ev.record_friction = EventTerm(func=record_friction, mode="startup",
                                       params={"asset_cfg": SceneEntityCfg("robot")})
        ev.add_base_mass = None
        ev.base_com.params["com_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}
        ev.payload = EventTerm(func=add_payload_random, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass": PAYLOAD["mass"], "com": PAYLOAD["com"],
            "inertia": PAYLOAD["inertia_about_com"], "extra_range": (0.0, 3.0)})
        ev.motor = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1), "operation": "scale"})
        ev.push_robot.interval_range_s = (10.0, 10.0)
        # rewards: Table III only
        self.rewards = HipanLowRewardsCfg()
        # terminations: trunk or head on anything
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", "Head_.*"]


@configclass
class HipanLowRewardsCfg:
    track_vel = RewTerm(func=track_lin_vel, weight=0.4, params={"command_name": "base_velocity", "scale": 0.25})
    track_yaw = RewTerm(func=track_yaw, weight=0.2, params={"command_name": "base_velocity", "scale": 0.25})
    track_height = RewTerm(func=track_height, weight=0.2, params={"command_name": "base_height", "scale": 0.0025})
    track_roll = RewTerm(func=track_roll, weight=0.2, params={"command_name": "base_roll", "scale": 0.05})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    smooth = RewTerm(func=smooth_action, weight=-0.1)
    pitch = RewTerm(func=pitch_abs, weight=-0.5)
    vz = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    wxy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1e-3)
    qd = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    qdd = RewTerm(func=mdp.joint_acc_l2, weight=-2e-7)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_limit = RewTerm(func=joint_limit_any, weight=-10.0)
    collision = RewTerm(func=collision_any, weight=-10.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"])})


@configclass
class HipanLowFTEnvCfg(HipanLowEnvCfg):
    """Fine-tune from a trained teacher: penalties at full weight from the first step."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.penalty_scale.params["start"] = 1.0
        # teacher 1 + squared reward plateaued at 9 cm height error (crouching costs more in penalties than 0.2 pays):
        # lower the height range step by step (as in our crouch stage 2, which reached 0.12 m) and weigh height higher
        h = self.commands.base_height
        h.curriculum, h.start_low, h.step, h.advance_error, h.rel_edge = True, 0.30, 0.01, 0.025, 0.2
        self.rewards.track_height.weight = 0.6


@configclass
class HipanLowGaitEnvCfg(HipanLowFTEnvCfg):
    """Fine-tune from teacher_ft2 with gait terms. On the real Go2 the Table III policy turned and side-stepped by
    sliding its feet on the floor (Table III has no air-time / slip term; every penalty favours small joint motion)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time = RewTerm(func=feet_air_time_moving, weight=0.5, params={
            "command_name": "base_velocity", "threshold": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
        self.rewards.feet_slide = RewTerm(func=feet_slide_l1, weight=-0.25, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
        # real Go2 walked at ~0.70x the commanded vx / wz (sim ~0.87x at 0.4 m/s): exp(-e^2/0.25) pays 88 % of the
        # reward at 70 % tracking. A narrower kernel makes the last 30 % worth it.
        self.rewards.track_vel.params["scale"] = 0.1
        self.rewards.track_yaw.params["scale"] = 0.1


@configclass
class HipanLowEnvCfg_PLAY(HipanLowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.events.push_robot = None


@configclass
class HipanLowPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "hipan_low"
    obs_groups = {"actor": ["policy", "heightmap"], "critic": ["policy", "heightmap"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.5),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2],
                                        activation="elu", global_pool="none", flatten=True))
    critic = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2],
                                        activation="elu", global_pool="none", flatten=True))
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5,
        num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01,
        max_grad_norm=1.0)


class HipanEnv:
    """ManagerBasedRLEnv whose per-step total reward is clipped at 0 (legged_gym ``only_positive_rewards``).

    HiPAN's Table III is mostly penalties; without the clip an early policy maximises return by falling at once.
    The paper does not state the clip (Isaac Gym + legged_gym-style code is assumed)."""

    def __new__(cls, *args, **kwargs):
        from isaaclab.envs import ManagerBasedRLEnv

        class _Env(ManagerBasedRLEnv):
            def load_managers(self):
                super().load_managers()
                raw = self.reward_manager.compute
                self.reward_manager.compute = lambda dt: raw(dt).clamp(min=0.0)

        return _Env(*args, **kwargs)
