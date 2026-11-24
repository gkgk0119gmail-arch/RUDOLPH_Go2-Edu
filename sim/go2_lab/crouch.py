"""Crouch walking: the posture-walk task + a body-height command, on the real robot's sensor payload.

Stage 1 of the low-level teacher: height only (pitch and roll commands stay 0). Observation = the 50 entries of the
posture-walk policy + commanded height, so its weights can be extended (scripts/extend_policy.py).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .locomotion import _play
from .posture import Go2PostureWalkEnvCfg, Go2PostureWalkPPORunnerCfg

ASSETS = Path(__file__).resolve().parents[1] / "assets"
GO2_XT16_USD = str(ASSETS / "go2_xt16" / "go2_xt16.usda")
PAYLOAD = json.loads((ASSETS / "go2_xt16" / "geometry.json").read_text())["payload"]


def body_height(env, asset_name: str = "robot", sensor_name: str = "ground_scan") -> torch.Tensor:
    """Base height above the mean ground under the trunk [m]."""
    z = env.scene[asset_name].data.root_pos_w.torch[:, 2]
    hit = env.scene[sensor_name].data.ray_hits_w.torch[..., 2]
    ok = torch.isfinite(hit)
    ground = torch.where(ok, hit, torch.zeros_like(hit)).sum(1) / ok.sum(1).clamp(min=1)
    return z - ground


class BodyHeightCommand(CommandTerm):
    """Target height, uniform in [low, high]; ``low`` steps down while the current range is tracked well."""

    cfg: BodyHeightCommandCfg

    def __init__(self, cfg: BodyHeightCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.target = torch.full((self.num_envs, 1), cfg.nominal, device=self.device)
        self.low = cfg.start_low if cfg.curriculum else cfg.range[0]
        self._err = 1.0  # running mean tracking error [m]
        self._since = 0
        self.metrics["error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["low_m"] = torch.full((self.num_envs,), self.low, device=self.device)

    def __str__(self) -> str:
        return f"BodyHeightCommand: {self.low:.2f}-{self.cfg.range[1]:.2f} m"

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _resample_command(self, env_ids: Sequence[int] | slice):
        env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]  # Isaac Lab >= 3.0.0 may pass a slice
        u = torch.empty(len(env_ids), device=self.device)
        h = u.uniform_(self.low, self.cfg.range[1])
        edge = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_edge
        self.target[env_ids, 0] = torch.where(edge, torch.full_like(h, self.low), h)

    def _update_command(self):
        if not self.cfg.curriculum:
            return
        self._since += 1
        if self._since >= self.cfg.min_steps and self._err < self.cfg.advance_error and self.low > self.cfg.range[0]:
            self.low = max(self.cfg.range[0], self.low - self.cfg.step)
            self._since = 0
            self._err = 1.0

    def _update_metrics(self):
        if "ground_scan" not in self._env.scene.sensors:
            return
        err = (body_height(self._env) - self.target[:, 0]).abs()
        self.metrics["error_m"] = err
        self.metrics["low_m"][:] = self.low
        settled = self._env.episode_length_buf > 50
        if settled.any():
            m = float(err[settled].mean())
            self._err = m if self._err >= 1.0 else 0.99 * self._err + 0.01 * m


@configclass
class BodyHeightCommandCfg(CommandTermCfg):
    class_type: type = BodyHeightCommand
    asset_name: str = "robot"
    range: tuple[float, float] = MISSING
    nominal: float = 0.30
    curriculum: bool = True
    start_low: float = 0.27
    step: float = 0.01
    min_steps: int = 480
    """Environment steps between two lowerings (480 = 20 PPO iterations of 24 steps)."""
    advance_error: float = 0.025
    rel_edge: float = 0.2
    """Fraction of commands at the lowest height, so the new edge is practised."""


def track_height_exp(env, command_name: str, std: float) -> torch.Tensor:
    err = body_height(env) - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-(err**2) / std**2)


def add_payload(env, env_ids, asset_cfg: SceneEntityCfg, mass: float, com, inertia):
    """Add a rigid payload (mass, CoM in the body frame, inertia about its CoM) to one body, exactly."""
    asset = env.scene[asset_cfg.name]
    b = asset_cfg.body_ids
    ids = torch.arange(env.num_envs, device=env.device)
    m0 = asset.data.body_mass.torch[:, b].clone()                         # [N, 1]
    pose0 = asset.data.body_com_pose_b.torch[:, b].clone()                # [N, 1, 7]
    I0 = asset.data.body_inertia.torch[:, b].clone().reshape(-1, 1, 3, 3)
    c0, cp = pose0[..., :3], torch.tensor(com, device=m0.device, dtype=m0.dtype)
    m = m0 + mass
    c = (m0[..., None] * c0 + mass * cp) / m[..., None]

    def shift(mm, d):
        eye = torch.eye(3, device=d.device)
        return mm[..., None, None] * ((d * d).sum(-1)[..., None, None] * eye - d[..., :, None] * d[..., None, :])

    Ip = torch.tensor(inertia, device=m0.device, dtype=m0.dtype)
    I = I0 + shift(m0, c0 - c) + Ip + shift(torch.full_like(m0, mass), cp - c)
    pose0[..., :3] = c
    asset.set_masses_index(masses=m, body_ids=b, env_ids=ids)
    asset.set_coms_index(coms=pose0, body_ids=b, env_ids=ids)
    asset.set_inertias_index(inertias=I.reshape(-1, 1, 9), body_ids=b, env_ids=ids)


@configclass
class Go2CrouchWalkEnvCfg(Go2PostureWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # real sensor layout (colliders belong to the base body: touching an overhang ends the episode)
        self.scene.robot.spawn.usd_path = GO2_XT16_USD
        self.events.payload = EventTerm(func=add_payload, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass": PAYLOAD["mass"], "com": PAYLOAD["com"],
            "inertia": PAYLOAD["inertia_about_com"]})
        # ground under the trunk: reward / metric only, never observed
        self.scene.ground_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base", offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw", pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[0.3, 0.15]),
            debug_vis=False, mesh_prim_paths=["/World/ground"])
        # commands
        v = self.commands.base_velocity
        v.ranges.lin_vel_x, v.ranges.lin_vel_y, v.ranges.ang_vel_z = (-0.5, 0.8), (-0.3, 0.3), (-1.0, 1.0)
        v.rel_standing_envs = 0.15
        p = self.commands.base_posture
        p.pitch_range = p.roll_range = (0.0, 0.0)
        self.commands.base_height = BodyHeightCommandCfg(resampling_time_range=(4.0, 8.0), range=(0.16, 0.32))
        self.observations.policy.base_height = ObsTerm(func=mdp.generated_commands,
                                                       params={"command_name": "base_height"})
        rew = self.rewards
        rew.base_height = None  # fixed 0.30 m
        rew.track_height = RewTerm(func=track_height_exp, weight=2.0, params={"command_name": "base_height",
                                                                            "std": 0.05})


@configclass
class Go2CrouchRobustEnvCfg(Go2CrouchWalkEnvCfg):
    """Stage 2: continue stage 1 with friction / motor-gain randomisation, more pushes, calf-contact penalty, and a
    lower height limit (0.12 m) so that the true lowest usable height shows up."""

    def __post_init__(self):
        super().__post_init__()
        f = self.events.physics_material.params
        f["static_friction_range"], f["dynamic_friction_range"] = (0.3, 1.25), (0.3, 1.0)
        f["make_consistent"] = True
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.motor_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
        self.commands.base_height.range, self.commands.base_height.start_low = (0.12, 0.32), 0.16
        self.rewards.thigh_contact.params["sensor_cfg"].body_names = [".*_thigh", ".*_calf"]


class DelayedJointPositionAction(JointPositionAction):
    """Joint position targets that reach the motors 0..max physics steps late (one delay per robot and episode)."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        n = cfg.delay_steps[1] + 1
        self._buf = self._offset.unsqueeze(0).repeat(n, 1, 1)  # ring: newest at _i
        self._i = 0
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None, **kw):
        super().reset(env_ids, **kw) if env_ids is not None else super().reset()
        if not hasattr(self, "_rows"):
            return
        ids = self._rows if env_ids is None else self._rows[env_ids]
        self._delay[ids] = torch.randint(self.cfg.delay_steps[0], self.cfg.delay_steps[1] + 1, (len(ids),), device=self.device)
        self._buf[:, ids] = self._offset[ids]

    def apply_actions(self):
        self._i = (self._i + 1) % len(self._buf)
        self._buf[self._i] = self.processed_actions
        target = self._buf[(self._i - self._delay) % len(self._buf), self._rows]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class DelayedJointPositionActionCfg(mdp.JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    delay_steps: tuple[int, int] = (0, 8)
    """Delay range in physics steps (5 ms each)."""


def delayed_action(cfg, lo: int, hi: int):
    a = cfg.actions.joint_pos
    cfg.actions.joint_pos = DelayedJointPositionActionCfg(asset_name=a.asset_name, joint_names=a.joint_names,
        scale=a.scale, use_default_offset=True, delay_steps=(lo, hi))


@configclass
class Go2CrouchDelayEnvCfg(Go2CrouchRobustEnvCfg):
    """Stage 3: stage 2 + 0-40 ms control delay; head contact ends the episode; height limit back to 0.14 m."""

    def __post_init__(self):
        super().__post_init__()
        delayed_action(self, 0, 8)
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", "Head_.*"]
        self.commands.base_height.range, self.commands.base_height.start_low = (0.14, 0.32), 0.14


def swing_height(env, asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    """Squared error of the moving feet's height above the local ground, weighted by their horizontal speed."""
    r = env.scene[asset_cfg.name].data
    z = r.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    ground = r.root_pos_w.torch[:, 2] - body_height(env)
    v = r.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return ((z - ground[:, None] - target).clamp(max=0.0).square() * torch.tanh(2.0 * v)).sum(1)


@configclass
class Go2CrouchStepEnvCfg(Go2CrouchDelayEnvCfg):
    """Stage 4: stage 3 + lifted swing feet, terrain of random difficulty (the distance curriculum never left its
    lowest level at these speeds) and more / higher sills."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        self.scene.terrain.max_init_terrain_level = None
        g = self.scene.terrain.terrain_generator
        g.curriculum, g.difficulty_range = False, (0.0, 1.0)
        g.sub_terrains["flat"].proportion, g.sub_terrains["random_rough"].proportion = 0.3, 0.3
        g.sub_terrains["sills"].proportion = 0.4
        g.sub_terrains["sills"].grid_height_range = (0.01, 0.05)
        self.rewards.swing_height = RewTerm(func=swing_height, weight=-60.0, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "target": 0.022 + 0.06})


@configclass
class Go2CrouchWalkEnvCfg_PLAY(Go2CrouchWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.commands.base_height.curriculum = False


@configclass
class Go2CrouchWalkPPORunnerCfg(Go2PostureWalkPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_crouch_walk"
        self.max_iterations = 1500
