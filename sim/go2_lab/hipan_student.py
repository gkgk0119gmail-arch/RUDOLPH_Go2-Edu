"""Deployable HiPAN low-level student: proprioceptive history -> action + motion-state estimate.

Input per step (62, the ``student`` observation group of ``hipan_low.py``, in this order):
  joint_pos_rel (12), joint_vel_rel * 0.05 (12), feet positions in the body frame (12), foot contacts (4),
  roll/pitch (2), base angular velocity * 0.25 (3), last action (12), velocity command [vx, vy, wz] (3),
  height command (1), roll command (1).
Estimate (5): motion state x_m = [v_B (3), base height, world roll], the privileged part the teacher sees.
"""
import torch
from torch import nn

H, D, EST = 50, 62, 5  # history steps (1 s at 50 Hz), entries per step, estimated motion state


class HipanStudent(nn.Module):
    def __init__(self, h=H, d=D, est=EST):
        super().__init__()
        self.h, self.d, self.est = h, d, est
        self.enc = nn.Sequential(nn.Linear(h * d, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 64))
        self.motion = nn.Linear(64, est)
        self.pi = nn.Sequential(nn.Linear(64 + est + d, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))

    def forward(self, hist):  # [N, H, D], newest last
        z = self.enc(hist.flatten(1))
        m = self.motion(z)
        return self.pi(torch.cat([z, m, hist[:, -1]], 1)), m


class History:
    def __init__(self, n, device, h=H, d=D):
        self.buf = torch.zeros(n, h, d, device=device)
        self.new = torch.ones(n, dtype=torch.bool, device=device)

    def push(self, x):
        """x: [N, D] student observation. After a reset the history is filled with the first observation."""
        self.buf = torch.cat([self.buf[:, 1:], x[:, None]], 1)
        if self.new.any():
            self.buf[self.new] = x[self.new][:, None]
        self.new[:] = False
        return self.buf

    def reset(self, done):
        self.new |= done.bool()
