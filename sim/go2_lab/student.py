"""Deployable crouch-walk student: proprioceptive history (no base linear velocity) -> action + velocity estimate."""
import torch
from torch import nn

H, D = 50, 48  # history steps (1 s), entries per step = policy obs without base linear velocity (obs[3:51])


class Student(nn.Module):
    def __init__(self, h=H, d=D):
        super().__init__()
        self.h, self.d = h, d
        self.enc = nn.Sequential(nn.Linear(h * d, 512), nn.ELU(), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 32))
        self.vel = nn.Linear(32, 3)
        self.pi = nn.Sequential(nn.Linear(32 + 3 + d, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))

    def forward(self, hist):  # [N, H, D], newest last
        z = self.enc(hist.flatten(1))
        v = self.vel(z)
        return self.pi(torch.cat([z, v, hist[:, -1]], 1)), v


class History:
    def __init__(self, n, device, h=H, d=D):
        self.buf = torch.zeros(n, h, d, device=device)
        self.new = torch.ones(n, dtype=torch.bool, device=device)

    def push(self, obs):
        """obs: [N, 51] policy observation. After a reset the history is filled with the first observation."""
        x = obs[:, 3:51]
        self.buf = torch.cat([self.buf[:, 1:], x[:, None]], 1)
        if self.new.any():
            self.buf[self.new] = x[self.new][:, None]
        self.new[:] = False
        return self.buf

    def reset(self, done):
        self.new |= done.bool()


def mlp_actor(sd):
    """rsl_rl MLP actor (3 hidden ELU layers) from its state dict."""
    def act(x):
        for i in (0, 2, 4):
            x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
        return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]
    return act
