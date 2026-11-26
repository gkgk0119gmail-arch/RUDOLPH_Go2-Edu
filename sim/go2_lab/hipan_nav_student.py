"""Deployable HiPAN high-level student: D435i depth + proprioception + goal -> command c, with a GRU memory.

Inputs per 0.1 s step:
  * depth [1, 64, 112]: distance to the image plane / 4 m, clipped to [0, 1] (D435i 87 x 58 deg, 13 deg down)
  * x (71) = ``nav`` (low-level student observation 62 incl. c_{t-1}, goal (3) in the yaw frame, desired speed 1)
    + the low student's motion-state estimate x_m (5)
Outputs: c (5, normalised to [-1, 1] like the teacher's actions) and an estimate of the teacher's map latent z_s.
"""
import torch
from torch import nn

D1 = 71
HW = (64, 112)


class RunningNorm(nn.Module):
    def __init__(self, d, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("n", torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        b = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.n + b
        d = m - self.mean
        self.mean += d * b / tot
        self.var = (self.var * self.n + v * b + d.square() * self.n * b / tot) / tot
        self.n = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var + self.eps).sqrt()).clamp(-5, 5)


class NavStudent(nn.Module):
    def __init__(self, d1=D1, z_dim=256, hid=256):
        super().__init__()
        self.norm = RunningNorm(d1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ELU(), nn.Flatten(),
            nn.Linear(64 * 4 * 7, 128), nn.ELU())
        self.fc = nn.Sequential(nn.Linear(d1, 128), nn.ELU())
        self.gru = nn.GRUCell(256, hid)
        self.z = nn.Linear(hid, z_dim)
        self.pi = nn.Sequential(nn.Linear(hid + z_dim + d1, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                nn.Linear(128, 5))
        self.hid = hid

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hid, device=device)

    def step(self, depth, x, h):
        """One step: depth [N, 1, H, W], x [N, d1], h [N, hid] -> (c [N, 5], z [N, z_dim], h)."""
        xn = self.norm(x)
        h = self.gru(torch.cat([self.cnn(depth), self.fc(xn)], 1), h)
        z = self.z(h)
        return self.pi(torch.cat([h, z, xn], 1)), z, h

    def unroll(self, depth, x, h, reset):
        """[T, N, ...] sequences; ``reset`` [T, N] zeroes the memory before step t."""
        A, Z = [], []
        for t in range(depth.shape[0]):
            h = h * (~reset[t]).float()[:, None]
            a, z, h = self.step(depth[t], x[t], h)
            A.append(a); Z.append(z)
        return torch.stack(A), torch.stack(Z), h
