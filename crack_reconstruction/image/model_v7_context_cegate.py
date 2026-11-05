from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = Block(3, 16)
        self.e2 = Block(16, 32)
        self.e3 = Block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.u1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d2 = Block(64, 32)
        self.d1 = Block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        z = self.d2(torch.cat([self.u2(c), b], 1))
        z = self.d1(torch.cat([self.u1(z), a], 1))
        return self.out(z)


class ContextCEGate(nn.Module):
    """Frozen DUTR-v7 2-D conservative residual gate."""

    def __init__(self, width: int = 24, max_corr: float = 1.0):
        super().__init__()
        self.max_corr = float(max_corr)
        self.stem = nn.Sequential(
            nn.Conv2d(8, width, 3, padding=1, bias=False),
            nn.GroupNorm(6, width),
            nn.GELU(),
        )
        self.d1 = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.d3 = nn.Conv2d(
            width, width, 3, padding=3, dilation=3, groups=width, bias=False
        )
        self.mix = nn.Sequential(
            nn.Conv2d(width * 2, width, 1, bias=False),
            nn.GroupNorm(6, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(6, width),
            nn.GELU(),
        )
        self.head = nn.Conv2d(width, 1, 1)
        self.gate = nn.Conv2d(width, 1, 1)
        self.scale = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        sup = F.avg_pool2d(p, 7, 1, 3)
        unc = 1.0 - (2.0 * q - 1.0).abs()
        qp = q - p
        cq = c - q

        gray = rgb.mean(1, keepdim=True)
        dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        edge = (dx + dy).clamp(0, 1)

        x = torch.cat([p, q, c, qp, cq, sup, unc, edge], 1)
        z = self.stem(x)
        z = self.mix(torch.cat([self.d1(z), self.d3(z)], 1))
        local_gate = torch.sigmoid(self.gate(z)) * (0.35 + 0.65 * unc)
        corr = (
            self.max_corr
            * torch.tanh(self.head(z))
            * local_gate
            * torch.tanh(self.scale)
        )
        ql = torch.logit(q.clamp(1e-4, 1 - 1e-4))
        return ql + corr
