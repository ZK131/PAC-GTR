from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DirectionalContextBlock(nn.Module):
    """Crack-oriented context block with horizontal, vertical and dilated branches."""

    def __init__(self, channels: int):
        super().__init__()
        c = channels
        self.h = nn.Conv2d(c, c, kernel_size=(1, 7), padding=(0, 3), groups=c, bias=False)
        self.v = nn.Conv2d(c, c, kernel_size=(7, 1), padding=(3, 0), groups=c, bias=False)
        self.d = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, groups=c, bias=False)
        self.l = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(4 * c, c, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.h(x), self.v(x), self.d(x), self.l(x)], dim=1)
        return self.act(x + self.mix(y))


class OmniTCR(nn.Module):
    """Topology/continuity-aware residual adapter on top of OmniCrack30k probabilities."""

    def __init__(self, base_channels: int = 32, max_residual: float = 4.0):
        super().__init__()
        c = base_channels
        self.max_residual = float(max_residual)

        self.stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
            nn.GELU(),
        )
        self.high = DirectionalContextBlock(c)
        self.mid = DirectionalContextBlock(c)
        self.low = DirectionalContextBlock(c)

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c, c, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
            nn.GELU(),
            DirectionalContextBlock(c),
        )

        self.residual_head = nn.Conv2d(c, 1, kernel_size=1)
        self.gate_head = nn.Conv2d(c, 1, kernel_size=1)
        self.skeleton_head = nn.Conv2d(c, 1, kernel_size=1)

    @staticmethod
    def prob_to_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, rgb: torch.Tensor, omni_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be [B,3,H,W], got {tuple(rgb.shape)}")
        if omni_prob.ndim != 4 or omni_prob.shape[1] != 1:
            raise ValueError(f"omni_prob must be [B,1,H,W], got {tuple(omni_prob.shape)}")
        if rgb.shape[-2:] != omni_prob.shape[-2:]:
            raise ValueError("rgb and omni_prob spatial sizes must match")

        p = omni_prob.clamp(1e-4, 1 - 1e-4)
        uncertainty = 1.0 - torch.abs(2.0 * p - 1.0)
        omni_logit = self.prob_to_logit(p)

        x = self.stem(torch.cat([rgb, p], dim=1))
        h = self.high(x)

        m = F.avg_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        m = self.mid(m)

        l = F.avg_pool2d(m, kernel_size=2, stride=2, ceil_mode=True)
        l = self.low(l)

        m_up = F.interpolate(m, size=h.shape[-2:], mode="bilinear", align_corners=False)
        l_up = F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([h, m_up, l_up], dim=1))

        learned_gate = torch.sigmoid(self.gate_head(feat))
        gate = learned_gate * uncertainty
        residual = self.max_residual * torch.tanh(self.residual_head(feat))
        refined_logit = omni_logit + gate * residual
        refined_prob = torch.sigmoid(refined_logit)

        return {
            "refined_logit": refined_logit,
            "refined_prob": refined_prob,
            "residual": residual,
            "gate": gate,
            "uncertainty": uncertainty,
            "skeleton_logit": self.skeleton_head(feat),
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
