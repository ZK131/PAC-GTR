from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from crack_reconstruction.image.model import DirectionalContextBlock


class OmniTCRV2(nn.Module):
    """Omni-TCR v2: dual-branch recovery/suppression refinement.

    Keeps the v1 backbone names for warm-start, then adds a probability-geometry
    stem and two asymmetric correction branches:
      - recovery branch: restores weak but locally supported crack pixels;
      - suppression branch: removes isolated / inconsistent false positives.
    """

    def __init__(self, base_channels: int = 32, max_residual: float = 4.0, max_dual: float = 2.5):
        super().__init__()
        c = base_channels
        self.max_residual = float(max_residual)
        self.max_dual = float(max_dual)

        # Same names/shapes as v1 for checkpoint warm-start.
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

        # New probability-geometry branch.
        self.geom_stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, c)), num_channels=c),
            nn.GELU(),
        )

        # Recovery and suppression are explicitly separated.
        self.recover_head = nn.Conv2d(c, 1, kernel_size=1)
        self.recover_gate_head = nn.Conv2d(c, 1, kernel_size=1)
        self.suppress_head = nn.Conv2d(c, 1, kernel_size=1)
        self.suppress_gate_head = nn.Conv2d(c, 1, kernel_size=1)

        # Boundary/centerline supervision heads.
        self.boundary_head = nn.Conv2d(c, 1, kernel_size=1)

        # Residual-adapter scales start at zero so v2 is exactly v1 at initialization,
        # then grow only when validation-supported gradients make the new branches useful.
        self.geom_scale = nn.Parameter(torch.tensor(0.0))
        self.dual_scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def prob_to_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    @staticmethod
    def prob_geometry(p: torch.Tensor):
        uncertainty = 1.0 - torch.abs(2.0 * p - 1.0)
        support = F.avg_pool2d(p, kernel_size=7, stride=1, padding=3)
        dx = F.pad(torch.abs(p[..., :, 1:] - p[..., :, :-1]), (0, 1, 0, 0))
        dy = F.pad(torch.abs(p[..., 1:, :] - p[..., :-1, :]), (0, 0, 0, 1))
        grad = (dx + dy).clamp(0.0, 1.0)
        return uncertainty, support, grad

    def forward(self, rgb: torch.Tensor, omni_prob: torch.Tensor):
        p = omni_prob.clamp(1e-4, 1.0 - 1e-4)
        omni_logit = self.prob_to_logit(p)
        uncertainty, support, grad = self.prob_geometry(p)

        x = self.stem(torch.cat([rgb, p], dim=1))
        geom = self.geom_stem(torch.cat([p, uncertainty, support, grad], dim=1))
        x = x + torch.tanh(self.geom_scale) * geom

        h = self.high(x)
        m = self.mid(F.avg_pool2d(h, 2, 2, ceil_mode=True))
        l = self.low(F.avg_pool2d(m, 2, 2, ceil_mode=True))
        m_up = F.interpolate(m, size=h.shape[-2:], mode="bilinear", align_corners=False)
        l_up = F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([h, m_up, l_up], dim=1))

        # v1 correction path retained.
        base_gate = torch.sigmoid(self.gate_head(feat)) * uncertainty
        base_residual = self.max_residual * torch.tanh(self.residual_head(feat))

        # Positive recovery: prefer ambiguous/low-probability pixels with neighborhood support.
        recover_gate = torch.sigmoid(self.recover_gate_head(feat)) * uncertainty * (1.0 - p) * (0.25 + 0.75 * support)
        recover = self.max_dual * torch.sigmoid(self.recover_head(feat))

        # Negative suppression: prefer predicted pixels with weak local support / strong boundary inconsistency.
        isolation = (1.0 - support).clamp(0.0, 1.0)
        suppress_gate = torch.sigmoid(self.suppress_gate_head(feat)) * p * (0.35 + 0.65 * isolation)
        suppress = self.max_dual * torch.sigmoid(self.suppress_head(feat))

        dual_raw = recover_gate * recover - suppress_gate * suppress
        dual_correction = torch.tanh(self.dual_scale) * dual_raw
        refined_logit = omni_logit + base_gate * base_residual + dual_correction
        refined_prob = torch.sigmoid(refined_logit)

        return {
            "refined_logit": refined_logit,
            "refined_prob": refined_prob,
            "residual": base_residual,
            "gate": base_gate,
            "uncertainty": uncertainty,
            "support": support,
            "prob_grad": grad,
            "recover_gate": recover_gate,
            "recover": recover,
            "suppress_gate": suppress_gate,
            "suppress": suppress,
            "dual_correction": dual_correction,
            "skeleton_logit": self.skeleton_head(feat),
            "boundary_logit": self.boundary_head(feat),
        }
