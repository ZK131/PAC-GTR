"""The RGB TopoCrackNet architecture and its training objective.

The model has one shared U-Net backbone and three task heads: crack region,
centreline and boundary.  At inference, only the region probability is passed
to the multi-view semantic-mapping stage.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize
from torch import nn


class ConvBlock(nn.Module):
    """Two convolutional layers with group normalization and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TopoCrackNet(nn.Module):
    """Lightweight multi-head U-Net for topology-aware crack segmentation."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        channels = [24, 48, 96, 192]
        self.enc1 = ConvBlock(in_channels, channels[0])
        self.down1 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)
        self.enc2 = ConvBlock(channels[1], channels[1])
        self.down2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1)
        self.enc3 = ConvBlock(channels[2], channels[2])
        self.down3 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1)
        self.enc4 = ConvBlock(channels[3], channels[3])

        self.up3 = ConvBlock(channels[3] + channels[2], channels[2])
        self.up2 = ConvBlock(channels[2] + channels[1], channels[1])
        self.up1 = ConvBlock(channels[1] + channels[0], channels[0])
        self.region_head = nn.Conv2d(channels[0], 1, 1)
        self.center_head = nn.Conv2d(channels[0], 1, 1)
        self.boundary_head = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _resize(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        d3 = self.up3(torch.cat([self._resize(e4, e3), e3], dim=1))
        d2 = self.up2(torch.cat([self._resize(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([self._resize(d2, e1), e1], dim=1))
        return self.region_head(d1), self.center_head(d1), self.boundary_head(d1)

    @torch.no_grad()
    def predict_region(self, images: torch.Tensor) -> torch.Tensor:
        """Return the crack-region probability map used at inference."""
        return torch.sigmoid(self(images)[0])


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    vertical = -nn.functional.max_pool2d(-x, (3, 1), 1, (1, 0))
    horizontal = -nn.functional.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_skeletonize(x: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    skeleton = torch.zeros_like(x)
    for _ in range(iterations):
        opened = nn.functional.max_pool2d(_soft_erode(x), 3, 1, 1)
        delta = (x - opened).clamp_min(0.0)
        skeleton = skeleton + (delta - skeleton * delta).clamp_min(0.0)
        x = _soft_erode(x)
    return skeleton


def soft_cldice_loss(region_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable centreline-overlap loss applied to the region branch."""
    probability = torch.sigmoid(region_logits).clamp(1e-4, 1.0 - 1e-4)
    prediction_skeleton = _soft_skeletonize(probability)
    reference_skeleton = torch.from_numpy(
        np.stack([skeletonize(image[0].detach().cpu().numpy() > 0.5) for image in target])
    ).to(target.device, dtype=torch.float32)[:, None]
    topology_precision = (prediction_skeleton * target).sum(dim=(1, 2, 3)) / (
        prediction_skeleton.sum(dim=(1, 2, 3)) + 1e-6
    )
    topology_recall = (reference_skeleton * probability).sum(dim=(1, 2, 3)) / (
        reference_skeleton.sum(dim=(1, 2, 3)) + 1e-6
    )
    return (1.0 - (2.0 * topology_precision * topology_recall + 1e-6) /
            (topology_precision + topology_recall + 1e-6)).mean()


def loss_fn(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    region_target: torch.Tensor,
    center_target: torch.Tensor,
    boundary_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Frozen objective: region + .25 centreline + .15 boundary + .15 clDice."""
    region_logits, center_logits, boundary_logits = outputs
    region_loss = 0.65 * nn.functional.binary_cross_entropy_with_logits(region_logits, region_target)
    region_loss += 0.35 * dice_loss(region_logits, region_target)
    center_loss = 0.50 * nn.functional.binary_cross_entropy_with_logits(center_logits, center_target)
    center_loss += 0.50 * dice_loss(center_logits, center_target)
    boundary_loss = nn.functional.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
    topology_loss = soft_cldice_loss(region_logits, region_target)
    total = region_loss + 0.25 * center_loss + 0.15 * boundary_loss + 0.15 * topology_loss
    details = {
        "region": float(region_loss.detach()),
        "center": float(center_loss.detach()),
        "boundary": float(boundary_loss.detach()),
        "cldice": float(topology_loss.detach()),
    }
    return total, details
