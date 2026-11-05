from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    skel_pred = soft_skeletonize(prob, iters)
    skel_true = soft_skeletonize(target, iters)

    dims = tuple(range(1, prob.ndim))
    tprec = ((skel_pred * target).sum(dim=dims) + eps) / (skel_pred.sum(dim=dims) + eps)
    tsens = ((skel_true * prob).sum(dim=dims) + eps) / (skel_true.sum(dim=dims) + eps)
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1.0 - cldice.mean()


def omni_tcr_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    bce_w: float = 1.0,
    dice_w: float = 1.0,
    cldice_w: float = 0.5,
    skel_w: float = 0.2,
    residual_l1_w: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["refined_logit"]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    cldice = soft_cldice_loss(logits, target)

    with torch.no_grad():
        target_skel = soft_skeletonize(target, iters=10).clamp(0, 1)
    skel = F.binary_cross_entropy_with_logits(outputs["skeleton_logit"], target_skel)
    residual_l1 = outputs["residual"].abs().mean()

    total = bce_w*bce + dice_w*dice + cldice_w*cldice + skel_w*skel + residual_l1_w*residual_l1
    return total, {
        "loss_total": total.detach(),
        "loss_bce": bce.detach(),
        "loss_dice": dice.detach(),
        "loss_cldice": cldice.detach(),
        "loss_skeleton": skel.detach(),
        "loss_residual_l1": residual_l1.detach(),
    }
