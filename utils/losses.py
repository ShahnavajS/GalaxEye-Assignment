from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        targets = targets.float().reshape(targets.shape[0], -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_factor * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


class CombinedBinaryLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = BinaryDiceLoss()
        self.focal_loss = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        if pos_weight is None:
            self.register_buffer("pos_weight", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight if self.pos_weight.numel() > 0 else None
        loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

        if self.bce_weight > 0.0:
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            loss = loss + self.bce_weight * bce

        if self.dice_weight > 0.0:
            dice = self.dice_loss(logits, targets)
            loss = loss + self.dice_weight * dice

        if self.focal_weight > 0.0:
            focal = self.focal_loss(logits, targets)
            loss = loss + self.focal_weight * focal

        return loss


def build_loss(
    name: str = "bce_dice",
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_weight: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    pos_weight: float | None = None,
) -> nn.Module:
    presets = {
        "bce": {"bce_weight": 1.0, "dice_weight": 0.0, "focal_weight": 0.0},
        "dice": {"bce_weight": 0.0, "dice_weight": 1.0, "focal_weight": 0.0},
        "focal": {"bce_weight": 0.0, "dice_weight": 0.0, "focal_weight": 1.0},
        "bce_dice": {"bce_weight": bce_weight, "dice_weight": dice_weight, "focal_weight": 0.0},
        "bce_focal": {"bce_weight": bce_weight, "dice_weight": 0.0, "focal_weight": focal_weight or 1.0},
        "dice_focal": {"bce_weight": 0.0, "dice_weight": dice_weight, "focal_weight": focal_weight or 1.0},
        "bce_dice_focal": {
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            "focal_weight": focal_weight or 1.0,
        },
    }

    normalized_name = name.lower()
    if normalized_name not in presets:
        raise ValueError(f"Unsupported loss name: {name}")

    weights = presets[normalized_name]
    return CombinedBinaryLoss(
        bce_weight=float(weights["bce_weight"]),
        dice_weight=float(weights["dice_weight"]),
        focal_weight=float(weights["focal_weight"]),
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        pos_weight=pos_weight,
    )
