from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

from .config import LossConfig
from .model import no_amp

if TYPE_CHECKING:
    from .data import SplitPaths


class FullLoss(nn.Module):
    """Compact region, overlap, Lovasz, and explicit edge-head objective."""

    def __init__(self, config: LossConfig, pos_weight: float = 4.0):
        super().__init__()
        self.config = config
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def segmentation_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().clamp(-20.0, 20.0)
        targets = targets.float()
        weighted_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        unweighted_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability_correct = torch.exp(-unweighted_bce)
        focal = (
            (1.0 - probability_correct) ** self.config.focal_gamma * unweighted_bce
        ).mean()
        probabilities = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()
        lovasz = self.lovasz_hinge(logits, targets)
        return (
            self.config.bce * weighted_bce
            + self.config.focal * focal
            + self.config.dice * dice
            + self.config.lovasz * lovasz
        )

    def focal_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().clamp(-20.0, 20.0)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability_correct = torch.exp(-bce)
        focal = ((1.0 - probability_correct) ** self.config.focal_gamma * bce).mean()
        probabilities = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0e-6) / (union + 1.0e-6)
        return self.config.focal * focal + self.config.dice * dice.mean()

    @staticmethod
    def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with no_amp(logits):
            logits = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
            targets = targets.float()
            probabilities = torch.sigmoid(logits)
            target_dilated = F.max_pool2d(targets, 3, stride=1, padding=1)
            target_eroded = -F.max_pool2d(-targets, 3, stride=1, padding=1)
            prediction_dilated = F.max_pool2d(probabilities, 3, stride=1, padding=1)
            prediction_eroded = -F.max_pool2d(-probabilities, 3, stride=1, padding=1)
            predicted_edge = torch.nan_to_num(
                prediction_dilated - prediction_eroded, nan=0.0
            ).clamp(0.0, 1.0)
            target_edge = (target_dilated - target_eroded).clamp(0.0, 1.0)
            return F.binary_cross_entropy(predicted_edge, target_edge)

    @staticmethod
    def edge_head_loss(edge_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Supervise the actual edge head with BCE and soft Dice."""
        with no_amp(edge_logits):
            edge_logits = edge_logits.float().clamp(-20.0, 20.0)
            targets = targets.float()
            dilated = F.max_pool2d(targets, 3, stride=1, padding=1)
            eroded = -F.max_pool2d(-targets, 3, stride=1, padding=1)
            target_edge = (dilated - eroded).clamp(0.0, 1.0)
            positives = target_edge.sum()
            negatives = target_edge.numel() - positives
            edge_pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 10.0)
            bce = F.binary_cross_entropy_with_logits(
                edge_logits, target_edge, pos_weight=edge_pos_weight.reshape(1)
            )
            probability = torch.sigmoid(edge_logits)
            intersection = (probability * target_edge).sum(dim=(2, 3))
            denominator = probability.sum(dim=(2, 3)) + target_edge.sum(dim=(2, 3))
            dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            return 0.5 * (bce + dice)

    def hard_pixel_mining(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().clamp(-20.0, 20.0)
        targets = targets.float()
        error_map = torch.abs(torch.sigmoid(logits) - targets)
        hard_mask = (error_map > 0.5).float()
        per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return per_pixel.mean() + self.config.hard * (per_pixel * hard_mask).mean()

    def region_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.sigmoid(logits.float().clamp(-20.0, 20.0))
        targets = targets.float()
        intersection = (probabilities * targets).sum(dim=(2, 3))
        prediction_sum = probabilities.sum(dim=(2, 3))
        target_sum = targets.sum(dim=(2, 3))
        union = prediction_sum + target_sum - intersection
        iou_loss = 1.0 - ((intersection + 1.0) / (union + 1.0)).mean()
        false_positive = (probabilities * (1.0 - targets)).sum(dim=(2, 3))
        false_negative = ((1.0 - probabilities) * targets).sum(dim=(2, 3))
        denominator = (
            intersection
            + self.config.tversky_alpha * false_positive
            + self.config.tversky_beta * false_negative
        )
        tversky_loss = 1.0 - ((intersection + 1.0) / (denominator + 1.0)).mean()
        return iou_loss, tversky_loss

    @staticmethod
    def lovasz_hinge(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.float().reshape(-1)
        targets_flat = targets.float().reshape(-1)
        signs = 2.0 * targets_flat - 1.0
        errors = 1.0 - logits_flat * signs
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground = targets_flat[permutation]
        total_foreground = foreground.sum()
        intersection = total_foreground - foreground.cumsum(0)
        union = total_foreground + (1.0 - foreground).cumsum(0)
        gradient = 1.0 - intersection / union.clamp_min(1.0)
        if gradient.numel() > 1:
            gradient = torch.cat([gradient[:1], gradient[1:] - gradient[:-1]])
        return torch.dot(F.relu(errors_sorted), gradient.detach())

    def ohem_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_pixel = F.binary_cross_entropy_with_logits(
            logits.float().clamp(-20.0, 20.0), targets.float(), reduction="none"
        ).reshape(-1)
        count = max(1, int(per_pixel.numel() * self.config.ohem_fraction))
        return torch.topk(per_pixel, count, sorted=False).values.mean()

    def forward(
        self,
        refined_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        auxiliary_logits,
        edge_logits: torch.Tensor,
        diffusion_loss: torch.Tensor,
        orthogonal_loss: torch.Tensor,
        gram_loss: torch.Tensor,
        change_std: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        refined = self.segmentation_loss(refined_logits, targets)
        coarse = self.segmentation_loss(coarse_logits, targets)
        weights = [0.4, 0.3, 0.3]
        if isinstance(auxiliary_logits, (tuple, list)):
            auxiliary = sum(
                weight * self.segmentation_loss(logit, targets)
                for weight, logit in zip(weights, auxiliary_logits)
            )
        else:
            auxiliary = self.segmentation_loss(auxiliary_logits, targets)
        boundary = self.edge_head_loss(edge_logits, targets)
        return (
            refined
            + self.config.coarse * coarse
            + self.config.auxiliary * auxiliary
            + self.config.boundary * boundary
            + self.config.diffusion * diffusion_loss
            + self.config.orthogonal * orthogonal_loss
            + self.config.orthogonal_gram * gram_loss
            + self.config.variance * self.variance_loss(change_std)
        )

    def variance_loss(self, change_std: torch.Tensor) -> torch.Tensor:
        """BMD-CD Eq. 4 anti-collapse hinge: penalize change channels whose
        per-channel standard deviation falls below the margin tau."""
        shortfall = F.relu(self.config.variance_margin - change_std.float())
        return shortfall.pow(2).mean()


def estimate_pos_weight(paths: "SplitPaths", max_files: int = 200) -> float:
    import cv2

    step = max(1, len(paths) // max_files)
    positives = 0
    total = 0
    for path in paths.labels[::step]:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read label while estimating class balance: {path}")
        mask = mask > 127
        positives += int(mask.sum())
        total += int(mask.size)
    return float(np.clip(np.sqrt((total - positives) / (positives + 1.0e-6)), 1.0, 4.0))
