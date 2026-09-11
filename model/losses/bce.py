"""Loss functions for SignalScope authenticity classifiers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with Logits and label smoothing.

    Prevents overconfidence on synthetic signatures that may shift
    across generator families, improving unseen generator generalization.
    """

    def __init__(self, label_smoothing: float = 0.05, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Applies label smoothing and calculates BCE loss."""
        # Smooth targets: 0 -> epsilon, 1 -> 1 - epsilon
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=pos_weight,
        )


class BinaryFocalLoss(nn.Module):
    """Focal Loss for focusing training on hard/borderline authenticity cases."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
