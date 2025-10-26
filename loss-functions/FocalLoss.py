"""
Focal Loss and Varifocal Loss Implementations for Object Detection
- Focal Loss: https://arxiv.org/abs/1708.02002
- Varifocal Loss: https://arxiv.org/abs/2008.13367
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float | None = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        # store alpha as float or None; don't mutate module attr inside forward
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # BCE term, unreduced
        bce = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # focal modulating factor
        prob = pred.sigmoid()
        p_t = label * prob + (1.0 - label) * (1.0 - prob)          # [B, A, C]
        modulating = (1.0 - p_t).pow(self.gamma)                    # [B, A, C]
        loss = bce * modulating

        # alpha balancing (optional)
        if self.alpha is not None:
            # broadcast scalar alpha to tensor on the right device/dtype
            alpha_t = label.new_tensor(self.alpha)
            alpha_t = label * alpha_t + (1.0 - label) * (1.0 - alpha_t)
            loss = alpha_t * loss                                    # [B, A, C]

        return loss