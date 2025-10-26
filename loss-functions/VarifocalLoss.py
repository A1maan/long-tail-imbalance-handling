"""
Varifocal Loss Implementation for Object Detection
Based on: https://arxiv.org/abs/2008.13367

VarifocalNet: An IoU-aware Dense Object Detector
Zhang et al. (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss by Zhang et al. (2020)
    
    Implements the Varifocal Loss function for addressing class imbalance in object detection
    by asymmetrically weighting positive and negative examples. Unlike Focal Loss which treats
    positive and negative samples symmetrically, Varifocal Loss:
    - Weights positive examples by the IoU-aware classification score (gt_score)
    - Weights negative examples by the predicted score raised to gamma power
    
    This is particularly effective for dense object detectors where classification scores
    should be IoU-aware (IACS - IoU-Aware Classification Score).
    
    Key Differences from Focal Loss:
    1. Asymmetric weighting: Different strategies for positive vs negative samples
    2. IoU-aware targets: Positive samples use IoU as regression target
    3. Quality-focused: Encourages high-quality detections (high IoU)
    4. Reduces noise: Down-weights low-quality positives automatically
    
    Attributes:
        gamma (float): The focusing parameter for negative examples (default: 2.0).
                      Controls how much to down-weight easy negatives.
        alpha (float): The balancing factor for positive examples (default: 0.75).
                      Balances the contribution of positive vs negative samples.
    
    References:
        VarifocalNet: An IoU-aware Dense Object Detector
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """
        Initialize the VarifocalLoss class.
        
        Args:
            gamma: Focusing parameter for hard negatives (typically 2.0)
                  Higher values increase focus on hard negatives
            alpha: Balancing factor for positive samples (typically 0.75)
                  Controls the weight given to positive samples
        """
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor, gt_score: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute Varifocal Loss.
        
        Args:
            pred: Predicted logits of shape [B, A, C] where
                  B = batch size, A = anchors, C = classes
            label: Binary ground truth labels of shape [B, A, C]
                   1 for positive samples, 0 for negatives
            gt_score: IoU-aware target scores of shape [B, A, C] (optional).
                     For positive samples, this should be the IoU between prediction and GT.
                     If None, uses label as gt_score (equivalent to standard training).
        
        Returns:
            Varifocal loss tensor (unreduced, same shape as input)
            
        Example:
            >>> criterion = VarifocalLoss(gamma=2.0, alpha=0.75)
            >>> pred = torch.randn(2, 100, 80)  # 2 images, 100 anchors, 80 classes
            >>> label = torch.zeros(2, 100, 80)
            >>> label[0, 5, 10] = 1.0  # positive sample
            >>> gt_iou = torch.zeros_like(label)
            >>> gt_iou[0, 5, 10] = 0.85  # IoU score for the positive
            >>> loss = criterion(pred, label, gt_iou)
            >>> total_loss = loss.mean()  # or loss.sum() depending on reduction strategy
        """
        # If gt_score is not provided, use label (for compatibility)
        if gt_score is None:
            gt_score = label
        
        # Compute predicted probabilities
        prob = pred.sigmoid()
        
        # Varifocal weighting:
        # - For positive samples (label=1): weight by gt_score (IoU)
        #   Higher IoU → higher weight → model learns to focus on quality
        # - For negative samples (label=0): weight by alpha * prob^gamma
        #   Higher confidence on negatives → higher weight → focus on hard negatives
        weight = (
            gt_score * label +                                    # positive: IoU-aware weight
            self.alpha * prob.pow(self.gamma) * (1.0 - label)    # negative: focal-like weight
        )
        
        # Varifocal targets:
        # - For positive samples: use gt_score (IoU value)
        #   Model learns to predict IoU as classification score
        # - For negative samples: use 0
        target = gt_score * label
        
        # Compute binary cross entropy with the varifocal targets
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        # Apply varifocal weighting
        loss = weight * bce
        
        return loss
