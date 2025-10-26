"""
Custom YOLO trainer with Focal Loss
Clean integration with Ultralytics' built-in FocalLoss

Supports:
- YOLOv8, YOLOv11, YOLOv12 (all sizes)
- RT-DETR (all sizes)
"""

import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from FocalLoss import FocalLoss


# ---- YOLO (v8/v11/v12) ----
class FLDetectionModel(DetectionModel):
    """
    Custom YOLO Detection Model with Focal Loss for classification.
    
    Replaces the standard BCE loss with Focal Loss to handle class imbalance.
    Gamma and alpha parameters can be configured via model args.
    """
    
    def init_criterion(self):
        """Initialize loss criterion with Focal Loss for classification."""
        # 1) Let the base build whatever it builds
        base = super().init_criterion()  # may return the loss, set self.criterion, or both

        # 2) Normalize to a local variable
        criterion = base if base is not None else getattr(self, "criterion", None)

        # 3) Fallback (older/edge cases) — construct explicitly if still missing
        if criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss  # name may be DetectionLoss in some versions
            criterion = v8DetectionLoss(self)

        # 4) Ensure the attribute exists (avoids your AttributeError)
        self.criterion = criterion

        # 5) Swap BCE -> FocalLoss (COCO-friendly defaults)
        gamma = float(getattr(self.args, "fl_gamma", 2.0))
        alpha = float(getattr(self.args, "fl_alpha", 0.25))

        if hasattr(self.criterion, "bce"):
            self.criterion.bce = FocalLoss(gamma=gamma, alpha=alpha)
        else:
            raise AttributeError(
                "The detection criterion doesn't expose `.bce`. "
                "Check your Ultralytics version and the loss class name/attrs."
            )

        print(f"[loss] YOLO cls -> FocalLoss(gamma={gamma}, alpha={alpha})")
        return self.criterion
    

class FocalLossReduced(nn.Module):
    """Wraps unreduced [B,N,C] focal into a scalar for RT-DETR."""

    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.base = FocalLoss(gamma, alpha)

    def forward(self, pred, label):
        out = self.base(pred, label)  # [B, N, C]

        return out.mean(1).sum()      # scalar


# ---- RT-DETR ----
class FLRTDETRModel(RTDETRDetectionModel):
    """
    Custom RT-DETR Detection Model with Focal Loss for classification.
    
    RT-DETR uses a different loss structure (DETR-style) but can also benefit
    from Focal Loss for handling class imbalance.
    """
    
    def init_criterion(self):
        gamma = float(getattr(self.args, "fl_gamma", 1.5))
        alpha = float(getattr(self.args, "fl_alpha", 0.25))

        # resolve nc: names -> head -> args
        nc = len(getattr(self, "names", []) or [])
        if not nc and hasattr(self, "model"):
            last = self.model[-1] if hasattr(self.model, "__getitem__") and len(self.model) else self.model
            nc = int(getattr(last, "nc", 0) or getattr(last, "num_classes", 0))
        if not nc:
            nc = int(getattr(self.args, "nc", 0))
        assert nc > 0, "Could not resolve number of classes (nc)."

        crit = RTDETRDetectionLoss(
            nc=nc,
            use_fl=True,
            use_vfl=False,
            gamma=gamma,
            alpha=alpha,
        )
        # swap in the reduced wrapper (important!)
        crit.fl = FocalLossReduced(gamma, alpha)

        self.criterion = crit
        print(f"[loss] RT-DETR -> FocalLoss(gamma={gamma}, alpha={alpha}, nc={nc})")
        return crit


class CustomDetectionTrainer(DetectionTrainer):
    """
    Custom Detection Trainer that uses Focal Loss models.
    
    Automatically detects whether the model is YOLO or RT-DETR and uses
    the appropriate Focal Loss implementation.
    """
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get model with Focal Loss based on configuration.
        
        Args:
            cfg: Model configuration file or dict
            weights: Path to model weights
            verbose: Whether to print verbose output
            
        Returns:
            Model with Focal Loss integrated
        """
        # Detect model type from config
        cfg_str = str(cfg).lower()
        is_rtdetr = ("rtdetr" in cfg_str) or ("rt-detr" in cfg_str)
        
        # Create appropriate model with Focal Loss
        if is_rtdetr:
            model = FLRTDETRModel(cfg=cfg, nc=self.data["nc"], verbose=verbose and self.args.verbose)
            print("[CustomTrainer] Using RT-DETR with Focal Loss")
        else:
            model = FLDetectionModel(cfg=cfg, nc=self.data["nc"], verbose=verbose and self.args.verbose)
            print("[CustomTrainer] Using YOLO with Focal Loss")
        
        # Load pretrained weights if provided
        if weights:
            model.load(weights)
            print(f"[CustomTrainer] Loaded weights from: {weights}")
        
        return model
