"""
Custom YOLO trainer with Focal Loss
Optimized for multi-GPU training
"""

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils.loss import FocalLoss


class CustomDetectionModel(DetectionModel):
    """
    Custom YOLO Detection Model with Focal Loss and Gradient Attribution
    Overrides the standard loss function to compute gradients w.r.t. input images
    for attribution analysis while using Focal Loss for imbalance handling
    
    Compatible with:
    - YOLOv8 (all sizes: n, s, m, l, x)
    - YOLOv11 (all sizes)
    - YOLOv12 (all sizes)
    - RT-DETR (all sizes: l, x)
    """
    
    def __init__(self, cfg="yolov8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize custom detection model
        
        Args:
            cfg: Model config file
            ch: Input channels
            nc: Number of classes
            verbose: Verbosity flag
        """
        super().__init__(cfg, ch, nc, verbose)
        # Use Ultralytics' built-in FocalLoss
        self.focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    
    def init_criterion(self):
        """
        Initialize the loss function with Focal Loss for classification
        Replaces the BCE component with FocalLoss in v8DetectionLoss
        """
        # Get the standard YOLO criterion (v8DetectionLoss)
        criterion = super().init_criterion()
        
        # Replace the BCE component with Focal Loss
        # v8DetectionLoss.bce is used for classification loss
        if hasattr(criterion, 'bce'):
            original_bce = type(criterion.bce).__name__
            # Create a NEW FocalLoss instance to ensure it's properly initialized
            criterion.bce = FocalLoss(gamma=2.0, alpha=0.25)
            print(f"[CustomDetectionModel] ✓ Replaced {original_bce} with FocalLoss")
            print(f"  FocalLoss params: gamma={criterion.bce.gamma}, alpha={criterion.bce.alpha}")
        else:
            print(f"[CustomDetectionModel] ⚠ Warning: criterion has no 'bce' attribute")
            print(f"  Available attributes: {dir(criterion)}")
        
        self.criterion = criterion
        print(f"[CustomDetectionModel] Initialized criterion: {type(criterion).__name__}")
        if hasattr(criterion, 'bce'):
            bce_type = type(criterion.bce).__name__
            print(f"  - Classification Loss: {bce_type}")
            if hasattr(criterion.bce, 'gamma') and hasattr(criterion.bce, 'alpha'):
                print(f"    → gamma={criterion.bce.gamma}, alpha={criterion.bce.alpha}")
        
        return self.criterion
    
    def loss(self, batch, preds=None):
        """
        Compute loss with gradient attribution
        
        Args:
            batch: Input batch containing images and labels
            preds: Predictions (optional, will forward if None)
            
        Returns:
            loss: Computed loss value
        """
        # Initialize criterion if not already done
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        
        # Create a copy with gradient tracking enabled (non-in-place operation)
        # This prevents modifying the original batch['img'] tensor
        imgs = batch['img'].clone().detach().requires_grad_(True)
        
        # Forward pass with gradient-enabled images
        preds = self.forward(imgs) if preds is None else preds
        
        # Compute standard loss using the original batch (not the cloned images)
        # This ensures the rest of the pipeline works correctly
        loss = self.criterion(preds, batch)
        
        try:
            # Get prediction scores for gradient computation
            pred_scores = self.get_pred_scores(preds)
            
            # Compute gradients w.r.t input images for attribution
            # Using autograd to enable testing w.r.t different loss components
            gradients = torch.autograd.grad(
                outputs=pred_scores.sum(),  # Sum to get scalar for backprop
                inputs=imgs,
                grad_outputs=None,  # Not needed when outputs is scalar
                retain_graph=True,
                create_graph=False  # Set True if you need higher order gradients
            )[0]
            
            # Store gradients for attribution analysis (optional)
            # You can process these gradients here or store them for later analysis
            # self.last_gradients = gradients.detach()
            
        except Exception as e:
            # If gradient computation fails, log warning but continue training
            # This ensures training doesn't crash for unsupported architectures
            if not hasattr(self, '_gradient_warning_shown'):
                print(f"\n⚠️  Warning: Gradient attribution failed: {e}")
                print("   Training will continue without gradient-based attribution.")
                print("   This might happen with certain model architectures.\n")
                self._gradient_warning_shown = True
        
        return loss
    
    def get_pred_scores(self, preds):
        """
        Extract prediction scores from model predictions
        Compatible with YOLO v8/v11/v12 and RT-DETR architectures
        
        Args:
            preds: Model predictions (can be tuple or tensor)
            
        Returns:
            pred_scores: Prediction confidence scores [batch, num_anchors, num_classes]
        """
        # Handle tuple output (training mode) vs single tensor (inference mode)
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Get detection head
        detect_head = self.model[-1]
        
        # Check model type by detection head class name
        head_type = detect_head.__class__.__name__
        
        if 'RTDETRDecoder' in head_type:
            # RT-DETR uses Transformer decoder with different output format
            # RT-DETR outputs: (pred_bboxes, pred_scores) directly
            # pred_scores shape: [batch, num_queries, num_classes]
            if isinstance(preds, tuple) and len(preds) >= 2:
                # Training mode: preds = (loss, (pred_bboxes, pred_scores))
                pred_bboxes, pred_scores = preds[1] if len(preds) > 1 else preds
            else:
                # Inference mode or direct output
                # For RT-DETR, we need to handle the output differently
                # Typically it returns dict with 'pred_logits' and 'pred_boxes'
                if isinstance(feats, dict):
                    pred_scores = feats.get('pred_logits', feats.get('scores'))
                elif isinstance(feats, (list, tuple)) and len(feats) >= 2:
                    pred_bboxes, pred_scores = feats
                else:
                    # Fallback: treat as scores directly
                    pred_scores = feats
            
            # Ensure proper shape [batch, num_queries, num_classes]
            if pred_scores.dim() == 2:
                pred_scores = pred_scores.unsqueeze(0)
                
        else:
            # Standard YOLO v8/v11/v12 anchor-free detection
            # Concatenate features from all detection scales
            # Each feature map is reshaped to [batch, no, -1] where no = num_outputs per anchor
            try:
                concat_feats = torch.cat(
                    [xi.view(feats[0].shape[0], detect_head.no, -1) for xi in feats], 
                    dim=2
                )
                
                # Split into bbox coordinates and class scores
                # reg_max * 4 = bbox regression parameters
                # nc = number of classes
                pred_bboxes, pred_scores = concat_feats.split(
                    (detect_head.reg_max * 4, detect_head.nc), 
                    dim=1
                )
                
                # Permute to [batch, num_anchors, num_classes]
                pred_scores = pred_scores.permute(0, 2, 1).contiguous()
                
            except (AttributeError, RuntimeError) as e:
                # Fallback for any unexpected architecture
                print(f"Warning: Using fallback score extraction for {head_type}: {e}")
                # Try to extract scores directly
                if isinstance(feats, (list, tuple)):
                    # Take the last feature map and try to extract class predictions
                    last_feat = feats[-1]
                    if last_feat.dim() >= 3:
                        pred_scores = last_feat
                    else:
                        pred_scores = last_feat.unsqueeze(1)
                else:
                    pred_scores = feats
        
        return pred_scores


class CustomRTDETRModel(RTDETRDetectionModel):
    """
    Custom RT-DETR Detection Model with Focal Loss and Gradient Attribution
    Extends RTDETRDetectionModel specifically for RT-DETR architectures
    
    Compatible with:
    - RT-DETR (all sizes: l, x)
    """
    
    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize custom RT-DETR detection model
        
        Args:
            cfg: Model config file
            ch: Input channels
            nc: Number of classes
            verbose: Verbosity flag
        """
        super().__init__(cfg, ch, nc, verbose)
        # Use Ultralytics' built-in FocalLoss
        self.focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    
    def init_criterion(self):
        """
        Initialize the loss function for RT-DETR
        RT-DETR uses DETRLoss which can have either FocalLoss (fl) or VarifocalLoss (vfl)
        We replace the varifocal/focal loss component with our FocalLoss
        """
        # Get the standard RT-DETR criterion (RTDETRDetectionLoss)
        criterion = super().init_criterion()
        
        print(f"[CustomRTDETRModel] Initialized criterion: {type(criterion).__name__}")
        
        # RT-DETR's DETRLoss uses either focal loss (fl) or varifocal loss (vfl)
        # We want to use FocalLoss instead of the default VarifocalLoss
        if hasattr(criterion, 'vfl') and criterion.vfl is not None:
            original_vfl = type(criterion.vfl).__name__
            print(f"[CustomRTDETRModel] Found {original_vfl} (Varifocal Loss)")
            print(f"[CustomRTDETRModel] Replacing VFL with FocalLoss...")
            # Replace VFL with our FocalLoss - create NEW instance for DDP safety
            criterion.vfl = None  # Disable VFL
            criterion.fl = FocalLoss(gamma=2.0, alpha=0.25)  # Enable our FocalLoss
            print(f"[CustomRTDETRModel] ✓ Replaced {original_vfl} with FocalLoss")
            print(f"  FocalLoss params: gamma={criterion.fl.gamma}, alpha={criterion.fl.alpha}")
        elif hasattr(criterion, 'fl') and criterion.fl is not None:
            original_fl = type(criterion.fl).__name__
            criterion.fl = FocalLoss(gamma=2.0, alpha=0.25)
            print(f"[CustomRTDETRModel] ✓ Replaced {original_fl} with FocalLoss")
            print(f"  FocalLoss params: gamma={criterion.fl.gamma}, alpha={criterion.fl.alpha}")
        else:
            print(f"[CustomRTDETRModel] ⚠ No focal loss component found in criterion")
            print(f"  Available attributes: {dir(criterion)}")
        
        self.criterion = criterion
        
        return self.criterion
    
    def loss(self, batch, preds=None):
        """
        Compute loss with gradient attribution for RT-DETR
        
        Args:
            batch: Input batch containing images and labels
            preds: Predictions (optional, will forward if None)
            
        Returns:
            loss: Computed loss value
        """
        # Initialize criterion if not already done
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        
        # Create a copy with gradient tracking enabled (non-in-place operation)
        imgs = batch['img'].clone().detach().requires_grad_(True)
        
        # Forward pass with gradient-enabled images
        preds = self.forward(imgs) if preds is None else preds
        
        # RT-DETR loss function expects specific format
        # Standard forward returns: (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        # We need to process these similar to the original RTDETRDetectionModel
        
        if preds is None:
            preds = self.forward(imgs)
        
        # Unpack predictions similar to standard RT-DETR model
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        
        # Handle denoising split if dn_meta exists
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        
        # Concatenate encoder and decoder predictions
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        
        # Prepare targets in the format expected by RTDETRDetectionLoss
        bs = imgs.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(imgs.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=imgs.device),
            "batch_idx": batch_idx.to(imgs.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }
        
        # Compute loss using criterion
        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        
        # Return in the format expected by trainer: (loss_sum, loss_items)
        # RT-DETR has ~12 losses but we only show the main three
        loss_sum = sum(loss.values())
        loss_items = torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=imgs.device
        )
        
        try:
            # Compute gradients w.r.t input images for attribution
            # Use dec_scores for gradient computation
            if dec_scores is not None and hasattr(dec_scores, 'sum'):
                gradients = torch.autograd.grad(
                    outputs=dec_scores.sum(),
                    inputs=imgs,
                    grad_outputs=None,
                    retain_graph=True,
                    create_graph=False
                )[0]
                # self.last_gradients = gradients.detach()
            
        except Exception as e:
            if not hasattr(self, '_gradient_warning_shown'):
                print(f"\n⚠️  Warning: RT-DETR gradient attribution failed: {e}")
                print("   Training will continue without gradient-based attribution.\n")
                self._gradient_warning_shown = True
        
        return loss_sum, loss_items


class CustomDetectionTrainer(DetectionTrainer):
    """
    Custom Detection Trainer that uses Focal Loss
    Extends DetectionTrainer to implement custom loss function
    Automatically selects the correct model class based on architecture
    """
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Returns a customized detection model with Focal Loss
        Automatically detects if model is RT-DETR and uses appropriate class
        
        Args:
            cfg: Model config (may be None)
            weights: Pre-trained weights
            verbose: Verbosity flag
            
        Returns:
            CustomDetectionModel or CustomRTDETRModel instance
        """
        # CRITICAL FIX: cfg can be None, use self.args.model as fallback
        if cfg is None:
            cfg = self.args.model
        
        print(f"[CustomTrainer] get_model called with cfg={cfg}, weights={weights}")
        
        # Detect if this is an RT-DETR model
        cfg_str = str(cfg).lower()
        is_rtdetr = 'rtdetr' in cfg_str or 'rt-detr' in cfg_str
        
        if is_rtdetr:
            print(f"[CustomTrainer] Detected RT-DETR model, using CustomRTDETRModel")
            model = CustomRTDETRModel(cfg=cfg, nc=self.data["nc"], verbose=verbose and self.args.verbose)
        else:
            print(f"[CustomTrainer] Detected YOLO model, using CustomDetectionModel")
            model = CustomDetectionModel(cfg=cfg, nc=self.data["nc"], verbose=verbose and self.args.verbose)
        
        if weights:
            model.load(weights)
        return model