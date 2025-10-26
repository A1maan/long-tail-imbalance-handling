"""
Training script for TorchVision object detection models with Focal Loss
Supports Faster R-CNN (with custom Focal Loss) and RetinaNet (built-in Focal Loss)

Implementation based on:
- Focal Loss paper: https://arxiv.org/abs/1708.02002
- TorchVision loss modification: https://github.com/pytorch/vision/issues/1882

DDP Training:
- Uses torch.multiprocessing.spawn() by default (set USE_DDP=True in main())
- Alternative: Use torchrun for cleaner process management:
  CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_torchvision.py
  (Then modify main() to use dist.init_process_group(backend="nccl", init_method="env://"))
"""

from typing import List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
from torch.amp import autocast, GradScaler

class CocoDetectionDataset(Dataset):
    """Minimal COCO detection dataset that returns image tensors and detection targets."""

    def __init__(self, image_dir: Path, ann_file: Path, image_transform=None):
        self.coco = COCO(str(ann_file))
        
        self.cat_ids = sorted(self.coco.getCatIds())          
        self.catid2contig = {c: i + 1 for i, c in enumerate(self.cat_ids)}  
        self.ids = sorted(self.coco.getImgIds())
        
        self.image_dir = Path(image_dir)
        self.image_transform = image_transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]

        # Load image
        image_info = coco.loadImgs(img_id)[0]
        image_path = self.image_dir / image_info["file_name"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image {image_path}: {e}")
            # Return next valid image
            return self.__getitem__((index + 1) % len(self.ids))

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annotations:
            x_min, y_min, width, height = ann["bbox"]
            if width <= 0 or height <= 0:
                continue
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", width * height)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        image_tensor = self.image_transform(image)
        return image_tensor, target


def collate_fn(batch):
    """Custom collate_fn to handle variable-size targets."""
    return tuple(zip(*batch))


def patch_faster_rcnn_focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> None:
    """
    Monkey-patch torchvision's Faster R-CNN classification loss to use Focal Loss.
    
    Based on the approach from: https://github.com/pytorch/vision/issues/1882
    This replaces the fastrcnn_loss function in roi_heads module before model creation.
    
    Args:
        alpha: Focal loss alpha parameter (weighting factor)
        gamma: Focal loss gamma parameter (focusing parameter)
    """
    import torchvision.models.detection.roi_heads as roi_heads_module

    def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        """
        Custom Faster R-CNN loss with Focal Loss for classification.
        
        Args:
            class_logits: Predicted class scores (N, num_classes)
            box_regression: Predicted box deltas (N, num_classes * 4)
            labels: Ground truth labels (N,)
            regression_targets: Ground truth box deltas (N, 4)
        """
        # Classification loss with Focal Loss
        # Compute cross entropy per sample
        if isinstance(labels, list):
            labels = torch.cat(labels, dim=0)
    
        # print("Regression targets type:", type(regression_targets))
        # print("Regression targets:", regression_targets)
        
        if isinstance(regression_targets, (list, tuple)):
            regression_targets = torch.cat(regression_targets, dim=0)
            
        ce_loss = F.cross_entropy(class_logits, labels, reduction='none')
        
        # Get probabilities for focal loss calculation
        p = torch.exp(-ce_loss)
        
        # Compute focal loss: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
        focal_loss = alpha * (1 - p) ** gamma * ce_loss
        classification_loss = focal_loss.mean()

        # Bounding box regression loss (unchanged from original implementation)
        # Only compute box loss for positive samples (labels > 0, background is 0)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        
        if sampled_pos_inds_subset.numel() == 0:
            # No positive samples, return zero box loss
            box_loss = torch.tensor(0.0, device=box_regression.device, dtype=box_regression.dtype)
        else:
            # Reshape box regression to (N, num_classes, 4)
            N = class_logits.shape[0]
            box_regression = box_regression.reshape(N, -1, 4)
            
            # Select box predictions for positive samples and their corresponding classes
            # Ensure indices are proper type (int64/long)
            sampled_pos_inds_subset = sampled_pos_inds_subset.long()
            labels_pos = labels_pos.long()
            
            box_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                beta=1.0 / 9.0,
                reduction="sum",
            )
            box_loss = box_loss / max(1, labels.numel())

        return classification_loss, box_loss

    # Patch the module-level function (this is the key insight from the GitHub issue)
    roi_heads_module.fastrcnn_loss = custom_fastrcnn_loss
    print(f"[INFO] Patched Faster R-CNN with Focal Loss (alpha={alpha}, gamma={gamma})")


class TorchVisionDetectionTrainer:
    """Trainer for TorchVision detection models using Focal Loss with DDP support."""

    def __init__(
        self, 
        model_name: str = "faster_rcnn", 
        device_ids: List[int] = None, 
        num_classes: int = 91, 
        model_dir: str = "/home/almaankhan/model",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        batch_size: int = 8,  # For info display only
        use_ddp: bool = True  # Use DDP instead of DataParallel
    ):
        """
        Initialize TorchVision detection model trainer with Focal Loss.
        
        Args:
            model_name: Model architecture ('faster_rcnn' or 'retinanet')
            device_ids: List of GPU device IDs to use
            num_classes: Number of classes (including background)
            model_dir: Directory containing pretrained model weights
            focal_alpha: Focal loss alpha parameter (default: 0.25)
            focal_gamma: Focal loss gamma parameter (default: 2.0)
            batch_size: Batch size per GPU (for display only)
            use_ddp: Use DistributedDataParallel (recommended) vs DataParallel
        """
        device_ids = device_ids or [0]
        self.device_ids = device_ids
        self.use_ddp = use_ddp and len(device_ids) > 1
        self.model_name = model_name
        self.model_dir = model_dir
        self.batch_size = batch_size
        
        # DDP setup
        if self.use_ddp:
            # Will be set when setup_ddp() is called
            self.rank = None
            self.local_rank = None
            self.world_size = len(device_ids)
            self.device = None
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.primary_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
            self.device = self.primary_device
        
        # Store model creation parameters for later
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_classes = num_classes
        self.model = None  # Will be created in setup_model()
        
    def setup_ddp(self, rank: int, world_size: int):
        """Setup DDP process group."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        self.rank = rank
        self.local_rank = rank
        
        # Map rank to actual CUDA device (handles non-sequential GPU IDs)
        cuda_idx = self.device_ids[rank]
        self.device = torch.device(f"cuda:{cuda_idx}")
        torch.cuda.set_device(cuda_idx)
        
    def setup_model(self):
        """Create and setup the model (called after DDP init if using DDP)."""
        if self.model_name == "faster_rcnn":
            # Patch Faster R-CNN to use Focal Loss BEFORE creating the model
            patch_faster_rcnn_focal_loss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            
            # Now create the model - it will use our patched focal loss
            base_model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
            
            # Load fine-tuned weights if available
            weights_path = Path(self.model_dir) / "faster_rcnn_resnet50_fpn_v2.pt"
            if weights_path.exists():
                if self.rank == 0:  # Only print on main process
                    print(f"[INFO] Loading Faster R-CNN weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device)
                base_model.load_state_dict(state_dict, strict=False)
                
        elif self.model_name == "retinanet":
            # RetinaNet already uses Focal Loss internally by design
            if self.rank == 0:
                print(f"[INFO] RetinaNet already uses Focal Loss internally (alpha={self.focal_alpha}, gamma={self.focal_gamma})")
            base_model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
            
            # Load fine-tuned weights if available
            weights_path = Path(self.model_dir) / "retinanet_resnet50_fpn_v2.pt"
            if weights_path.exists():
                if self.rank == 0:
                    print(f"[INFO] Loading RetinaNet weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device)
                base_model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Move model to device
        base_model.to(self.device)
        
        # Wrap with DDP or DataParallel
        if self.use_ddp:
            # Get actual CUDA device index for DDP
            cuda_idx = self.device.index if hasattr(self.device, 'index') else self.device_ids[self.local_rank]
            self.model = DDP(
                base_model, 
                device_ids=[cuda_idx], 
                output_device=cuda_idx,
                find_unused_parameters=False  # Typical for torchvision detection models
            )
            if self.rank == 0:
                print(f"[INFO] Using DistributedDataParallel on {self.world_size} GPUs: {self.device_ids}")
                print(f"[INFO] Effective batch size: {self.batch_size} per GPU × {self.world_size} GPUs = {self.batch_size * self.world_size} total")
        elif len(self.device_ids) > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(base_model, device_ids=self.device_ids)
            print(f"[INFO] Using DataParallel on GPUs: {self.device_ids}")
            print(f"[INFO] Effective batch size: {self.batch_size} per GPU × {len(self.device_ids)} GPUs = {self.batch_size * len(self.device_ids)} total")
        else:
            self.model = base_model
            print(f"[INFO] Using single GPU: {self.device_ids[0]}")

    def _ddp_reduce_loss(self, loss_value: float) -> float:
        """Reduce loss across all DDP processes to get global average."""
        if not self.use_ddp or not dist.is_initialized():
            return loss_value
        
        loss_tensor = torch.tensor(loss_value, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / dist.get_world_size()).item()

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.005,
        weight_decay: float = 0.0005,
        save_dir: Path = Path("runs/torchvision"),
        patience: int = 10,
    ) -> None:
        """Train the model with optional DDP support."""
        
        # Create gradient scaler for mixed precision
        scaler = GradScaler('cuda')
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        save_dir = save_dir / self.model_name
        if self.rank == 0:  # Only create dir on main process
            save_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        patience_counter = 0
        improvement_threshold = 0.01  # Minimum loss improvement to reset patience

        for epoch in range(epochs):
            if self.use_ddp:
                # Set epoch for DistributedSampler to shuffle differently each epoch
                train_loader.sampler.set_epoch(epoch)
            
            self.model.train()
            running_loss = 0.0

            # Only show progress bar on main process
            if self.rank == 0:
                pbar = tqdm(train_loader, desc=f"{self.model_name} Epoch {epoch+1}/{epochs}")
            else:
                pbar = train_loader
                
            for images, targets in pbar:
                # Move data to device with non_blocking for faster transfer
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [
                    {k: v.to(self.device, non_blocking=True) for k, v in target.items()}
                    for target in targets
                ]
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than setting to 0
                
                with autocast(device_type='cuda', enabled=True):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += losses.item()
                if self.rank == 0:  # Only update progress bar on main process
                    pbar.set_postfix({"loss": losses.item()})

            lr_scheduler.step()
            
            # Calculate epoch loss and reduce across all ranks for DDP
            epoch_loss_local = running_loss / max(len(train_loader), 1)
            avg_loss = self._ddp_reduce_loss(epoch_loss_local)
            
            # --- Rank 0: compute and log patience ---
            if self.rank == 0:
                print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

                if best_loss - avg_loss > improvement_threshold:
                    best_loss = avg_loss
                    patience_counter = 0  # Reset patience counter on improvement
                    
                    # Save model - unwrap DDP/DataParallel if necessary
                    if self.use_ddp:
                        model_to_save = self.model.module
                    elif isinstance(self.model, nn.DataParallel):
                        model_to_save = self.model.module
                    else:
                        model_to_save = self.model
                    torch.save(model_to_save.state_dict(), save_dir / "best.pt")
                    print(f"  ✓ Saved new best model (loss={best_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  ! No improvement. Patience: {patience_counter}/{patience}")
            
            # --- All ranks: synchronize early-stopping decision ---
            if self.use_ddp:
                # Rank 0 holds the decision, others send a dummy False; broadcast overwrites it
                should_stop_tensor = torch.tensor(
                    patience_counter >= patience if self.rank == 0 else False,
                    dtype=torch.bool, device=self.device
                )
                dist.broadcast(should_stop_tensor, src=0)
                if should_stop_tensor.item():
                    if self.rank == 0:
                        print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                        print(f"Best loss achieved: {best_loss:.4f}")
                    # Synchronize before breaking
                    dist.barrier()
                    break
            else:
                # Non-DDP early stopping
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                    print(f"Best loss achieved: {best_loss:.4f}")
                    break
            
            # Checkpoint saving (only rank 0)
            if self.rank == 0:
                    checkpoint_path = save_dir / f"checkpoint_epoch{epoch+1}.pt"
                    
                    # Unwrap DDP/DataParallel if necessary
                    if self.use_ddp:
                        model_to_save = self.model.module
                    elif isinstance(self.model, nn.DataParallel):
                        model_to_save = self.model.module
                    else:
                        model_to_save = self.model
                    
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": avg_loss,
                        },
                        checkpoint_path,
                    )
                    print(f"  ✓ Saved checkpoint to {checkpoint_path}")

        if self.rank == 0:
            print(f"\nTraining complete. Best loss: {best_loss:.4f}")
            print(f"Models saved in: {save_dir}")
        
        # Cleanup DDP
        if self.use_ddp:
            dist.destroy_process_group()


def get_coco_dataloaders(
    coco_root: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    use_ddp: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create COCO dataloader with optional DDP support."""
    train_images = coco_root / "images" / "train2017"
    train_annotations = coco_root / "annotations" / "instances_train2017.json"

    if not train_images.exists() or not train_annotations.exists():
        raise FileNotFoundError("COCO training data not found. Check coco_root path.")

    dataset = CocoDetectionDataset(train_images, train_annotations)

    # Use DistributedSampler for DDP
    if use_ddp:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # DDP sampler handles shuffling
            shuffle=False,    # Must be False when using sampler
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,   # Avoids odd-sized last batches across ranks
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return loader


def train_ddp_worker(rank, world_size, trainer_config, train_config):
    """Worker function for DDP training (runs on each GPU)."""
    # Setup DDP for this process
    trainer_config['trainer'].setup_ddp(rank, world_size)
    trainer_config['trainer'].setup_model()
    
    # Create dataloader for this process
    train_loader = get_coco_dataloaders(
        train_config['coco_root'],
        batch_size=train_config['batch_size_per_gpu'],
        num_workers=train_config['num_workers'],
        use_ddp=True,
        world_size=world_size,
        rank=rank,
    )
    
    # Train
    trainer_config['trainer'].train(
        train_loader=train_loader,
        epochs=train_config['epochs'],
        patience=train_config['patience'],
    )


def main():
    COCO_ROOT = Path("/home/almaankhan/data/coco")
    EPOCHS = 10
    BATCH_SIZE_PER_GPU = 4  # Batch size per GPU
    NUM_WORKERS = 4
    GPU_DEVICES = [0, 1, 2]
    MODEL_DIR = "/home/almaankhan/model"  # Directory with pretrained weights
    MODELS = ["retinanet"]
    USE_DDP = True  # Set to False to use DataParallel instead

    if not COCO_ROOT.exists():
        print(f"Error: COCO root not found at {COCO_ROOT}")
        return

    world_size = len(GPU_DEVICES)
    
    if USE_DDP:
        print(f"\n[INFO] Using DistributedDataParallel (DDP)")
        print(f"  - GPUs: {GPU_DEVICES}")
        print(f"  - Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"  - Total effective batch size: {BATCH_SIZE_PER_GPU * world_size}\n")
    else:
        print(f"\n[INFO] Using DataParallel")
        print(f"  - GPUs: {GPU_DEVICES}")
        print(f"  - Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        print(f"  - Total batch size: {BATCH_SIZE_PER_GPU * world_size}\n")

    for model_name in MODELS:
        print("\n" + "#" * 60)
        print(f"# Training {model_name.upper()} with TorchVision + Focal Loss")
        print("#" * 60 + "\n")

        # Initialize trainer
        trainer = TorchVisionDetectionTrainer(
            model_name=model_name, 
            device_ids=GPU_DEVICES, 
            model_dir=MODEL_DIR,
            focal_alpha=0.25,  # Weighting factor for positive class
            focal_gamma=2.0,   # Focusing parameter to down-weight easy examples
            batch_size=BATCH_SIZE_PER_GPU,  # For display purposes
            use_ddp=USE_DDP
        )
        
        if USE_DDP:
            # Launch DDP training across multiple processes
            import torch.multiprocessing as mp
            
            trainer_config = {'trainer': trainer}
            train_config = {
                'coco_root': COCO_ROOT,
                'batch_size_per_gpu': BATCH_SIZE_PER_GPU,
                'num_workers': NUM_WORKERS,
                'epochs': EPOCHS,
                'patience': 10,
            }
            
            mp.spawn(
                train_ddp_worker,
                args=(world_size, trainer_config, train_config),
                nprocs=world_size,
                join=True
            )
        else:
            # Regular training with DataParallel or single GPU
            trainer.setup_model()
            
            total_batch_size = BATCH_SIZE_PER_GPU * len(GPU_DEVICES) if len(GPU_DEVICES) > 1 else BATCH_SIZE_PER_GPU
            
            try:
                train_loader = get_coco_dataloaders(
                    COCO_ROOT,
                    batch_size=total_batch_size,
                    num_workers=NUM_WORKERS,
                    use_ddp=False,
                )
            except FileNotFoundError as exc:
                print(exc)
                return
            
            trainer.train(
                train_loader=train_loader, 
                epochs=EPOCHS,
                patience=10
            )

        print(f"\n✓ {model_name.upper()} training complete!\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print("\n" + "=" * 60)
        print("GPU Information")
        print("=" * 60)
        print(f"Total GPUs Available: {gpu_count}\n")
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            memory_gb = props.total_memory / 1e9
            print(f"GPU {idx}: {props.name} ({memory_gb:.1f} GB)")
        print("=" * 60 + "\n")
    else:
        print("Warning: GPU not available. Training on CPU will be extremely slow.\n")

    main()
