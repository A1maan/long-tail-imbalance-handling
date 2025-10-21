"""
Training script for TorchVision object detection models with Focal Loss
Supports Faster R-CNN (with custom Focal Loss) and RetinaNet (built-in Focal Loss)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
import sys

# Add loss functions directory to path
loss_functions_dir = Path(__file__).parent
sys.path.insert(0, str(loss_functions_dir))

from FocalLoss import FocalLoss


class CocoDetectionDataset(Dataset):
    """Minimal COCO detection dataset that returns image tensors and detection targets."""

    def __init__(self, image_dir: Path, ann_file: Path, image_transform=None):
        self.coco = COCO(str(ann_file))
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
        image = Image.open(image_path).convert("RGB")

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


def patch_faster_rcnn_focal_loss(focal_loss_fn: FocalLoss) -> None:
    """Monkey-patch torchvision's Faster R-CNN classification loss to use Focal Loss."""
    import torchvision.models.detection.roi_heads as roi_heads_module

    def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        # Classification loss with Focal Loss
        classification_loss = focal_loss_fn(class_logits, labels)

        # Bounding box regression loss (same as original implementation)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        if sampled_pos_inds_subset.numel() == 0:
            box_loss = torch.tensor(0.0, device=box_regression.device)
        else:
            box_regression = box_regression.reshape(class_logits.shape[0], -1, 4)
            box_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                beta=1.0 / 9.0,
                reduction="sum",
            )
            box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    roi_heads_module.fastrcnn_loss = custom_fastrcnn_loss
    print("[INFO] Patched Faster R-CNN classifier loss with Focal Loss")


class TorchVisionDetectionTrainer:
    """Trainer for TorchVision detection models using Focal Loss."""

    def __init__(self, model_name: str = "faster_rcnn", device_ids: List[int] = None, num_classes: int = 91, model_dir: str = "/home/almaankhan/model"):
        device_ids = device_ids or [0]
        self.device_ids = device_ids
        self.primary_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_dir = model_dir

        if model_name == "faster_rcnn":
            focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
            patch_faster_rcnn_focal_loss(focal_loss)
            base_model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
            # Load fine-tuned weights if available
            weights_path = Path(model_dir) / "faster_rcnn_resnet50_fpn_v2.pt"
            if weights_path.exists():
                print(f"[INFO] Loading Faster R-CNN weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.primary_device)
                base_model.load_state_dict(state_dict, strict=False)
        elif model_name == "retinanet":
            print("[INFO] RetinaNet already uses Focal Loss internally")
            base_model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
            # Load fine-tuned weights if available
            weights_path = Path(model_dir) / "retinanet_resnet50_fpn_v2.pt"
            if weights_path.exists():
                print(f"[INFO] Loading RetinaNet weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.primary_device)
                base_model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        if len(device_ids) > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(base_model, device_ids=device_ids)
            print(f"[INFO] Using DataParallel on devices: {device_ids}")
        else:
            self.model = base_model

        self.model.to(self.primary_device)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.005,
        weight_decay: float = 0.0005,
        save_dir: Path = Path("runs/torchvision"),
    ) -> None:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        save_dir = save_dir / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"{self.model_name} Epoch {epoch+1}/{epochs}")
            for images, targets in pbar:
                images = [img.to(self.primary_device) for img in images]
                targets = [
                    {k: v.to(self.primary_device) for k, v in target.items()}
                    for target in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                pbar.set_postfix({"loss": losses.item()})

            lr_scheduler.step()
            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_dir / "best.pt")
                print(f"  ✓ Saved new best model (loss={best_loss:.4f})")

            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    checkpoint_path,
                )
                print(f"  ✓ Saved checkpoint to {checkpoint_path}")

        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        print(f"Models saved in: {save_dir}")


def get_coco_dataloaders(
    coco_root: Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    train_images = coco_root / "images" / "train2017"
    train_annotations = coco_root / "annotations" / "instances_train2017.json"

    if not train_images.exists() or not train_annotations.exists():
        raise FileNotFoundError("COCO training data not found. Check coco_root path.")

    dataset = CocoDetectionDataset(train_images, train_annotations)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return loader


def main():
    COCO_ROOT = Path("/home/almaankhan/data/coco")
    EPOCHS = 100
    BATCH_SIZE = 12  # Total batch size (will be divided by number of GPUs when using DataParallel)
    NUM_WORKERS = 8
    GPU_DEVICES = [0, 1, 2]
    MODEL_DIR = "/home/almaankhan/model"  # Directory with pretrained weights
    MODELS = ["faster_rcnn", "retinanet"]

    if not COCO_ROOT.exists():
        print(f"Error: COCO root not found at {COCO_ROOT}")
        return

    try:
        train_loader = get_coco_dataloaders(
            COCO_ROOT,
            batch_size=max(1, BATCH_SIZE // max(len(GPU_DEVICES), 1)),
            num_workers=NUM_WORKERS,
        )
    except FileNotFoundError as exc:
        print(exc)
        return

    for model_name in MODELS:
        print("\n" + "#" * 60)
        print(f"# Training {model_name.upper()} with TorchVision")
        print("#" * 60 + "\n")

        trainer = TorchVisionDetectionTrainer(model_name=model_name, device_ids=GPU_DEVICES, model_dir=MODEL_DIR)
        trainer.train(train_loader=train_loader, epochs=EPOCHS)

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
