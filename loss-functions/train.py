"""
Training script for YOLO models with custom Focal Loss
Integrates custom Focal Loss with Ultralytics YOLO for handling class imbalance
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path
import sys
import os

# Add loss functions directory to path
loss_functions_dir = Path(__file__).parent
sys.path.insert(0, str(loss_functions_dir))

from FocalLoss import FocalLoss


class CustomDetectionModel(DetectionModel):
    """
    Custom YOLO Detection Model with Focal Loss
    Overrides the standard loss function with Focal Loss for imbalance handling
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
        self.focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    def init_criterion(self):
        """
        Initialize the loss function - OVERRIDDEN to use Focal Loss
        """
        # Replace standard criterion with Focal Loss
        self.criterion = self.focal_loss_fn
        print("[INFO] Initialized Focal Loss as criterion")


class CustomDetectionTrainer(DetectionTrainer):
    """
    Custom Detection Trainer that uses Focal Loss
    Extends DetectionTrainer to implement custom loss function
    """
    
    def get_model(self, cfg, weights):
        """
        Returns a customized detection model with Focal Loss
        
        Args:
            cfg: Model config
            weights: Pre-trained weights
            
        Returns:
            CustomDetectionModel instance
        """
        model = CustomDetectionModel(cfg=cfg, nc=self.data["nc"], verbose=False)
        if weights:
            model.load(weights)
        return model


class YOLOFocalLossTrainer:
    """
    Unified trainer for YOLO models with Focal Loss
    Supports distributed training across multiple GPUs
    """
    
    def __init__(self, model_name="yolov8s", device_ids=[0, 1, 2]):
        """
        Initialize the trainer
        
        Args:
            model_name: YOLO model size (yolov8n, yolov8s, yolov8m, etc.)
            device_ids: List of GPU device IDs to use (default: [0, 1, 2])
        """
        self.model_name = model_name
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        self.model = YOLO(f"{model_name}.pt")
        
    def train(self, 
              data_yaml,
              epochs=100,
              imgsz=640,
              batch_size=16,
              patience=20,
              save_dir="runs/detect",
              project_name="yolo_focal_loss"):
        """
        Train YOLO model with Focal Loss using custom trainer
        Distributes training across multiple GPUs
        
        Args:
            data_yaml: Path to data.yaml file for COCO dataset
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Total batch size (will be divided among GPUs)
            patience: Early stopping patience
            save_dir: Directory to save results
            project_name: Project name for results
        """
        
        # Calculate batch size per GPU
        batch_size_per_gpu = max(1, batch_size // self.num_gpus)
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name.upper()} with Focal Loss")
        print(f"{'='*60}")
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Total Batch Size: {batch_size}")
        print(f"Batch Size per GPU: {batch_size_per_gpu}")
        print(f"Image Size: {imgsz}")
        print(f"GPUs: {self.num_gpus} (Device IDs: {self.device_ids})")
        print(f"Loss Function: Focal Loss (alpha=0.25, gamma=2.0)")
        print(f"{'='*60}\n")
        
        # Train the model with custom trainer and distributed GPUs
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size_per_gpu,  # Per-GPU batch size
            device=self.device_ids,  # Multiple GPU IDs for distributed training
            patience=patience,
            save=True,
            project=save_dir,
            name=project_name,
            verbose=True,
            # Use custom trainer with Focal Loss
            trainer=CustomDetectionTrainer,
            # Data augmentation
            augment=True,
            mosaic=1.0,
            close_mosaic=10,
            cache=True,
            # Additional optimization
            fliplr=0.5,
            flipud=0.5,
            # Distributed training optimization
            workers=8,  # More workers for multi-GPU
        )
        
        return results



def main():
    """
    Example usage of the custom YOLO trainer with Focal Loss
    Distributed training across 3 GPUs
    """
    
    # Configuration
    DATASET_YAML = "/home/almaankhan/data/data.yaml"  # Update this path
    EPOCHS = 100
    BATCH_SIZE = 48  # Total batch size (16 per GPU with 3 GPUs)
    IMG_SIZE = 640
    MODELS = ["yolov8s", "yolo11s", "yolo12s"]
    GPU_DEVICES = [0, 1, 2]  # Use GPUs 0, 1, 2
    
    # Check if dataset exists
    if not Path(DATASET_YAML).exists():
        print(f"Error: Dataset YAML not found at {DATASET_YAML}")
        print("Please ensure you have the COCO dataset and data.yaml file ready")
        return
    
    # Train multiple models
    for model_name in MODELS:
        print(f"\n{'#'*60}")
        print(f"# Training {model_name.upper()}")
        print(f"{'#'*60}\n")
        
        # Initialize trainer with multi-GPU support
        trainer = YOLOFocalLossTrainer(model_name=model_name, device_ids=GPU_DEVICES)
        
        # Train with Focal Loss (distributed across 3 GPUs)
        train_results = trainer.train(
            data_yaml=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch_size=BATCH_SIZE,  # Total batch size
            patience=20,
            project_name=f"{model_name}_focal_loss_distributed",
        )
        
        print(f"\n✓ {model_name.upper()} training complete!")
        print(f"Results saved to: runs/detect/{model_name}_focal_loss_distributed/\n")


if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"GPU Information")
        print(f"{'='*60}")
        print(f"Total GPUs Available: {num_gpus}\n")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        print(f"{'='*60}\n")
        
        if num_gpus < 3:
            print(f"Warning: Only {num_gpus} GPU(s) found, but script is configured for 3 GPUs")
            print(f"Adjusting to use available GPU(s)...\n")
    else:
        print("GPU not available, using CPU (training will be very slow)\n")
    
    main()
