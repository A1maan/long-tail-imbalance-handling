"""
Training script for YOLO and RT-DETR models with Focal Loss
Integrates Ultralytics' built-in Focal Loss for handling class imbalance

Supported Models:
- YOLOv8 (all sizes: n, s, m, l, x)
- YOLOv11 (all sizes)  
- YOLOv12 (all sizes)
- RT-DETR (all sizes: l, x)

Features:
- Focal Loss (from Ultralytics) for class imbalance (gamma=2.0, alpha=0.25)
- Gradient-based attribution for saliency analysis
- Multi-GPU distributed training
- Automatic model architecture detection
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
current_pythonpath = os.environ.get('PYTHONPATH', '')
if script_dir not in current_pythonpath:
    os.environ['PYTHONPATH'] = f"{script_dir}:{current_pythonpath}" if current_pythonpath else script_dir
    print(f"[INFO] Set PYTHONPATH to: {os.environ['PYTHONPATH']}")

# Add to sys.path for current process too
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from custom_trainer import CustomDetectionTrainer

def print_model_loss_info(model):
    """
    Print detailed information about the model's loss function
    Works with both YOLO and RT-DETR models
    
    Args:
        model: YOLO or RT-DETR model object
    """
    print(f"\n{'='*60}")
    print(f"Model Loss Function Information")
    print(f"{'='*60}")
    
    # Access the underlying model
    if hasattr(model, 'model'):
        yolo_model = model.model
        print(f"Model type: {type(yolo_model).__name__}")
        print(f"Model class: {yolo_model.__class__.__name__}")
        
        # Check for criterion
        if hasattr(yolo_model, 'criterion'):
            criterion = yolo_model.criterion
            criterion_name = type(criterion).__name__
            print(f"✓ Criterion found: {criterion_name}")
            
            # Check if this is an RT-DETR model (uses DETRLoss/RTDETRDetectionLoss)
            if 'DETR' in criterion_name or 'RTDETR' in criterion_name:
                print(f"  Model Type: RT-DETR (Transformer-based)")
                print(f"  Loss Type: {criterion_name}")
                
                # Check for focal loss components
                if hasattr(criterion, 'vfl') and criterion.vfl is not None:
                    vfl_type = type(criterion.vfl).__name__
                    print(f"  Classification Loss (vfl): {vfl_type}")
                    if hasattr(criterion.vfl, 'gamma'):
                        print(f"    - Gamma: {criterion.vfl.gamma}")
                    if hasattr(criterion.vfl, 'alpha'):
                        print(f"    - Alpha: {criterion.vfl.alpha}")
                elif hasattr(criterion, 'fl') and criterion.fl is not None:
                    fl_type = type(criterion.fl).__name__
                    print(f"  Classification Loss (fl): {fl_type}")
                    if 'FocalLoss' in fl_type:
                        print(f"    ✓ Using FOCAL LOSS!")
                        if hasattr(criterion.fl, 'gamma'):
                            print(f"      - Gamma: {criterion.fl.gamma}")
                        if hasattr(criterion.fl, 'alpha'):
                            alpha_val = criterion.fl.alpha
                            print(f"      - Alpha: {alpha_val}")
                else:
                    print(f"  ⚠ No focal/varifocal loss component found")
                    
            else:
                # Standard YOLO model
                print(f"  Model Type: YOLO (Anchor-free CNN)")
                
                # Check bce component (the classification loss)
                if hasattr(criterion, 'bce'):
                    bce_type = type(criterion.bce).__name__
                    print(f"  Classification Loss (bce): {bce_type}")
                    
                    if 'FocalLoss' in bce_type or 'Focal' in bce_type:
                        print(f"    ✓ Using FOCAL LOSS!")
                        if hasattr(criterion.bce, 'alpha'):
                            alpha_val = criterion.bce.alpha
                            print(f"      - Alpha: {alpha_val}")
                        if hasattr(criterion.bce, 'gamma'):
                            print(f"      - Gamma: {criterion.bce.gamma}")
                    else:
                        print(f"    ⚠ Using standard {bce_type} (Focal Loss NOT applied)")
                else:
                    print(f"  ⚠ No 'bce' attribute found in criterion")
        else:
            print(f"✗ No criterion attribute found")
        
        # Print model structure summary
        print(f"\nModel structure:")
        total_params = sum(p.numel() for p in yolo_model.parameters())
        trainable_params = sum(p.numel() for p in yolo_model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Print detection head info
        if hasattr(yolo_model, 'model') and len(yolo_model.model) > 0:
            detect_head = yolo_model.model[-1]
            head_type = detect_head.__class__.__name__
            print(f"  - Detection head: {head_type}")
            
            if 'RTDETRDecoder' in head_type:
                print(f"    → Transformer-based decoder")
            elif 'Detect' in head_type:
                print(f"    → Anchor-free detection")
        
    print(f"{'='*60}\n")


def train_with_focal_loss(model_name="yolov8s", 
                          data_yaml="/home/almaankhan/data/coco/coco.yaml",
                          epochs=100,
                          imgsz=640,
                          batch_size=64,
                        #   patience=20,
                          device_ids=[0, 1, 2],
                          model_dir="/home/almaankhan/model/baseline"):
    """
    Train YOLO model with custom Focal Loss trainer
    
    Args:
        model_name: YOLO model size (yolov8s, yolo11s, yolo12s, rtdetr-l)
        data_yaml: Path to data.yaml file for COCO dataset
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        patience: Early stopping patience
        device_ids: List of GPU device IDs to use
        model_dir: Directory containing model weights
    """
    # Load from full path to avoid downloading
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"[INFO] Loading {model_name} from {model_path}")
    model = YOLO(model_path)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with Focal Loss")
    print(f"{'='*60}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"GPUs: {len(device_ids)} (Device IDs: {device_ids})")
    print(f"Loss Function: Focal Loss (alpha=0.25, gamma=2.0)")
    print(f"{'='*60}\n")
    
    # Print current loss function info
    print_model_loss_info(model)
    
    # Train with custom trainer as per Ultralytics documentation
    results = model.train(
        trainer=CustomDetectionTrainer,  # Pass the custom trainer class
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_ids,
        # patience=patience,
        save=True,
        project="runs/detect",
        name=f"{model_name}_focal_loss",
        verbose=True,
        amp=False,  # Disable AMP to prevent model download during checks
        optimizer='AdamW',
        # Learning rate and warmup
        lr0=1e-3,  # Initial learning rate
        lrf=0.01,  # Final learning rate (for cosine scheduler)
        warmup_epochs=0,  # No warmup
        cos_lr=True,  # Use cosine annealing scheduler
        # Data augmentation
        augment=True,
        mosaic=1.0,
        close_mosaic=10,
        cache=False,
        # Additional optimization
        fliplr=0.5,
        flipud=0.5,
        workers=4,  # Reduce workers to avoid multiprocessing issues
        val=True
    )
    
    return results



def main():
    """
    Example usage of the custom YOLO trainer with Focal Loss
    Distributed training across 3 GPUs
    """
    
    # Configuration
    DATASET_YAML = "/home/almaankhan/data/coco/coco.yaml"
    EPOCHS = 40
    MODELS = ["yolo12s", "rtdetr-l"]
    # MODELS = ["yolov8s"]
    GPU_DEVICES = [0, 1, 2]  # Start with single GPU to test, can scale to [0,1,2] later
    IMG_SIZE = 640
    # MODELS = ["yolov8s", "yolo11s", "yolo12s", "rtdetr-l"]
    BATCH_SIZE = 18
    MODEL_DIR = "/home/almaankhan/model/baseline"  # Directory with pretrained weights
    
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
                
        # Train with Focal Loss using custom trainer
        train_results = train_with_focal_loss(
            model_name=model_name,
            data_yaml=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch_size=BATCH_SIZE,
            # patience=20,
            device_ids=[GPU_DEVICES] if isinstance(GPU_DEVICES, int) else GPU_DEVICES,
            model_dir=MODEL_DIR
        )
        
        print(f"\n✓ {model_name.upper()} training complete!")
        print(f"Results saved to: runs/detect/{model_name}_focal_loss/\n")


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
