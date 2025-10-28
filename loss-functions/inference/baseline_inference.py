"""
Inference script to evaluate all object detection models on COCO validation dataset
Generates comprehensive baseline metrics including mAP and class-wise performance
Critical for comparing imbalance handling techniques
"""

import torch
import torchvision
from ultralytics import YOLO, RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from pathlib import Path
from datetime import datetime
import json
import csv
import numpy as np
from tqdm import tqdm
import sys

# Import custom_trainer to make it available for YOLO model loading
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from custom_trainer import CustomDetectionTrainer, FLDetectionModel, FLRTDETRModel
    
    # Register safe globals for PyTorch 2.6+ compatibility
    import torch.serialization
    torch.serialization.add_safe_globals([FLDetectionModel, FLRTDETRModel, CustomDetectionTrainer])
    
    # Also add the old class name for backwards compatibility if model was saved with old name
    try:
        # Create an alias for the old class name
        import sys
        import custom_trainer
        if not hasattr(custom_trainer, 'CustomRTDETRModel'):
            custom_trainer.CustomRTDETRModel = FLRTDETRModel
        torch.serialization.add_safe_globals([custom_trainer.CustomRTDETRModel])
    except Exception:
        pass
        
except ImportError as e:
    print(f"Warning: Could not import custom_trainer: {e}")
    print("Some models with custom trainers may not load correctly")

class BaselineInferenceRunner:
    """
    Runs inference on all available models and generates comprehensive baseline metrics
    Stores primary metrics (mAP, precision, recall) and class-wise performance
    Essential for comparing imbalance handling techniques
    """
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Class distribution: head (frequent), medium, tail (rare)
    # Based on COCO dataset distribution
    HEAD_CLASSES = ['person', 'car', 'dog', 'cat', 'bicycle', 'boat', 'bird', 'chair', 
                    'cow', 'horse', 'sheep', 'traffic light', 'stop sign', 'bench']
    TAIL_CLASSES = ['toothbrush', 'hair drier', 'scissors', 'vase', 'clock', 'book',
                    'remote', 'keyboard', 'mouse', 'laptop', 'oven', 'toaster', 'sink']
    
    def __init__(self, val_dir="/home/almaankhan/data/coco/images/val2017",
                 ann_file="/home/almaankhan/data/coco/annotations/instances_val2017.json",
                 model_dir="/home/almaankhan/model/loss-functions/focal-loss",
                 output_dir="runs/inference"):
        """
        Initialize inference runner
        
        Args:
            val_dir: Path to validation images
            ann_file: Path to COCO annotations JSON
            model_dir: Path to model weights
            output_dir: Directory to save results
        """
        self.val_dir = val_dir
        self.ann_file = ann_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create separate directories for ultralytics and torchvision
        self.ultralytics_output_dir = Path(self.output_dir) / "ultralytics"
        self.torchvision_output_dir = Path(self.output_dir) / "torchvision"
        self.ultralytics_output_dir.mkdir(parents=True, exist_ok=True)
        self.torchvision_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.detailed_metrics = {}
        
        # Load COCO annotations and create proper YOLO-to-COCO category ID mapping
        # CRITICAL: COCO has 80 classes with IDs 1-90 (non-contiguous, missing 11 IDs)
        # YOLO uses contiguous indices 0-79
        # DO NOT use yolo_index + 1 = coco_id (this is WRONG!)
        print("Loading COCO annotations for category mapping...")
        coco_gt = COCO(self.ann_file)
        cats = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda x: x['id'])
        self.yolo_to_coco_id = {idx: cat['id'] for idx, cat in enumerate(cats)}
        print(f"✓ Created YOLO→COCO mapping for {len(self.yolo_to_coco_id)} categories")
        
    def calculate_metrics_from_coco(self, model_name, results_coco, compute_per_class=True):
        """
        Calculate comprehensive metrics using COCO evaluation API
        
        Args:
            model_name: Name of the model
            results_coco: COCO results format predictions
            compute_per_class: Whether to compute per-class metrics (slow)
            
        Returns:
            Dictionary with all metrics
        """
        try:
            # Initialize COCO API
            coco_gt = COCO(self.ann_file)
            coco_dt = coco_gt.loadRes(results_coco)
            
            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract primary metrics
            stats = coco_eval.stats
            metrics = {
                "model_name": model_name,
                "mAP@0.5": float(stats[1]),
                "mAP@0.75": float(stats[2]),
                "mAP@[0.5:0.95]": float(stats[0]),  # Overall mAP
                "precision": float(stats[3]),
                "recall": float(stats[8]),
            }
            
            # Calculate class-wise performance (optional, can be slow)
            class_metrics = {}
            if compute_per_class:
                print(f"  Computing per-class metrics (this may take a while)...")
                for idx, class_name in enumerate(self.COCO_CLASSES):
                    # Use proper COCO category ID (not idx + 1!)
                    coco_cat_id = self.yolo_to_coco_id[idx]
                    
                    coco_eval_class = COCOeval(coco_gt, coco_dt, "bbox")
                    coco_eval_class.params.catIds = [coco_cat_id]
                    coco_eval_class.evaluate()
                    coco_eval_class.accumulate()
                    
                    # Check if stats exist AND the AP value is not NaN
                    # (NaN happens when a category has no predictions or no ground truth instances)
                    if len(coco_eval_class.stats) > 0 and not np.isnan(coco_eval_class.stats[0]):
                        class_metrics[class_name] = {
                            "AP": float(coco_eval_class.stats[0]),
                            "AP@0.5": float(coco_eval_class.stats[1]),
                            "AP@0.75": float(coco_eval_class.stats[2]),
                        }
            else:
                print(f"  Skipping per-class metrics (use compute_per_class=True to enable)")
            
            # Aggregate head/medium/tail class performance
            head_aps = []
            medium_aps = []
            tail_aps = []
            
            if compute_per_class and class_metrics:
                # Debug: show what we have
                print(f"  ✓ Computing class aggregates from {len(class_metrics)} classes with valid AP")
                
                head_aps = [class_metrics[c]["AP"] for c in self.HEAD_CLASSES if c in class_metrics]
                tail_aps = [class_metrics[c]["AP"] for c in self.TAIL_CLASSES if c in class_metrics]
                
                metrics["head_classes_AP"] = float(np.mean(head_aps)) if head_aps else 0.0
                metrics["tail_classes_AP"] = float(np.mean(tail_aps)) if tail_aps else 0.0
                
                # Calculate medium classes AP
                all_class_names = set(self.COCO_CLASSES)
                medium_class_names = all_class_names - set(self.HEAD_CLASSES) - set(self.TAIL_CLASSES)
                medium_aps = [class_metrics[c]["AP"] for c in medium_class_names if c in class_metrics]
                metrics["medium_classes_AP"] = float(np.mean(medium_aps)) if medium_aps else 0.0
                
                # Debug: show aggregation results
                print(f"    - Head classes with AP: {len(head_aps)}/{len(self.HEAD_CLASSES)}")
                print(f"    - Medium classes with AP: {len(medium_aps)}/{len(medium_class_names)}")
                print(f"    - Tail classes with AP: {len(tail_aps)}/{len(self.TAIL_CLASSES)}")
            elif compute_per_class:
                # Debug: show why aggregation failed
                print(f"  ⚠ Per-class metrics requested but class_metrics is empty!")
                print(f"    This means all categories returned NaN (no valid predictions)")
                # Set to None if not computed
                metrics["head_classes_AP"] = None
                metrics["tail_classes_AP"] = None
                metrics["medium_classes_AP"] = None
            else:
                # Set to None if not computed
                metrics["head_classes_AP"] = None
                metrics["tail_classes_AP"] = None
                metrics["medium_classes_AP"] = None
            
            # Store detailed per-class metrics
            self.detailed_metrics[model_name] = {
                "primary_metrics": metrics,
                "class_wise": class_metrics,
                "head_classes": head_aps,
                "medium_classes": medium_aps,
                "tail_classes": tail_aps,
            }
            
            return metrics
            
        except Exception as e:
            print(f"✗ Error calculating COCO metrics: {str(e)}")
            return None
    
    def run_yolo_inference(self, model_path, model_name, confidence=0.25, compute_per_class=True, quick_test=False):
        """
        Run inference with YOLO models (YOLOv8, YOLOv11, YOLOv12, RT-DETR)
        
        Args:
            model_path: Path to model weights
            model_name: Name of the model
            confidence: Confidence threshold
            compute_per_class: Whether to compute per-class metrics (default: True for full evaluation)
            quick_test: If True, only process 100 images for quick testing
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\n{'='*60}")
        print(f"Running inference: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print(f"Loading model from: {model_path}")
            try:
                # For RT-DETR models, try loading with task specification
                if "rtdetr" in model_name.lower() or "rt-detr" in model_name.lower():
                    print(f"  Detected RT-DETR model, loading with task='detect'")
                    model = RTDETR(model_path)
                else:
                    model = YOLO(model_path)
            except ModuleNotFoundError as e:
                if "custom_trainer" in str(e):
                    print(f"⚠️  Model requires 'custom_trainer' module for inference")
                    print(f"   This model was trained with CustomDetectionTrainer")
                    print(f"   To run inference, you need to:")
                    print(f"   1. Import custom_trainer: from custom_trainer import CustomDetectionTrainer")
                    print(f"   2. OR use an official Ultralytics model")
                    print(f"   Skipping this model...")
                    return None
                else:
                    raise
            except Exception as e:
                print(f"⚠️  Error loading model: {str(e)}")
                print(f"   Attempting alternative loading method...")
                try:
                    # Try loading with weights_only=False for PyTorch 2.6+ compatibility
                    import torch
                    
                    # For RT-DETR models, try loading with weights_only=False
                    if "rtdetr" in model_name.lower():
                        print(f"   RT-DETR model detected - attempting load with weights_only=False")
                        
                        # First, ensure CustomRTDETRModel alias exists
                        import custom_trainer
                        if not hasattr(custom_trainer, 'CustomRTDETRModel'):
                            custom_trainer.CustomRTDETRModel = custom_trainer.FLRTDETRModel
                            print(f"   Created alias: CustomRTDETRModel -> FLRTDETRModel")
                        
                        # Try loading the model with YOLO, which should now work
                        try:
                            model = YOLO(model_path, task='detect')
                            print(f"   ✓ Successfully loaded RT-DETR model")
                        except Exception as load_err:
                            print(f"   Still failed: {str(load_err)}")
                            print(f"   Skipping this model...")
                            return None
                    else:
                        # For non-RT-DETR, try standard loading
                        model = YOLO(model_path)
                except Exception as e2:
                    print(f"✗ Failed to load model: {str(e2)}")
                    return None
            
            # Run inference on validation set
            print(f"Model device: {self.device}")
            print(f"Inference on: {self.val_dir}")
            
            results = model.predict(
                source=self.val_dir,
                conf=confidence,
                device=0 if self.device == "cuda" else "cpu",
                verbose=False,
                save=False,
            )
            
            # Convert to COCO format
            coco_results = []
            for result in results:
                # Extract image ID from filename (e.g., "000000000139.jpg" -> 139)
                image_id = int(Path(result.path).stem)
                
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    conf = float(box.conf[0])
                    yolo_cls = int(box.cls[0])  # YOLO class index (0-79)
                    
                    # Convert YOLO index to COCO category ID using proper mapping
                    # CRITICAL: COCO IDs are non-contiguous (1-90 with gaps)
                    coco_cat_id = self.yolo_to_coco_id[yolo_cls]
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": coco_cat_id,  # Correct COCO category ID
                        "bbox": [float(x1), float(y1), w, h],
                        "score": conf
                    })
            
            # Save predictions to JSON first
            predictions_dir = Path(self.output_dir) / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            pred_file = predictions_dir / f"{model_name.replace(' ', '_')}_predictions.json"
            
            with open(pred_file, 'w') as f:
                json.dump(coco_results, f)
            print(f"  ✓ Saved predictions to: {pred_file}")
            
            # Calculate primary metrics only (skip per-class for now)
            metrics = self.calculate_metrics_from_coco(model_name, coco_results, compute_per_class=False)
            
            if metrics:
                # Add inference speed and basic stats
                num_images = len(results)
                total_detections = sum([len(r.boxes) for r in results])
                avg_detections = total_detections / num_images if num_images > 0 else 0
                inference_time = results[0].speed['inference'] if results else 0
                
                # Debug info
                print(f"  - COCO format predictions: {len(coco_results)}")
                print(f"  - Unique image IDs: {len(set(r['image_id'] for r in coco_results))}")
                
                metrics.update({
                    "model_type": "YOLO-based",
                    "num_images": num_images,
                    "total_detections": total_detections,
                    "avg_detections_per_image": avg_detections,
                    "confidence_threshold": confidence,
                    "inference_time_ms": inference_time,
                    "device": self.device,
                    "timestamp": datetime.now().isoformat(),
                })
                
                print(f"✓ {model_name} inference complete!")
                print(f"  - mAP@[0.5:0.95]: {metrics['mAP@[0.5:0.95]']:.4f}")
                if metrics.get('head_classes_AP') is not None:
                    print(f"  - Head classes AP: {metrics['head_classes_AP']:.4f}")
                    print(f"  - Tail classes AP: {metrics['tail_classes_AP']:.4f}")
                print(f"  - Precision: {metrics['precision']:.4f}")
                print(f"  - Recall: {metrics['recall']:.4f}")
                
                return metrics
            
        except Exception as e:
            print(f"✗ Error running inference on {model_name}: {str(e)}")
            return None
    
    def run_torchvision_inference(self, model_name, model_type="faster_rcnn", use_coco_eval=True, compute_per_class=True):
        """
        Run inference with TorchVision models (Faster R-CNN, Mask R-CNN, RetinaNet)
        
        Args:
            model_name: Name of the model
            model_type: Type of model (faster_rcnn, mask_rcnn, retinanet)
            use_coco_eval: Whether to run full COCO evaluation (default: True)
            compute_per_class: Whether to compute per-class metrics (default: True for full evaluation)
            
        Returns:
            Dictionary with inference metrics
        """
        print(f"\n{'='*60}")
        print(f"Running inference: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            if model_type == "faster_rcnn":
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
                weights_path = os.path.join(self.model_dir, "faster_rcnn/best.pt")
            
            # Load weights
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"  - Loaded weights from: {weights_path}")
            else:
                print(f"  - Warning: Weights not found at {weights_path}, using random initialization")
            
            model.to(self.device)
            model.eval()
            
            # Count images
            val_images = list(Path(self.val_dir).glob("*.jpg"))
            print(f"Model device: {self.device}")
            print(f"Number of validation images: {len(val_images)}")
            
            # Determine how many images to process
            num_images_to_process = len(val_images) if use_coco_eval else 100
            print(f"Processing: {num_images_to_process} images")
            
            # Collect COCO format predictions if using COCO eval
            coco_results = [] if use_coco_eval else None
            total_detections = 0
            inference_times = []
            from PIL import Image
            
            with torch.no_grad():
                for idx, img_path in enumerate(val_images[:num_images_to_process]):
                    if (idx + 1) % 500 == 0:
                        print(f"  - Processed {idx+1}/{num_images_to_process} images...")
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                        img_tensor = img_tensor.unsqueeze(0).to(self.device)
                        
                        # Inference timing
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if start_time:
                            start_time.record()
                        
                        outputs = model(img_tensor)
                        
                        if end_time:
                            end_time.record()
                            torch.cuda.synchronize()
                            inference_time = start_time.elapsed_time(end_time)
                            inference_times.append(inference_time)
                        
                        num_detections = len(outputs[0]["boxes"])
                        total_detections += num_detections
                        
                        # Convert to COCO format if needed
                        if use_coco_eval:
                            image_id = int(img_path.stem)  # Extract image ID from filename
                            boxes = outputs[0]["boxes"].cpu().numpy()
                            scores = outputs[0]["scores"].cpu().numpy()
                            labels = outputs[0]["labels"].cpu().numpy()
                            
                            for box, score, label in zip(boxes, scores, labels):
                                x1, y1, x2, y2 = box
                                w = x2 - x1
                                h = y2 - y1
                                
                                # TorchVision uses COCO category IDs directly (1-91)
                                # But we need to map them to our dataset's categories
                                # TorchVision: 1-indexed (1-90 with gaps, same as COCO)
                                coco_cat_id = int(label)
                                
                                # Only include if score > 0.5
                                if score > 0.5:
                                    coco_results.append({
                                        "image_id": image_id,
                                        "category_id": coco_cat_id,
                                        "bbox": [float(x1), float(y1), float(w), float(h)],
                                        "score": float(score)
                                    })
                        
                    except Exception as e:
                        print(f"  Warning: Could not process {img_path.name}: {str(e)}")
                        continue
            
            avg_detections = total_detections / num_images_to_process if num_images_to_process > 0 else 0
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            
            # Save predictions to JSON first
            if use_coco_eval and coco_results:
                predictions_dir = Path(self.output_dir) / "predictions"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                pred_file = predictions_dir / f"{model_name.replace(' ', '_')}_predictions.json"
                
                with open(pred_file, 'w') as f:
                    json.dump(coco_results, f)
                print(f"  ✓ Saved predictions to: {pred_file}")
            
            metric = {
                "model_name": model_name,
                "model_type": "TorchVision",
                "num_images": num_images_to_process,
                "total_detections": total_detections,
                "avg_detections_per_image": avg_detections,
                "confidence_threshold": 0.5,
                "inference_time_ms": avg_inference_time,
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Run COCO evaluation if requested (skip per-class for now)
            if use_coco_eval and coco_results:
                print(f"  - Running COCO evaluation on {len(coco_results)} predictions...")
                coco_metrics = self.calculate_metrics_from_coco(model_name, coco_results, compute_per_class=False)
                if coco_metrics:
                    metric.update(coco_metrics)
            
            print(f"✓ {model_name} inference complete!")
            print(f"  - Images processed: {num_images_to_process}")
            print(f"  - Total detections: {total_detections}")
            print(f"  - Avg detections/image: {avg_detections:.2f}")
            print(f"  - Avg inference time: {avg_inference_time:.2f}ms")
            if "mAP@[0.5:0.95]" in metric:
                print(f"  - mAP@[0.5:0.95]: {metric['mAP@[0.5:0.95]']:.4f}")
            
            return metric
            
        except Exception as e:
            print(f"✗ Error running inference on {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_inferences(self):
        """
        Run inference on all available models
        """
        print(f"\n{'#'*60}")
        print(f"# BASELINE INFERENCE - All Models")
        print(f"{'#'*60}")
        print(f"Validation dataset: {self.val_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        # # YOLO-based models
        # NOTE: RT-DETR models trained with custom trainers may fail to load for inference
        # If RT-DETR fails, comment it out or train without custom trainer for inference
        yolo_models = [
            ("yolo8/best.pt", "YOLOv8s"),
            ("yolo11/best.pt", "YOLO11s"),
            ("yolo12/best.pt", "YOLO12s"),
            ("rtdetr-l/best.pt", "RT-DETR-L"),  # Comment out if loading fails
        ]
        
        print(f"\n{'*'*60}")
        print(f"* YOLO-based Models")
        print(f"{'*'*60}")
        
        for model_file, model_name in yolo_models:
            model_path = os.path.join(self.model_dir, model_file)
            if os.path.exists(model_path):
                result = self.run_yolo_inference(model_path, model_name)
                if result:
                    self.results.append(result)
            else:
                print(f"✗ Model not found: {model_path}")
        
        # TorchVision models
        print(f"\n{'*'*60}")
        print(f"* TorchVision Models")
        print(f"{'*'*60}")
        
        torchvision_models = [
            ("faster_rcnn/best.pt", "Faster R-CNN", "faster_rcnn"),
        ]
        
        for model_file, model_name, model_type in torchvision_models:
            model_path = os.path.join(self.model_dir, model_file)
            if os.path.exists(model_path):
                # Run with full COCO evaluation on all validation images
                result = self.run_torchvision_inference(model_name, model_type, use_coco_eval=True)
                if result:
                    self.results.append(result)
            else:
                print(f"✗ Model not found: {model_path}")
        
        return self.results
    
    def save_results(self):
        """
        Save comprehensive inference results to JSON, CSV, and class-wise analysis
        Separates outputs into ultralytics and torchvision directories
        """
        if not self.results:
            print("No results to save")
            return
        
        # Separate results by model type
        ultralytics_results = [r for r in self.results if r.get("model_type") == "YOLO-based"]
        torchvision_results = [r for r in self.results if r.get("model_type") != "YOLO-based"]
        
        # 1. Save primary metrics (JSON) - separated by framework
        if ultralytics_results:
            json_path = os.path.join(self.ultralytics_output_dir, "results_primary_metrics.json")
            with open(json_path, "w") as f:
                json.dump(ultralytics_results, f, indent=2)
            print(f"\n✓ Ultralytics primary metrics saved to: {json_path}")
        
        if torchvision_results:
            json_path = os.path.join(self.torchvision_output_dir, "results_primary_metrics.json")
            with open(json_path, "w") as f:
                json.dump(torchvision_results, f, indent=2)
            print(f"\n✓ TorchVision primary metrics saved to: {json_path}")
        
        # 2. Save primary metrics (CSV) - separated by framework
        if ultralytics_results:
            csv_path = os.path.join(self.ultralytics_output_dir, "results_summary.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ultralytics_results[0].keys())
                writer.writeheader()
                writer.writerows(ultralytics_results)
            print(f"✓ Ultralytics summary CSV saved to: {csv_path}")
        
        if torchvision_results:
            csv_path = os.path.join(self.torchvision_output_dir, "results_summary.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=torchvision_results[0].keys())
                writer.writeheader()
                writer.writerows(torchvision_results)
            print(f"✓ TorchVision summary CSV saved to: {csv_path}")
        
        # 3. Save detailed class-wise metrics (JSON) - separated by framework
        if self.detailed_metrics:
            ultralytics_detailed = {k: v for k, v in self.detailed_metrics.items() 
                                   if any("YOLO" in model_name for model_name in self.detailed_metrics[k].keys())}
            torchvision_detailed = {k: v for k, v in self.detailed_metrics.items() 
                                   if not any("YOLO" in model_name for model_name in self.detailed_metrics[k].keys())}
            
            if ultralytics_detailed:
                detailed_path = os.path.join(self.ultralytics_output_dir, "results_class_wise.json")
                with open(detailed_path, "w") as f:
                    json.dump(ultralytics_detailed, f, indent=2)
                print(f"✓ Ultralytics class-wise metrics saved to: {detailed_path}")
            
            if torchvision_detailed:
                detailed_path = os.path.join(self.torchvision_output_dir, "results_class_wise.json")
                with open(detailed_path, "w") as f:
                    json.dump(torchvision_detailed, f, indent=2)
                print(f"✓ TorchVision class-wise metrics saved to: {detailed_path}")
        
        # 4. Create comprehensive comparison CSV - separated by framework
        if ultralytics_results:
            comparison_path = os.path.join(self.ultralytics_output_dir, "results_comprehensive.csv")
            with open(comparison_path, "w", newline="") as f:
                fieldnames = [
                    "Model", "Type", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]",
                    "Precision", "Recall", "Head_AP", "Medium_AP", "Tail_AP",
                    "Num_Images", "Avg_Detections", "Inference_Time_ms", "Device"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in ultralytics_results:
                    def safe_format(value, default=0.0, fmt=".4f"):
                        if value is None:
                            return f"{default:{fmt}}"
                        return f"{value:{fmt}}"
                    
                    writer.writerow({
                        "Model": result["model_name"],
                        "Type": result["model_type"],
                        "mAP@0.5": safe_format(result.get('mAP@0.5'), fmt=".4f"),
                        "mAP@0.75": safe_format(result.get('mAP@0.75'), fmt=".4f"),
                        "mAP@[0.5:0.95]": safe_format(result.get('mAP@[0.5:0.95]'), fmt=".4f"),
                        "Precision": safe_format(result.get('precision'), fmt=".4f"),
                        "Recall": safe_format(result.get('recall'), fmt=".4f"),
                        "Head_AP": safe_format(result.get('head_classes_AP'), fmt=".4f"),
                        "Medium_AP": safe_format(result.get('medium_classes_AP'), fmt=".4f"),
                        "Tail_AP": safe_format(result.get('tail_classes_AP'), fmt=".4f"),
                        "Num_Images": result.get('num_images', 0),
                        "Avg_Detections": safe_format(result.get('avg_detections_per_image'), fmt=".2f"),
                        "Inference_Time_ms": safe_format(result.get('inference_time_ms'), fmt=".2f"),
                        "Device": result.get('device', 'N/A'),
                    })
            print(f"✓ Ultralytics comprehensive CSV saved to: {comparison_path}")
        
        if torchvision_results:
            comparison_path = os.path.join(self.torchvision_output_dir, "results_comprehensive.csv")
            with open(comparison_path, "w", newline="") as f:
                fieldnames = [
                    "Model", "Type", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]",
                    "Precision", "Recall", "Head_AP", "Medium_AP", "Tail_AP",
                    "Num_Images", "Avg_Detections", "Inference_Time_ms", "Device"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in torchvision_results:
                    def safe_format(value, default=0.0, fmt=".4f"):
                        if value is None:
                            return f"{default:{fmt}}"
                        return f"{value:{fmt}}"
                    
                    writer.writerow({
                        "Model": result["model_name"],
                        "Type": result["model_type"],
                        "mAP@0.5": safe_format(result.get('mAP@0.5'), fmt=".4f"),
                        "mAP@0.75": safe_format(result.get('mAP@0.75'), fmt=".4f"),
                        "mAP@[0.5:0.95]": safe_format(result.get('mAP@[0.5:0.95]'), fmt=".4f"),
                        "Precision": safe_format(result.get('precision'), fmt=".4f"),
                        "Recall": safe_format(result.get('recall'), fmt=".4f"),
                        "Head_AP": safe_format(result.get('head_classes_AP'), fmt=".4f"),
                        "Medium_AP": safe_format(result.get('medium_classes_AP'), fmt=".4f"),
                        "Tail_AP": safe_format(result.get('tail_classes_AP'), fmt=".4f"),
                        "Num_Images": result.get('num_images', 0),
                        "Avg_Detections": safe_format(result.get('avg_detections_per_image'), fmt=".2f"),
                        "Inference_Time_ms": safe_format(result.get('inference_time_ms'), fmt=".2f"),
                        "Device": result.get('device', 'N/A'),
                    })
            print(f"✓ TorchVision comprehensive CSV saved to: {comparison_path}")
        
        # 5. Print detailed summary table
        print(f"\n{'='*100}")
        print(f"BASELINE RESULTS SUMMARY - PRIMARY METRICS")
        print(f"{'='*100}")
        print(f"{'Model':<20} {'mAP@[0.5:0.95]':<15} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}")
        print(f"{'-'*100}")
        
        for result in self.results:
            print(f"{result['model_name']:<20} {result.get('mAP@[0.5:0.95]', 0):<15.4f} "
                  f"{result.get('mAP@0.5', 0):<12.4f} {result.get('precision', 0):<12.4f} "
                  f"{result.get('recall', 0):<12.4f}")
        
        # 6. Print class-wise performance summary
        print(f"\n{'='*100}")
        print(f"CLASS-WISE PERFORMANCE - CRITICAL FOR IMBALANCE RESEARCH")
        print(f"{'='*100}")
        print(f"{'Model':<20} {'Head_Classes_AP':<18} {'Medium_Classes_AP':<20} {'Tail_Classes_AP':<18}")
        print(f"{'-'*100}")
        
        for result in self.results:
            head_ap = result.get('head_classes_AP') or 0.0
            medium_ap = result.get('medium_classes_AP') or 0.0
            tail_ap = result.get('tail_classes_AP') or 0.0
            print(f"{result['model_name']:<20} {head_ap:<18.4f} "
                  f"{medium_ap:<20.4f} {tail_ap:<18.4f}")
        
        print(f"{'='*100}\n")


def main():
    """
    Main function to run baseline inference with comprehensive metrics
    """
    
    # Check datasets exist
    val_dir = "/home/almaankhan/data/coco/images/val2017"
    ann_file = "/home/almaankhan/data/coco/annotations/instances_val2017.json"
    
    if not Path(val_dir).exists():
        print(f"Error: Validation dataset not found at {val_dir}")
        return
    
    if not Path(ann_file).exists():
        print(f"Error: Annotation file not found at {ann_file}")
        return
    
    # Initialize runner with COCO annotation file
    runner = BaselineInferenceRunner(
        val_dir=val_dir,
        ann_file=ann_file,
        model_dir="/home/almaankhan/model/loss-functions/focal-loss",
        output_dir="runs/inference"
    )
    
    # Run all inferences
    runner.run_all_inferences()
    
    # Save results
    runner.save_results()


if __name__ == "__main__":
    import numpy as np
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("\nWarning: GPU not available, using CPU\n")
    
    main()
