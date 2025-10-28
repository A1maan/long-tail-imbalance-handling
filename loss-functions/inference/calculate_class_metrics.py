#!/usr/bin/env python3
"""
Calculate per-class metrics from prediction JSONs.
Loads predictions saved by baseline_inference.py and calculates metrics for each class.
"""

import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ClassMetricsCalculator:
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Long-tail grouping - MUST match baseline_inference.py
    HEAD_CLASSES = [
        'person', 'car', 'dog', 'cat', 'bicycle', 'boat', 'bird', 'chair', 
        'cow', 'horse', 'sheep', 'traffic light', 'stop sign', 'bench'
    ]
    
    TAIL_CLASSES = [
        'toothbrush', 'hair drier', 'scissors', 'vase', 'clock', 'book',
        'remote', 'keyboard', 'mouse', 'laptop', 'oven', 'toaster', 'sink'
    ]
    
    # MEDIUM_CLASSES is computed as: all classes - HEAD - TAIL
    # (No explicit list needed, calculated in aggregate_by_frequency)
    
    def __init__(self, coco_annotation_file="/home/almaankhan/data/coco/annotations/instances_val2017.json"):
        """Initialize with COCO annotations."""
        self.coco_annotation_file = Path(coco_annotation_file)
        
        # Load annotations
        print("Loading COCO annotations...")
        self.coco_gt = COCO(str(self.coco_annotation_file))
        
        # Load and create category mappings
        with open(self.coco_annotation_file) as f:
            coco_data = json.load(f)
        
        cats = sorted(coco_data['categories'], key=lambda x: x['id'])
        
        # Create three mappings:
        # 1. YOLO index (0-79) → COCO category ID
        self.yolo_to_coco_id = {idx: cat['id'] for idx, cat in enumerate(cats)}
        
        # 2. COCO category ID → class name
        self.coco_id_to_name = {cat['id']: cat['name'] for cat in cats}
        
        # 3. Class name → COCO category ID (for HEAD/TAIL lookup)
        self.coco_name_to_id = {cat['name']: cat['id'] for cat in cats}
        
        print(f"Loaded {len(cats)} categories from COCO annotations")
    
    def calculate_per_class_metrics(self, predictions_json):
        """
        Calculate metrics for each class.
        
        Args:
            predictions_json: List of predictions in COCO format
            
        Returns:
            Dictionary mapping category_id to metrics
        """
        # Load predictions as COCO results
        coco_dt = self.coco_gt.loadRes(predictions_json)
        
        class_metrics = {}
        valid_count = 0
        
        for idx, class_name in enumerate(self.COCO_CLASSES):
            try:
                coco_cat_id = self.yolo_to_coco_id[idx]
                
                # Create evaluator for this class
                coco_eval_class = COCOeval(self.coco_gt, coco_dt, "bbox")
                coco_eval_class.params.catIds = [coco_cat_id]
                coco_eval_class.evaluate()
                coco_eval_class.accumulate()
                coco_eval_class.summarize()
                
                # Check if stats are valid (not NaN)
                if len(coco_eval_class.stats) > 0 and not np.isnan(coco_eval_class.stats[0]):
                    # Store by BOTH category_id and class_name for flexibility
                    class_metrics[coco_cat_id] = {
                        "class_name": class_name,
                        "AP": float(coco_eval_class.stats[0]),
                        "AP@0.5": float(coco_eval_class.stats[1]),
                        "AP@0.75": float(coco_eval_class.stats[2]),
                    }
                    valid_count += 1
            except Exception as e:
                pass  # Skip classes with errors
        
        print(f"  Calculated metrics for {valid_count}/{len(self.COCO_CLASSES)} classes")
        return class_metrics
    
    def aggregate_by_frequency(self, class_metrics):
        """
        Aggregate metrics into head/medium/tail groups
        
        Args:
            class_metrics: Dictionary of per-class metrics keyed by category_id
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not class_metrics:
            return {
                "head_classes_AP": None,
                "medium_classes_AP": None,
                "tail_classes_AP": None,
                "num_head": 0,
                "num_medium": 0,
                "num_tail": 0,
            }
        
        # Convert class names to category IDs for lookup
        head_cat_ids = [self.coco_name_to_id[name] for name in self.HEAD_CLASSES if name in self.coco_name_to_id]
        tail_cat_ids = [self.coco_name_to_id[name] for name in self.TAIL_CLASSES if name in self.coco_name_to_id]
        
        # Head classes
        head_aps = [class_metrics[cat_id]["AP"] for cat_id in head_cat_ids if cat_id in class_metrics]
        head_ap = float(np.mean(head_aps)) if head_aps else 0.0
        
        # Tail classes
        tail_aps = [class_metrics[cat_id]["AP"] for cat_id in tail_cat_ids if cat_id in class_metrics]
        tail_ap = float(np.mean(tail_aps)) if tail_aps else 0.0
        
        # Medium classes (everything else - computed dynamically)
        all_cat_ids = set(class_metrics.keys())
        medium_cat_ids_set = all_cat_ids - set(head_cat_ids) - set(tail_cat_ids)
        medium_aps = [class_metrics[cat_id]["AP"] for cat_id in medium_cat_ids_set]
        medium_ap = float(np.mean(medium_aps)) if medium_aps else 0.0
        
        return {
            "head_classes_AP": head_ap,
            "medium_classes_AP": medium_ap,
            "tail_classes_AP": tail_ap,
            "num_head": len(head_aps),
            "num_medium": len(medium_aps),
            "num_tail": len(tail_aps),
        }
    
    def process_all_predictions(self, predictions_dir="runs/inference/predictions"):
        """
        Process all prediction JSONs in a directory.
        Automatically separates YOLO-based (ultralytics) from torchvision predictions.
        
        Args:
            predictions_dir: Directory containing prediction JSON files
            
        Returns:
            Dictionary mapping model name to class-wise metrics
        """
        predictions_dir = Path(predictions_dir)
        if not predictions_dir.exists():
            print(f"Error: Predictions directory not found: {predictions_dir}")
            return {}
        
        results = {}
        json_files = list(predictions_dir.glob("*_predictions.json"))
        
        if not json_files:
            print(f"No prediction files found in {predictions_dir}")
            return {}
        
        print(f"\nProcessing {len(json_files)} prediction files...")
        
        for pred_file in sorted(json_files):
            model_name = pred_file.stem.replace("_predictions", "")
            print(f"\n{model_name}:")
            
            # Load predictions
            with open(pred_file) as f:
                predictions = json.load(f)
            
            # Calculate metrics
            class_metrics = self.calculate_per_class_metrics(predictions)
            
            # Calculate overall metrics
            coco_dt = self.coco_gt.loadRes(predictions)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            overall_metrics = {
                "AP": float(coco_eval.stats[0]) if len(coco_eval.stats) > 0 and not np.isnan(coco_eval.stats[0]) else 0,
                "AP@0.5": float(coco_eval.stats[1]) if len(coco_eval.stats) > 1 and not np.isnan(coco_eval.stats[1]) else 0,
                "AP@0.75": float(coco_eval.stats[2]) if len(coco_eval.stats) > 2 and not np.isnan(coco_eval.stats[2]) else 0,
            }
            
            # Aggregate by frequency
            frequency_agg = self.aggregate_by_frequency(class_metrics)
            
            # Combine results
            results[model_name] = {
                "overall": overall_metrics,
                **frequency_agg,
                "per_class": class_metrics
            }
            
            # Safe formatting for None values
            head_ap = frequency_agg['head_classes_AP'] if frequency_agg['head_classes_AP'] is not None else 0
            medium_ap = frequency_agg['medium_classes_AP'] if frequency_agg['medium_classes_AP'] is not None else 0
            tail_ap = frequency_agg['tail_classes_AP'] if frequency_agg['tail_classes_AP'] is not None else 0
            
            print(f"  Overall AP: {overall_metrics['AP']:.4f}")
            print(f"  Head AP: {head_ap:.4f} ({frequency_agg['num_head']} classes)")
            print(f"  Medium AP: {medium_ap:.4f} ({frequency_agg['num_medium']} classes)")
            print(f"  Tail AP: {tail_ap:.4f} ({frequency_agg['num_tail']} classes)")
        
        return results
    
    def save_summary(self, results, output_dir="runs/inference"):
        """
        Save results summary to JSON, separated by framework.
        
        Args:
            results: Dictionary of model_name -> metrics
            output_dir: Base output directory (will separate into ultralytics/ and torchvision/)
        """
        output_dir = Path(output_dir)
        
        # Separate results by framework
        ultralytics_results = {}
        torchvision_results = {}
        
        for model_name, metrics in results.items():
            summary_item = {
                "overall": metrics["overall"],
                "head_classes_AP": metrics["head_classes_AP"],
                "medium_classes_AP": metrics["medium_classes_AP"],
                "tail_classes_AP": metrics["tail_classes_AP"],
                "num_head": metrics["num_head"],
                "num_medium": metrics["num_medium"],
                "num_tail": metrics["num_tail"],
            }
            
            # Detect framework based on model name
            # Note: RT-DETR model name has hyphen: "RT-DETR-L"
            model_lower = model_name.lower().replace("-", "")
            if any(name in model_lower for name in ["yolo", "rtdetr"]):
                ultralytics_results[model_name] = summary_item
            else:
                torchvision_results[model_name] = summary_item
        
        # Save to framework-specific directories
        if ultralytics_results:
            ultralytics_dir = output_dir / "ultralytics"
            ultralytics_dir.mkdir(parents=True, exist_ok=True)
            output_path = ultralytics_dir / "metrics_summary.json"
            with open(output_path, 'w') as f:
                json.dump(ultralytics_results, f, indent=2)
            print(f"✓ Ultralytics summary saved to {output_path}")
        
        if torchvision_results:
            torchvision_dir = output_dir / "torchvision"
            torchvision_dir.mkdir(parents=True, exist_ok=True)
            output_path = torchvision_dir / "metrics_summary.json"
            with open(output_path, 'w') as f:
                json.dump(torchvision_results, f, indent=2)
            print(f"✓ TorchVision summary saved to {output_path}")
        
        return {"ultralytics": ultralytics_results, "torchvision": torchvision_results}


def main():
    """Main entry point."""
    calculator = ClassMetricsCalculator()
    # Use the same predictions directory as baseline_inference.py
    results = calculator.process_all_predictions(predictions_dir="runs/inference/predictions")
    # Save to framework-specific directories
    summary = calculator.save_summary(results, output_dir="runs/inference")


if __name__ == "__main__":
    main()
