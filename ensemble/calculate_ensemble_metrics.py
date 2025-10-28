#!/usr/bin/env python3
"""
Calculate metrics for ensemble predictions.
Loads ensemble predictions JSON and calculates comprehensive metrics including
per-class performance and head/medium/tail aggregation.
"""

import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class EnsembleMetricsCalculator:
    """Calculate metrics for ensemble predictions."""
    
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
    
    def __init__(self, coco_annotation_file="/home/almaankhan/data/coco/annotations/instances_val2017.json",
                 ensemble_predictions_json="predictions/predictions.json"):
        """
        Initialize metrics calculator.
        
        Args:
            coco_annotation_file: Path to COCO annotations JSON
            ensemble_predictions_json: Path to ensemble predictions JSON
        """
        self.coco_annotation_file = Path(coco_annotation_file)
        self.ensemble_predictions_json = Path(ensemble_predictions_json)
        
        # Load COCO annotations
        print("Loading COCO annotations...")
        self.coco_gt = COCO(str(self.coco_annotation_file))
        
        # Load and create category mappings
        with open(self.coco_annotation_file) as f:
            coco_data = json.load(f)
        
        cats = sorted(coco_data['categories'], key=lambda x: x['id'])
        
        # Create mappings:
        # 1. YOLO index (0-79) → COCO category ID
        self.yolo_to_coco_id = {idx: cat['id'] for idx, cat in enumerate(cats)}
        
        # 2. COCO category ID → class name
        self.coco_id_to_name = {cat['id']: cat['name'] for cat in cats}
        
        # 3. Class name → COCO category ID (for HEAD/TAIL lookup)
        self.coco_name_to_id = {cat['name']: cat['id'] for cat in cats}
        
        print(f"✓ Loaded {len(cats)} categories from COCO annotations")
    
    def convert_ensemble_to_coco_format(self, ensemble_predictions):
        """
        Convert ensemble predictions to COCO format.
        
        Ensemble format: [{"image_id", "image_name", "width", "height", "detections": [{"bbox": [x,y,w,h], "score", "category_id"}]}, ...]
        COCO format: [{"image_id", "category_id", "bbox": [x,y,w,h], "score"}, ...]
        
        NOTE: If ensemble predictions use YOLO indices (0-79) instead of COCO category IDs (1-90),
              they will be converted using self.yolo_to_coco_id mapping.
        
        Args:
            ensemble_predictions: List of ensemble predictions
            
        Returns:
            List of predictions in COCO format
        """
        coco_format = []
        
        for image_pred in ensemble_predictions:
            image_id = image_pred['image_id']
            
            for det in image_pred['detections']:
                bbox = det['bbox']  # [x, y, w, h] in COCO format
                category_id = int(det['category_id'])
                
                # Check if category_id needs mapping from YOLO index to COCO ID
                # YOLO indices are 0-79, COCO category IDs start at 1
                if category_id in self.yolo_to_coco_id:
                    # Assume it's a YOLO index, map it to COCO category ID
                    category_id = self.yolo_to_coco_id[category_id]
                
                coco_format.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "score": float(det['score'])
                })
        
        print(f"✓ Converted {len(coco_format)} detections to COCO format")
        return coco_format
    
    def calculate_metrics(self):
        """
        Calculate comprehensive metrics for ensemble predictions.
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\nLoading ensemble predictions from {self.ensemble_predictions_json}...")
        
        # Load ensemble predictions
        if not self.ensemble_predictions_json.exists():
            print(f"✗ Ensemble predictions file not found: {self.ensemble_predictions_json}")
            return None
        
        with open(self.ensemble_predictions_json) as f:
            ensemble_predictions = json.load(f)
        
        print(f"✓ Loaded {len(ensemble_predictions)} image predictions")
        
        # Convert to COCO format
        coco_predictions = self.convert_ensemble_to_coco_format(ensemble_predictions)
        
        # Initialize COCO evaluator
        print("\nEvaluating ensemble predictions...")
        coco_dt = self.coco_gt.loadRes(coco_predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract primary metrics
        stats = coco_eval.stats
        metrics = {
            "model_name": "Ensemble",
            "mAP@[0.5:0.95]": float(stats[0]),
            "mAP@0.5": float(stats[1]),
            "mAP@0.75": float(stats[2]),
            "precision": float(stats[3]),
            "recall": float(stats[8]),
        }
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE METRICS")
        print(f"{'='*60}")
        print(f"mAP@[0.5:0.95]: {metrics['mAP@[0.5:0.95]']:.4f}")
        print(f"mAP@0.5:       {metrics['mAP@0.5']:.4f}")
        print(f"mAP@0.75:      {metrics['mAP@0.75']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        
        # Calculate per-class metrics
        print("\nCalculating per-class metrics...")
        class_metrics = self.calculate_per_class_metrics(coco_predictions)
        
        # Aggregate by frequency
        print("\nAggregating by class frequency...")
        frequency_agg = self.aggregate_by_frequency(class_metrics)
        
        print(f"\n{'='*60}")
        print(f"CLASS-WISE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Head Classes AP:   {frequency_agg['head_classes_AP']:.4f} ({frequency_agg['num_head']} classes)")
        print(f"Medium Classes AP: {frequency_agg['medium_classes_AP']:.4f} ({frequency_agg['num_medium']} classes)")
        print(f"Tail Classes AP:   {frequency_agg['tail_classes_AP']:.4f} ({frequency_agg['num_tail']} classes)")
        print(f"{'='*60}\n")
        
        # Combine all results
        results = {
            "primary_metrics": metrics,
            "class_wise": class_metrics,
            **frequency_agg
        }
        
        return results
    
    def calculate_per_class_metrics(self, coco_predictions):
        """
        Calculate metrics for each class.
        
        Args:
            coco_predictions: Predictions in COCO format
            
        Returns:
            Dictionary mapping category_id to metrics
        """
        coco_dt = self.coco_gt.loadRes(coco_predictions)
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
                    class_metrics[coco_cat_id] = {
                        "class_name": class_name,
                        "AP": float(coco_eval_class.stats[0]),
                        "AP@0.5": float(coco_eval_class.stats[1]),
                        "AP@0.75": float(coco_eval_class.stats[2]),
                    }
                    valid_count += 1
            except Exception as e:
                pass  # Skip classes with errors
        
        print(f"✓ Calculated metrics for {valid_count}/{len(self.COCO_CLASSES)} classes")
        return class_metrics
    
    def aggregate_by_frequency(self, class_metrics):
        """
        Aggregate metrics into head/medium/tail groups.
        
        Args:
            class_metrics: Dictionary of per-class metrics keyed by category_id
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not class_metrics:
            return {
                "head_classes_AP": 0.0,
                "medium_classes_AP": 0.0,
                "tail_classes_AP": 0.0,
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
    
    def save_results(self, results, output_dir="predictions"):
        """
        Save metrics results to JSON files.
        
        Args:
            results: Dictionary with calculated metrics
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save primary metrics
        primary_metrics = results['primary_metrics']
        primary_output = output_dir / "ensemble_metrics.json"
        with open(primary_output, 'w') as f:
            json.dump(primary_metrics, f, indent=2)
        print(f"✓ Saved primary metrics to {primary_output}")
        
        # 2. Save class-wise metrics
        class_output = output_dir / "ensemble_class_metrics.json"
        with open(class_output, 'w') as f:
            json.dump(results['class_wise'], f, indent=2)
        print(f"✓ Saved class-wise metrics to {class_output}")
        
        # 3. Save aggregated metrics
        agg_output = output_dir / "ensemble_aggregated_metrics.json"
        agg_metrics = {
            "head_classes_AP": results['head_classes_AP'],
            "medium_classes_AP": results['medium_classes_AP'],
            "tail_classes_AP": results['tail_classes_AP'],
            "num_head": results['num_head'],
            "num_medium": results['num_medium'],
            "num_tail": results['num_tail'],
        }
        with open(agg_output, 'w') as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"✓ Saved aggregated metrics to {agg_output}")
        
        # 4. Create summary comparison file
        summary_output = output_dir / "ensemble_summary.json"
        summary = {
            "model": "Ensemble (Weighted Boxes Fusion)",
            "overall": primary_metrics,
            "class_frequency": {
                "head": {"AP": results['head_classes_AP'], "count": results['num_head']},
                "medium": {"AP": results['medium_classes_AP'], "count": results['num_medium']},
                "tail": {"AP": results['tail_classes_AP'], "count": results['num_tail']},
            }
        }
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_output}")


def main():
    """Main entry point."""
    calculator = EnsembleMetricsCalculator(
        coco_annotation_file="/home/almaankhan/data/coco/annotations/instances_val2017.json",
        ensemble_predictions_json="predictions/predictions.json"
    )
    
    # Calculate metrics
    results = calculator.calculate_metrics()
    
    if results:
        # Save results
        calculator.save_results(results, output_dir="predictions")
        print("\n✓ Ensemble metrics calculation completed!")
    else:
        print("\n✗ Failed to calculate ensemble metrics")


if __name__ == "__main__":
    main()
