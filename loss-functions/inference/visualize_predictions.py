#!/usr/bin/env python3
"""
Visualize model predictions and create comparison charts.
Generates:
1. Sample images with bounding boxes from each model
2. Metrics comparison charts (AP, AP50, AP75)
3. Head/Medium/Tail performance comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class PredictionVisualizer:
    def __init__(self, coco_data_path="/home/almaankhan/data/coco/annotations/instances_val2017.json",
                 predictions_dir="runs/inference/predictions",
                 images_dir="/home/almaankhan/data/coco/images/val2017"):
        self.coco_data_path = Path(coco_data_path)
        self.predictions_dir = Path(predictions_dir)
        self.images_dir = Path(images_dir)
        
        # Load COCO annotations
        print("Loading COCO annotations...")
        with open(self.coco_data_path) as f:
            self.coco_data = json.load(f)
        
        self.images_by_id = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Load all predictions
        print("Loading predictions...")
        self.predictions = {}
        self.load_predictions()
        
        # Create output directories - separate by framework
        self.output_dir = Path("runs/inference/visualizations")
        self.ultralytics_output_dir = self.output_dir / "ultralytics"
        self.torchvision_output_dir = self.output_dir / "torchvision"
        self.ultralytics_output_dir.mkdir(parents=True, exist_ok=True)
        self.torchvision_output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_predictions(self):
        """Load all prediction JSON files."""
        for pred_file in self.predictions_dir.glob("*_predictions.json"):
            model_name = pred_file.stem.replace("_predictions", "")
            print(f"  Loading {model_name}...")
            with open(pred_file) as f:
                self.predictions[model_name] = json.load(f)
    
    def visualize_sample_predictions(self, num_samples=5, conf_threshold=0.3):
        """Generate sample images with predictions from each model."""
        print(f"\nGenerating sample predictions for {num_samples} images...")
        
        # Get sample image IDs
        image_ids = list(self.images_by_id.keys())[:num_samples]
        
        for model_name, preds in self.predictions.items():
            print(f"  Visualizing {model_name}...")
            
            # Determine framework and output directory
            if any(name in model_name.lower() for name in ["yolo", "rtdetr"]):
                output_dir = self.ultralytics_output_dir
            else:
                output_dir = self.torchvision_output_dir
            
            # Create predictions lookup by image_id
            preds_by_image = {}
            for pred in preds:
                img_id = pred['image_id']
                if img_id not in preds_by_image:
                    preds_by_image[img_id] = []
                if pred['score'] >= conf_threshold:
                    preds_by_image[img_id].append(pred)
            
            # Visualize samples
            fig, axes = plt.subplots(num_samples, 1, figsize=(16, 5*num_samples))
            if num_samples == 1:
                axes = [axes]
            
            for idx, img_id in enumerate(image_ids):
                img_info = self.images_by_id[img_id]
                img_path = self.images_dir / img_info['file_name']
                
                if not img_path.exists():
                    print(f"    Warning: Image not found: {img_path}")
                    continue
                
                img = Image.open(img_path)
                ax = axes[idx]
                ax.imshow(img)
                
                # Draw predictions
                for pred in preds_by_image.get(img_id, []):
                    bbox = pred['bbox']  # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # Draw rectangle
                    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                            edgecolor='g', facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                    
                    # Draw label
                    cat_name = self.categories.get(pred['category_id'], 'Unknown')
                    score = pred['score']
                    label = f"{cat_name} ({score:.2f})"
                    ax.text(x, y-5, label, fontsize=8, color='green', 
                           bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_title(f"{img_info['file_name']} - {len(preds_by_image.get(img_id, []))} detections")
                ax.axis('off')
            
            plt.tight_layout()
            output_path = output_dir / f"samples_{model_name}.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {output_path}")
    
    def create_metrics_comparison_chart(self, metrics_summary):
        """Create comparison charts for all models, separated by framework."""
        print("\nCreating metrics comparison charts...")
        
        if not metrics_summary:
            print("  No metrics data available")
            return
        
        # Separate by framework
        ultralytics_models = {}
        torchvision_models = {}
        
        for model_name, metrics in metrics_summary.items():
            if any(name in model_name.lower() for name in ["yolo", "rtdetr"]):
                ultralytics_models[model_name] = metrics
            else:
                torchvision_models[model_name] = metrics
        
        # Create charts for each framework
        self._create_framework_comparison(ultralytics_models, "Ultralytics", self.ultralytics_output_dir)
        self._create_framework_comparison(torchvision_models, "TorchVision", self.torchvision_output_dir)
    
    def _create_framework_comparison(self, metrics_dict, framework_name, output_dir):
        """Create comparison charts for a specific framework."""
        if not metrics_dict:
            return
        
        print(f"  Creating {framework_name} charts...")
        models = list(metrics_dict.keys())
        
        # Extract metrics
        overall_ap = [metrics_dict[m].get('overall', {}).get('AP', 0) for m in models]
        overall_ap50 = [metrics_dict[m].get('overall', {}).get('AP@0.5', 0) for m in models]
        overall_ap75 = [metrics_dict[m].get('overall', {}).get('AP@0.75', 0) for m in models]
        
        head_ap = [metrics_dict[m].get('head_classes_AP', 0) for m in models]
        medium_ap = [metrics_dict[m].get('medium_classes_AP', 0) for m in models]
        tail_ap = [metrics_dict[m].get('tail_classes_AP', 0) for m in models]
        
        # 1. Overall metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        metrics_names = ['AP', 'AP@0.5', 'AP@0.75']
        metrics_values = [overall_ap, overall_ap50, overall_ap75]
        
        for ax, metric_name, values in zip(axes, metrics_names, metrics_values):
            bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'{framework_name} - Overall {metric_name}', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 0.6)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"{framework_name.lower()}_metrics_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")
        
        # 2. Head/Medium/Tail comparison
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, head_ap, width, label='Head Classes', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, medium_ap, width, label='Medium Classes', color='#3498db', alpha=0.8)
        bars3 = ax.bar(x + width, tail_ap, width, label='Tail Classes', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('AP', fontsize=12, fontweight='bold')
        ax.set_title(f'{framework_name} - Performance by Class Frequency', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / f"{framework_name.lower()}_head_medium_tail_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")
        
        # 3. Heatmap of all metrics
        self._create_framework_heatmap(models, metrics_dict, framework_name, output_dir)
    
    def _create_framework_heatmap(self, models, metrics_dict, framework_name, output_dir):
        """Create a heatmap of all metrics for a specific framework."""
        print(f"    Creating {framework_name} metrics heatmap...")
        
        # Collect all metrics
        metrics_data = []
        metric_names = []
        
        for model in models:
            model_metrics = metrics_dict[model]
            row = []
            
            # Overall metrics
            row.append(model_metrics.get('overall', {}).get('AP', 0))
            row.append(model_metrics.get('overall', {}).get('AP@0.5', 0))
            row.append(model_metrics.get('overall', {}).get('AP@0.75', 0))
            
            # Head/Medium/Tail
            row.append(model_metrics.get('head_classes_AP', 0))
            row.append(model_metrics.get('medium_classes_AP', 0))
            row.append(model_metrics.get('tail_classes_AP', 0))
            
            metrics_data.append(row)
        
        metric_names = ['Overall AP', 'Overall AP@0.5', 'Overall AP@0.75',
                       'Head AP', 'Medium AP', 'Tail AP']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.6)
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{metrics_data[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='AP Score')
        plt.title(f'{framework_name} Performance Heatmap', fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_path = output_dir / f"{framework_name.lower()}_metrics_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")
    
    def create_metrics_heatmap(self, models, metrics_summary):
        """Create a heatmap of all metrics. (Deprecated - use _create_framework_heatmap)"""
        print("  Creating metrics heatmap...")
        
        # Collect all metrics
        metrics_data = []
        metric_names = []
        
        for model in models:
            model_metrics = metrics_summary[model]
            row = []
            
            # Overall metrics
            row.append(model_metrics.get('overall', {}).get('AP', 0))
            row.append(model_metrics.get('overall', {}).get('AP@0.5', 0))
            row.append(model_metrics.get('overall', {}).get('AP@0.75', 0))
            
            # Head/Medium/Tail
            row.append(model_metrics.get('head_classes_AP', 0))
            row.append(model_metrics.get('medium_classes_AP', 0))
            row.append(model_metrics.get('tail_classes_AP', 0))
            
            metrics_data.append(row)
        
        metric_names = ['Overall AP', 'Overall AP@0.5', 'Overall AP@0.75',
                       'Head AP', 'Medium AP', 'Tail AP']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.6)
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{metrics_data[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='AP Score')
        plt.title('Model Performance Heatmap', fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / "metrics_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def create_summary_table(self, metrics_summary):
        """Create and save summary tables, separated by framework."""
        print("\nCreating summary tables...")
        
        if not metrics_summary:
            print("  No metrics data available")
            return
        
        # Separate by framework
        ultralytics_models = {}
        torchvision_models = {}
        
        for model_name, metrics in metrics_summary.items():
            if any(name in model_name.lower() for name in ["yolo", "rtdetr"]):
                ultralytics_models[model_name] = metrics
            else:
                torchvision_models[model_name] = metrics
        
        # Create tables for each framework
        self._create_framework_summary_table(ultralytics_models, "Ultralytics", self.ultralytics_output_dir)
        self._create_framework_summary_table(torchvision_models, "TorchVision", self.torchvision_output_dir)
    
    def _create_framework_summary_table(self, metrics_dict, framework_name, output_dir):
        """Create and save a summary table for a specific framework."""
        if not metrics_dict:
            return
        
        print(f"  Creating {framework_name} summary table...")
        
        # Create table data
        rows = []
        for model, metrics in metrics_dict.items():
            rows.append({
                'Model': model,
                'Overall AP': f"{metrics.get('overall', {}).get('AP', 0):.4f}",
                'AP@0.5': f"{metrics.get('overall', {}).get('AP@0.5', 0):.4f}",
                'AP@0.75': f"{metrics.get('overall', {}).get('AP@0.75', 0):.4f}",
                'Head AP': f"{metrics.get('head_classes_AP', 0):.4f}",
                'Medium AP': f"{metrics.get('medium_classes_AP', 0):.4f}",
                'Tail AP': f"{metrics.get('tail_classes_AP', 0):.4f}",
                '# Head': metrics.get('num_head', 0),
                '# Medium': metrics.get('num_medium', 0),
                '# Tail': metrics.get('num_tail', 0),
            })
        
        # Save to JSON
        output_path = output_dir / "metrics_summary.json"
        with open(output_path, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"    Saved: {output_path}")
        
        # Print table
        print(f"\n{'='*130}")
        print(f"{framework_name} METRICS SUMMARY")
        print(f"{'='*130}")
        print(f"{'Model':<20} {'Overall AP':<12} {'AP@0.5':<12} {'AP@0.75':<12} {'Head AP':<12} {'Medium AP':<12} {'Tail AP':<12}")
        print(f"{'-'*130}")
        for row in rows:
            print(f"{row['Model']:<20} {row['Overall AP']:<12} {row['AP@0.5']:<12} {row['AP@0.75']:<12} {row['Head AP']:<12} {row['Medium AP']:<12} {row['Tail AP']:<12}")
        print(f"{'='*130}\n")


def run_calculate_class_metrics():
    """Run calculate_class_metrics.py to get metrics summary."""
    print("Calculating class metrics...")
    import subprocess
    result = subprocess.run(['python', 'calculate_class_metrics.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Class metrics calculated successfully")
        # Try to load the metrics summary from both framework directories
        ultralytics_file = Path("runs/inference/ultralytics/metrics_summary.json")
        torchvision_file = Path("runs/inference/torchvision/metrics_summary.json")
        
        combined_metrics = {}
        if ultralytics_file.exists():
            with open(ultralytics_file) as f:
                ultralytics_data = json.load(f)
                combined_metrics.update(ultralytics_data)
        if torchvision_file.exists():
            with open(torchvision_file) as f:
                torchvision_data = json.load(f)
                combined_metrics.update(torchvision_data)
        
        return combined_metrics if combined_metrics else {}
    else:
        print(f"✗ Error calculating class metrics:\n{result.stderr}")
    
    return {}


if __name__ == "__main__":
    # Calculate metrics
    metrics_summary = run_calculate_class_metrics()
    
    # Create visualizer
    visualizer = PredictionVisualizer()
    
    # Generate visualizations
    visualizer.visualize_sample_predictions(num_samples=5, conf_threshold=0.3)
    visualizer.create_metrics_comparison_chart(metrics_summary)
    visualizer.create_summary_table(metrics_summary)
    
    print("\n✓ All visualizations completed!")
    print(f"  Output directory: {visualizer.output_dir}")
