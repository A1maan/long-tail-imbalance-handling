#!/usr/bin/env python3
"""
Filter ensemble predictions by confidence threshold and benchmark performance.

Tests multiple confidence thresholds to find the optimal one.

Usage:
  python filter_and_benchmark_ensemble.py --thresholds 0.2 0.3 0.4 0.5 0.6 0.7
"""

import argparse
import json
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def filter_predictions_by_threshold(data, threshold):
    """Filter detections by confidence score threshold."""
    filtered_data = []
    total_before = 0
    total_after = 0
    
    for image_pred in data:
        total_before += len(image_pred.get('detections', []))
        
        filtered_dets = [
            det for det in image_pred.get('detections', [])
            if det.get('score', 0) >= threshold
        ]
        
        total_after += len(filtered_dets)
        
        filtered_data.append({
            'image_id': image_pred['image_id'],
            'image_name': image_pred['image_name'],
            'width': image_pred.get('width'),
            'height': image_pred.get('height'),
            'detections': filtered_dets
        })
    
    return filtered_data, total_before, total_after


def save_predictions(data, output_path):
    """Save predictions to JSON."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_metrics_for_threshold(predictions_file):
    """Run calculate_ensemble_metrics.py and return metrics."""
    # Backup original
    backup_path = Path('predictions/predictions.json.bak')
    predictions_path = Path('predictions/predictions.json')
    
    if not backup_path.exists():
        shutil.copy2(predictions_path, backup_path)
    
    # Copy filtered predictions
    shutil.copy2(predictions_file, predictions_path)
    
    # Run metrics calculation
    result = subprocess.run(
        ['python3', 'calculate_ensemble_metrics.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error calculating metrics: {result.stderr}")
        return None
    
    # Load results
    with open('predictions/ensemble_summary.json') as f:
        summary = json.load(f)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, default='predictions/predictions.json',
                       help='Input ensemble predictions JSON')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       help='Confidence thresholds to test')
    parser.add_argument('--output-dir', type=str, default='threshold_analysis',
                       help='Output directory for results')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original predictions
    print(f"Loading predictions from {input_path}...")
    with open(input_path) as f:
        original_data = json.load(f)
    
    print(f"✓ Loaded {len(original_data)} images\n")
    
    # Test each threshold
    results = []
    print("=" * 95)
    print("TESTING CONFIDENCE THRESHOLDS")
    print("=" * 95)
    
    for threshold in sorted(args.thresholds):
        print(f"\n[Threshold: {threshold:.2f}]", flush=True)
        
        # Filter
        filtered_data, total_before, total_after = filter_predictions_by_threshold(
            original_data, threshold
        )
        
        print(f"  Detections: {total_before} → {total_after} ({total_after/total_before*100:.1f}%)")
        
        # Save
        filtered_path = output_dir / f"predictions_threshold_{threshold:.2f}.json"
        save_predictions(filtered_data, filtered_path)
        
        # Calculate metrics
        print(f"  Calculating metrics...", flush=True)
        summary = calculate_metrics_for_threshold(str(filtered_path))
        
        if summary:
            overall = summary['overall']
            freq = summary['class_frequency']
            
            result_entry = {
                'threshold': threshold,
                'num_detections': total_after,
                'mAP': overall['mAP@[0.5:0.95]'],
                'mAP@0.5': overall['mAP@0.5'],
                'mAP@0.75': overall['mAP@0.75'],
                'head_AP': freq['head']['AP'],
                'medium_AP': freq['medium']['AP'],
                'tail_AP': freq['tail']['AP'],
            }
            results.append(result_entry)
            
            print(f"  mAP@[0.5:0.95]: {overall['mAP@[0.5:0.95]']:.6f}")
            print(f"  mAP@0.5:        {overall['mAP@0.5']:.6f}")
            print(f"  Head/Medium/Tail AP: {freq['head']['AP']:.4f} / {freq['medium']['AP']:.4f} / {freq['tail']['AP']:.4f}")
    
    # Summary
    print("\n" + "=" * 95)
    print("THRESHOLD COMPARISON")
    print("=" * 95)
    
    if results:
        best_result = max(results, key=lambda x: x['mAP'])
        best_threshold = best_result['threshold']
        
        print(f"\n{'Threshold':<12} {'Detections':<15} {'mAP':<15} {'mAP@0.5':<15} {'Head AP':<15}")
        print("-" * 95)
        
        for result in results:
            marker = " ← BEST" if result['threshold'] == best_threshold else ""
            print(f"{result['threshold']:<12.2f} {result['num_detections']:<15} "
                  f"{result['mAP']:<15.6f} {result['mAP@0.5']:<15.6f} {result['head_AP']:<15.6f}{marker}")
        
        print(f"\n{'='*95}")
        print(f"OPTIMAL THRESHOLD: {best_threshold:.2f}")
        print(f"Best mAP: {best_result['mAP']:.6f}")
        print(f"{'='*95}")
        
        # Save results
        results_path = output_dir / "threshold_comparison.json"
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'optimal_threshold': best_threshold,
                'best_map': best_result['mAP'],
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        # Restore and create optimized
        print(f"\nRestoring original predictions...")
        backup_path = Path('predictions/predictions.json.bak')
        if backup_path.exists():
            shutil.copy2(backup_path, 'predictions/predictions.json')
        
        print(f"Creating optimized predictions with threshold {best_threshold:.2f}...")
        optimized_data, _, _ = filter_predictions_by_threshold(original_data, best_threshold)
        optimized_path = output_dir / "predictions_optimized.json"
        save_predictions(optimized_data, optimized_path)
        
        print(f"✓ Optimized predictions saved to {optimized_path}")
        print(f"\nTo use optimized ensemble:")
        print(f"  cp {optimized_path} predictions/predictions.json")
        print(f"  python calculate_ensemble_metrics.py")


if __name__ == '__main__':
    main()
