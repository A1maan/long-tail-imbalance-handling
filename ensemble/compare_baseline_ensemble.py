#!/usr/bin/env python3
"""
Compare baseline and ensemble results with visualizations.

Generates:
  - Overall metrics comparison (bar chart)
  - Class-wise performance (head/medium/tail)
  - Model rankings
  - Metrics summary table (HTML)

Usage:
  python compare_baseline_ensemble.py
  python compare_baseline_ensemble.py --baseline /path/to/baseline/metrics_summary.json
  python compare_baseline_ensemble.py --ensemble /path/to/ensemble/metrics_summary.json
  python compare_baseline_ensemble.py --output results/comparisons
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


# Style defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_baseline_metrics(filepath):
    """Load baseline metrics summary."""
    with open(filepath) as f:
        data = json.load(f)
    
    baseline = {}
    for model_name, metrics in data.items():
        baseline[model_name] = {
            'AP': metrics['overall'].get('AP', 0),
            'AP@0.5': metrics['overall'].get('AP@0.5', 0),
            'AP@0.75': metrics['overall'].get('AP@0.75', 0),
            'head_AP': metrics.get('head_classes_AP', 0),
            'medium_AP': metrics.get('medium_classes_AP', 0),
            'tail_AP': metrics.get('tail_classes_AP', 0),
            'num_head': metrics.get('num_head', 0),
            'num_medium': metrics.get('num_medium', 0),
            'num_tail': metrics.get('num_tail', 0),
        }
    return baseline


def load_ensemble_metrics(filepath):
    """Load ensemble metrics summary."""
    with open(filepath) as f:
        data = json.load(f)
    
    overall = data.get('overall', {})
    freq = data.get('class_frequency', {})
    
    ensemble = {
        'Ensemble': {
            'AP': overall.get('mAP@[0.5:0.95]', 0),
            'AP@0.5': overall.get('mAP@0.5', 0),
            'AP@0.75': overall.get('mAP@0.75', 0),
            'head_AP': freq.get('head', {}).get('AP', 0),
            'medium_AP': freq.get('medium', {}).get('AP', 0),
            'tail_AP': freq.get('tail', {}).get('AP', 0),
            'num_head': freq.get('head', {}).get('count', 0),
            'num_medium': freq.get('medium', {}).get('count', 0),
            'num_tail': freq.get('tail', {}).get('count', 0),
        }
    }
    return ensemble


def create_overall_comparison(baseline, ensemble, output_dir):
    """Create overall metrics comparison chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Baseline vs Ensemble: Overall Metrics', fontsize=14, fontweight='bold')
    
    metrics = ['AP', 'AP@0.5', 'AP@0.75']
    colors_baseline = sns.color_palette("Blues", n_colors=len(baseline))
    color_ensemble = '#E74C3C'  # Red for ensemble
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Baseline models
        baseline_values = [baseline[model][metric] for model in sorted(baseline.keys())]
        baseline_names = sorted(baseline.keys())
        x_pos = np.arange(len(baseline_names))
        
        bars1 = ax.bar(x_pos - 0.2, baseline_values, 0.4, label='Baseline Models', color=colors_baseline, alpha=0.8)
        
        # Ensemble
        ensemble_value = ensemble['Ensemble'][metric]
        bars2 = ax.bar(len(baseline_names) - 0.2, ensemble_value, 0.4, label='Ensemble', color=color_ensemble, alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(list(x_pos) + [len(baseline_names)])
        ax.set_xticklabels(baseline_names + ['Ensemble'], rotation=45, ha='right')
        ax.set_ylim(0, max(baseline_values + [ensemble_value]) * 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "01_overall_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved overall metrics comparison to {output_path}")
    plt.close()


def create_class_frequency_comparison(baseline, ensemble, output_dir):
    """Create head/medium/tail class comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Class-wise Performance: Head / Medium / Tail', fontsize=14, fontweight='bold')
    
    class_types = ['head_AP', 'medium_AP', 'tail_AP']
    class_labels = ['Head Classes', 'Medium Classes', 'Tail Classes']
    colors_baseline = sns.color_palette("Greens", n_colors=len(baseline))
    color_ensemble = '#E74C3C'
    
    for idx, (class_type, class_label) in enumerate(zip(class_types, class_labels)):
        ax = axes[idx]
        
        baseline_values = [baseline[model][class_type] for model in sorted(baseline.keys())]
        baseline_names = sorted(baseline.keys())
        x_pos = np.arange(len(baseline_names))
        
        bars1 = ax.bar(x_pos - 0.2, baseline_values, 0.4, label='Baseline Models', color=colors_baseline, alpha=0.8)
        ensemble_value = ensemble['Ensemble'][class_type]
        bars2 = ax.bar(len(baseline_names) - 0.2, ensemble_value, 0.4, label='Ensemble', color=color_ensemble, alpha=0.8)
        
        ax.set_ylabel('AP Score', fontweight='bold')
        ax.set_title(class_label, fontsize=12, fontweight='bold')
        ax.set_xticks(list(x_pos) + [len(baseline_names)])
        ax.set_xticklabels(baseline_names + ['Ensemble'], rotation=45, ha='right')
        ax.set_ylim(0, max(baseline_values + [ensemble_value]) * 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "02_class_frequency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class frequency comparison to {output_path}")
    plt.close()


def create_model_rankings(baseline, ensemble, output_dir):
    """Create model rankings for each metric."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Model Rankings by Metric', fontsize=14, fontweight='bold')
    
    metrics = ['AP', 'AP@0.5', 'AP@0.75', 'head_AP', 'medium_AP', 'tail_AP']
    metric_labels = ['mAP@[0.5:0.95]', 'mAP@0.5', 'mAP@0.75', 'Head AP', 'Medium AP', 'Tail AP']
    
    all_models = {**baseline, **ensemble}
    colors_baseline = sns.color_palette("Blues", n_colors=len(baseline))
    color_ensemble = '#E74C3C'
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        
        scores = [(model, data[metric]) for model, data in all_models.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        names, values = zip(*scores)
        colors = [color_ensemble if name == 'Ensemble' else colors_baseline[i % len(colors_baseline)]
                 for i, name in enumerate(names)]
        
        bars = ax.barh(names, values, color=colors, alpha=0.8)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, bar.get_y() + bar.get_height()/2.,
                   f' {value:.4f}',
                   va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "03_model_rankings.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved model rankings to {output_path}")
    plt.close()


def create_metrics_table(baseline, ensemble, output_dir):
    """Create an HTML metrics comparison table."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Baseline vs Ensemble Comparison</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
        "    h1 { color: #333; text-align: center; }",
        "    table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }",
        "    th { background-color: #34495e; color: white; padding: 12px; text-align: left; font-weight: bold; }",
        "    td { padding: 10px; border-bottom: 1px solid #ddd; }",
        "    tr:hover { background-color: #f0f0f0; }",
        "    .ensemble { background-color: #ffe6e6; font-weight: bold; }",
        "    .best { background-color: #e8f8e8; }",
        "    .worst { background-color: #ffe8e8; }",
        "    .metric-section { margin-top: 30px; }",
        "    .timestamp { text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Baseline vs Ensemble Comparison Report</h1>",
        f"  <p style='text-align: center; color: #666;'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    
    # Overall metrics table
    html_lines.extend([
        "  <div class='metric-section'>",
        "    <h2>Overall Metrics (mAP)</h2>",
        "    <table>",
        "      <tr>",
        "        <th>Model</th>",
        "        <th>mAP@[0.5:0.95]</th>",
        "        <th>mAP@0.5</th>",
        "        <th>mAP@0.75</th>",
        "      </tr>",
    ])
    
    all_models = sorted(baseline.keys()) + ['Ensemble']
    for model in all_models:
        if model == 'Ensemble':
            data = ensemble['Ensemble']
            row_class = 'ensemble'
        else:
            data = baseline[model]
            row_class = ''
        
        html_lines.append(f"      <tr class='{row_class}'>")
        html_lines.append(f"        <td><b>{model}</b></td>")
        html_lines.append(f"        <td>{data['AP']:.4f}</td>")
        html_lines.append(f"        <td>{data['AP@0.5']:.4f}</td>")
        html_lines.append(f"        <td>{data['AP@0.75']:.4f}</td>")
        html_lines.append(f"      </tr>")
    
    html_lines.extend([
        "    </table>",
        "  </div>",
    ])
    
    # Class-wise metrics table
    html_lines.extend([
        "  <div class='metric-section'>",
        "    <h2>Class-wise Performance (AP)</h2>",
        "    <table>",
        "      <tr>",
        "        <th>Model</th>",
        "        <th>Head AP</th>",
        "        <th>Medium AP</th>",
        "        <th>Tail AP</th>",
        "      </tr>",
    ])
    
    for model in all_models:
        if model == 'Ensemble':
            data = ensemble['Ensemble']
            row_class = 'ensemble'
        else:
            data = baseline[model]
            row_class = ''
        
        html_lines.append(f"      <tr class='{row_class}'>")
        html_lines.append(f"        <td><b>{model}</b></td>")
        html_lines.append(f"        <td>{data['head_AP']:.4f}</td>")
        html_lines.append(f"        <td>{data['medium_AP']:.4f}</td>")
        html_lines.append(f"        <td>{data['tail_AP']:.4f}</td>")
        html_lines.append(f"      </tr>")
    
    html_lines.extend([
        "    </table>",
        "  </div>",
    ])
    
    # Statistics
    html_lines.extend([
        "  <div class='metric-section'>",
        "    <h2>Statistics</h2>",
        "    <table>",
        "      <tr>",
        "        <th>Metric</th>",
        "        <th>Best Baseline</th>",
        "        <th>Ensemble</th>",
        "        <th>Difference</th>",
        "      </tr>",
    ])
    
    for metric, label in zip(['AP', 'head_AP', 'medium_AP', 'tail_AP'], 
                            ['mAP@[0.5:0.95]', 'Head AP', 'Medium AP', 'Tail AP']):
        baseline_values = [baseline[m][metric] for m in baseline.keys()]
        best_baseline = max(baseline_values)
        ensemble_val = ensemble['Ensemble'][metric]
        diff = ensemble_val - best_baseline
        diff_pct = (diff / best_baseline * 100) if best_baseline > 0 else 0
        
        row_class = 'best' if diff > 0 else 'worst'
        html_lines.append(f"      <tr class='{row_class}'>")
        html_lines.append(f"        <td><b>{label}</b></td>")
        html_lines.append(f"        <td>{best_baseline:.4f}</td>")
        html_lines.append(f"        <td>{ensemble_val:.4f}</td>")
        html_lines.append(f"        <td>{diff:+.4f} ({diff_pct:+.2f}%)</td>")
        html_lines.append(f"      </tr>")
    
    html_lines.extend([
        "    </table>",
        "  </div>",
        "  <div class='timestamp'>",
        f"    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "  </div>",
        "</body>",
        "</html>",
    ])
    
    output_path = Path(output_dir) / "metrics_comparison.html"
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_lines))
    
    print(f"✓ Saved HTML metrics table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Get the project root directory (parent of ensemble directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    default_baseline = project_root / 'baseline' / 'runs' / 'baseline_inference' / 'metrics_summary.json'
    default_ensemble = script_dir / 'predictions' / 'ensemble_summary.json'
    default_output = script_dir / 'comparisons'
    
    parser.add_argument('--baseline', type=str, 
                       default=str(default_baseline),
                       help='Path to baseline metrics summary JSON')
    parser.add_argument('--ensemble', type=str,
                       default=str(default_ensemble),
                       help='Path to ensemble metrics summary JSON')
    parser.add_argument('--output', type=str, default=str(default_output),
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Validate files exist
    baseline_path = Path(args.baseline)
    ensemble_path = Path(args.ensemble)
    
    if not baseline_path.exists():
        print(f"✗ Baseline metrics file not found: {baseline_path}")
        sys.exit(1)
    
    if not ensemble_path.exists():
        print(f"✗ Ensemble metrics file not found: {ensemble_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    baseline = load_baseline_metrics(baseline_path)
    ensemble = load_ensemble_metrics(ensemble_path)
    
    print(f"✓ Loaded {len(baseline)} baseline models")
    print(f"✓ Loaded ensemble metrics")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_overall_comparison(baseline, ensemble, output_dir)
    create_class_frequency_comparison(baseline, ensemble, output_dir)
    create_model_rankings(baseline, ensemble, output_dir)
    create_metrics_table(baseline, ensemble, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
