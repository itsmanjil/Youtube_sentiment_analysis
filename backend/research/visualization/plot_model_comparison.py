"""
Model Performance Comparison Visualization

Creates bar charts and comparison plots for:
- Accuracy comparison across all models
- F1-Macro comparison
- Per-class F1 comparison
- Comprehensive performance dashboard
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """Load all model results from JSON files."""
    results_dir = Path(results_dir)
    results = {}
    
    # Load baseline results
    for model in ['logreg', 'svm', 'tfidf']:
        result_file = results_dir / f'baseline_{model}.json'
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                results[model] = data.get(model, data)
    
    # Load hybrid model results
    hybrid_file = results_dir / '../output/thesis_full_gpu/test_results.json'
    if Path(hybrid_file).exists():
        with open(hybrid_file) as f:
            results['hybrid_dl'] = json.load(f)
    
    # Load ensemble results
    ensemble_file = results_dir / 'final_comparison.json'
    if ensemble_file.exists():
        with open(ensemble_file) as f:
            data = json.load(f)
            if 'models' in data and 'pso_ensemble' in data['models']:
                results['ensemble'] = data['models']['pso_ensemble']
    
    return results


def plot_metric_comparison(results, metric, title, output_path):
    """Create bar chart for a single metric."""
    models = list(results.keys())
    values = [results[m].get(metric, 0) * 100 for m in models]
    
    # Sort by value
    sorted_data = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_data)
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved {metric} comparison to: {output_path}')
    plt.close()


def generate_comparison_plots(results_dir, output_dir):
    """Generate all comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for: {', '.join(results.keys())}")
    
    # Individual metric plots
    plot_metric_comparison(results, 'accuracy', 
                          'Model Accuracy Comparison',
                          output_dir / 'comparison_accuracy.png')
    
    plot_metric_comparison(results, 'f1_macro',
                          'Model F1-Macro Comparison', 
                          output_dir / 'comparison_f1_macro.png')
    
    print("\nAll comparison plots generated successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate model comparison plots')
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--output', default='./plots', help='Output directory')
    args = parser.parse_args()
    
    generate_comparison_plots(args.results_dir, args.output)
