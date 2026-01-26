#!/usr/bin/env python3
"""
Visualization Demo for Thesis

Demonstrates all thesis figure generation capabilities.

Usage:
    python -m research.visualization.demo

Author: [Your Name]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np

from research.visualization.plots import (
    ThesisFigureGenerator,
    plot_convergence,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_cross_domain_heatmap,
    plot_ensemble_weights,
    plot_per_class_f1,
    plot_fuzzy_membership,
    MATPLOTLIB_AVAILABLE,
)


def main():
    """Run visualization demo."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   THESIS VISUALIZATION DEMO                                      ║
║   Auto-Generate Publication-Ready Figures                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib not installed!")
        print("Run: pip install matplotlib seaborn")
        return

    # Create output directory
    output_dir = Path("./figures")
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}\n")

    # 1. PSO Convergence Plot
    print("[1/7] Generating PSO Convergence Plot...")
    convergence_history = [
        0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.915,
        0.92, 0.923, 0.925, 0.927, 0.928, 0.929, 0.930, 0.931, 0.932, 0.933
    ]
    plot_convergence(
        convergence_history,
        title="Adaptive PSO Convergence for Ensemble Optimization",
        save_path=str(output_dir / "pso_convergence.png")
    )

    # 2. Confusion Matrix
    print("[2/7] Generating Confusion Matrix...")
    cm = np.array([
        [142, 8, 5],
        [12, 98, 15],
        [6, 10, 154]
    ])
    plot_confusion_matrix(
        cm,
        class_names=['Negative', 'Neutral', 'Positive'],
        title="Confusion Matrix: Fuzzy Ensemble on IMDB",
        save_path=str(output_dir / "confusion_matrix.png")
    )

    # Normalized version
    plot_confusion_matrix(
        cm,
        normalize=True,
        title="Normalized Confusion Matrix",
        save_path=str(output_dir / "confusion_matrix_normalized.png")
    )

    # 3. Model Comparison
    print("[3/7] Generating Model Comparison Chart...")
    plot_model_comparison(
        model_names=['LogReg', 'SVM', 'TF-IDF', 'BERT', 'Fuzzy', 'PSO-Ensemble'],
        metrics={
            'Accuracy': [0.847, 0.862, 0.831, 0.912, 0.823, 0.934],
            'F1 (Macro)': [0.835, 0.851, 0.819, 0.905, 0.811, 0.928],
            'F1 (Weighted)': [0.844, 0.859, 0.828, 0.910, 0.820, 0.932],
        },
        title="Model Performance Comparison on Sentiment140",
        save_path=str(output_dir / "model_comparison.png")
    )

    # 4. Cross-Domain Heatmap
    print("[4/7] Generating Cross-Domain Heatmap...")
    domains = ['Twitter', 'IMDB', 'Amazon', 'SST']
    cross_domain_matrix = np.array([
        [0.89, 0.72, 0.68, 0.75],
        [0.71, 0.92, 0.78, 0.80],
        [0.65, 0.76, 0.91, 0.73],
        [0.73, 0.79, 0.74, 0.88]
    ])
    plot_cross_domain_heatmap(
        cross_domain_matrix,
        domains,
        title="Cross-Domain Generalization (F1 Score)",
        save_path=str(output_dir / "cross_domain_heatmap.png")
    )

    # 5. Ensemble Weights
    print("[5/7] Generating Ensemble Weights Visualization...")
    plot_ensemble_weights(
        model_names=['LogReg', 'SVM', 'TF-IDF', 'Fuzzy'],
        weights=[0.22, 0.35, 0.25, 0.18],
        title="PSO-Optimized Ensemble Weights",
        save_path=str(output_dir / "ensemble_weights.png")
    )

    # 6. Per-Class F1 Radar
    print("[6/7] Generating Per-Class F1 Radar Chart...")
    plot_per_class_f1(
        model_names=['LogReg', 'SVM', 'Fuzzy', 'Ensemble'],
        per_class_f1={
            'LogReg': {'Negative': 0.85, 'Neutral': 0.72, 'Positive': 0.88},
            'SVM': {'Negative': 0.87, 'Neutral': 0.75, 'Positive': 0.86},
            'Fuzzy': {'Negative': 0.80, 'Neutral': 0.78, 'Positive': 0.82},
            'Ensemble': {'Negative': 0.91, 'Neutral': 0.83, 'Positive': 0.93},
        },
        title="Per-Class F1 Scores by Model",
        save_path=str(output_dir / "per_class_f1_radar.png")
    )

    # 7. Fuzzy Membership Functions
    print("[7/7] Generating Fuzzy Membership Functions...")
    plot_fuzzy_membership(
        title="Fuzzy Membership Functions for Sentiment Classification",
        save_path=str(output_dir / "fuzzy_membership.png")
    )

    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)

    # List generated files
    figures = list(output_dir.glob("*.png"))
    print(f"\nGenerated {len(figures)} figures:")
    for fig in sorted(figures):
        print(f"  - {fig.name}")

    # LaTeX includes
    print("\n" + "-" * 60)
    print("LATEX FIGURE INCLUDES:")
    print("-" * 60)
    for fig in sorted(figures):
        name = fig.stem.replace('_', ' ').title()
        print(f"""
\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.8\\textwidth]{{figures/{fig.name}}}
  \\caption{{{name}}}
  \\label{{fig:{fig.stem}}}
\\end{{figure}}
""")

    print("\n" + "=" * 60)
    print("Ready for thesis! Copy figures/ to your LaTeX project.")
    print("=" * 60)


if __name__ == '__main__':
    main()
