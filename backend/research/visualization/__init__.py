"""
Thesis Visualization Module

Auto-generates publication-ready figures for thesis:
- PSO convergence plots
- Confusion matrix heatmaps
- Model comparison bar charts
- ROC curves
- Cross-domain heatmaps
- Ensemble weight visualizations

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from .plots import (
    ThesisFigureGenerator,
    plot_convergence,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_cross_domain_heatmap,
    plot_ensemble_weights,
    plot_per_class_f1,
    plot_fuzzy_membership,
)

__all__ = [
    'ThesisFigureGenerator',
    'plot_convergence',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_roc_curves',
    'plot_cross_domain_heatmap',
    'plot_ensemble_weights',
    'plot_per_class_f1',
    'plot_fuzzy_membership',
]
