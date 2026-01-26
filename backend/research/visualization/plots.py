"""
Thesis Figure Generation

Publication-ready visualizations for computational intelligence thesis.
All figures follow academic standards with proper formatting.

Author: [Your Name]
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Check for seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


# =============================================================================
# Style Configuration
# =============================================================================

# Academic color palette
THESIS_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'quinary': '#3B1F2B',      # Dark
    'positive': '#2E7D32',     # Green
    'negative': '#C62828',     # Red
    'neutral': '#757575',      # Gray
}

THESIS_STYLE = {
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
}


def apply_thesis_style():
    """Apply thesis-specific matplotlib style."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(THESIS_STYLE)
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette([
                THESIS_COLORS['primary'],
                THESIS_COLORS['secondary'],
                THESIS_COLORS['tertiary'],
                THESIS_COLORS['quaternary'],
            ])


# =============================================================================
# Individual Plot Functions
# =============================================================================

def plot_convergence(
    history: List[float],
    title: str = "PSO Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Best Fitness (Accuracy)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Any]:
    """
    Plot optimization convergence curve.

    Parameters
    ----------
    history : List[float]
        Fitness values over iterations
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(1, len(history) + 1)
    ax.plot(iterations, history,
            color=THESIS_COLORS['primary'],
            linewidth=2,
            marker='o',
            markersize=4,
            label='Best Fitness')

    # Add convergence annotations
    best_idx = np.argmax(history)
    ax.axhline(y=history[best_idx], color=THESIS_COLORS['secondary'],
               linestyle='--', alpha=0.7, label=f'Best: {history[best_idx]:.4f}')
    ax.scatter([best_idx + 1], [history[best_idx]],
               color=THESIS_COLORS['tertiary'], s=100, zorder=5,
               marker='*', label='Convergence Point')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim(1, len(history))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    normalize: bool = False,
) -> Optional[Any]:
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : List[str]
        Class labels
    normalize : bool
        Normalize by row sums

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    else:
        im = ax.imshow(cm, cmap=cmap)
        plt.colorbar(im, ax=ax)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm[i, j]
                text = f'{val:{fmt}}'
                ax.text(j, i, text, ha='center', va='center', color='white' if val > cm.max()/2 else 'black')
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[Any]:
    """
    Plot grouped bar chart comparing models.

    Parameters
    ----------
    model_names : List[str]
        Names of models
    metrics : Dict[str, List[float]]
        Metric name -> values for each model

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    colors = [THESIS_COLORS['primary'], THESIS_COLORS['secondary'],
              THESIS_COLORS['tertiary'], THESIS_COLORS['quaternary']]

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name,
                      color=colors[i % len(colors)], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_roc_curves(
    fpr_dict: Dict[str, np.ndarray],
    tpr_dict: Dict[str, np.ndarray],
    auc_dict: Dict[str, float],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Optional[Any]:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    fpr_dict : Dict[str, np.ndarray]
        False positive rates by model
    tpr_dict : Dict[str, np.ndarray]
        True positive rates by model
    auc_dict : Dict[str, float]
        AUC scores by model

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    fig, ax = plt.subplots(figsize=figsize)

    colors = [THESIS_COLORS['primary'], THESIS_COLORS['secondary'],
              THESIS_COLORS['tertiary'], THESIS_COLORS['quaternary']]

    for i, model_name in enumerate(fpr_dict.keys()):
        fpr = fpr_dict[model_name]
        tpr = tpr_dict[model_name]
        auc = auc_dict.get(model_name, 0)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{model_name} (AUC = {auc:.3f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_cross_domain_heatmap(
    matrix: np.ndarray,
    domain_names: List[str],
    title: str = "Cross-Domain Evaluation (F1 Score)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[Any]:
    """
    Plot cross-domain evaluation heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        Cross-domain scores (train x test)
    domain_names : List[str]
        Domain names

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    fig, ax = plt.subplots(figsize=figsize)

    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=domain_names, yticklabels=domain_names,
                    ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'})
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='F1 Score')
        for i in range(len(domain_names)):
            for j in range(len(domain_names)):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center')
        ax.set_xticks(range(len(domain_names)))
        ax.set_yticks(range(len(domain_names)))
        ax.set_xticklabels(domain_names)
        ax.set_yticklabels(domain_names)

    ax.set_xlabel('Test Domain')
    ax.set_ylabel('Train Domain')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_ensemble_weights(
    model_names: List[str],
    weights: List[float],
    title: str = "PSO-Optimized Ensemble Weights",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[Any]:
    """
    Plot ensemble weights as pie chart and bar chart.

    Parameters
    ----------
    model_names : List[str]
        Model names
    weights : List[float]
        Optimized weights

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = [THESIS_COLORS['primary'], THESIS_COLORS['secondary'],
              THESIS_COLORS['tertiary'], THESIS_COLORS['quaternary'],
              THESIS_COLORS['positive']]

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        weights,
        labels=model_names,
        autopct='%1.1f%%',
        colors=colors[:len(weights)],
        explode=[0.05] * len(weights),
        shadow=True,
        startangle=90
    )
    ax1.set_title('Weight Distribution')

    # Bar chart
    bars = ax2.bar(model_names, weights, color=colors[:len(weights)], alpha=0.85)
    for bar, weight in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Weight')
    ax2.set_title('Optimized Weights')
    ax2.set_ylim(0, max(weights) * 1.2)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_per_class_f1(
    model_names: List[str],
    per_class_f1: Dict[str, Dict[str, float]],
    title: str = "Per-Class F1 Scores",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[Any]:
    """
    Plot per-class F1 scores as radar/spider chart.

    Parameters
    ----------
    model_names : List[str]
        Model names
    per_class_f1 : Dict[str, Dict[str, float]]
        Model -> class -> F1 score

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    classes = ['Negative', 'Neutral', 'Positive']
    n_classes = len(classes)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = [THESIS_COLORS['primary'], THESIS_COLORS['secondary'],
              THESIS_COLORS['tertiary'], THESIS_COLORS['quaternary']]

    for i, model_name in enumerate(model_names):
        if model_name in per_class_f1:
            values = [per_class_f1[model_name].get(c, 0) for c in classes]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2,
                    color=colors[i % len(colors)], label=model_name)
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_fuzzy_membership(
    x_range: Tuple[float, float] = (-1, 1),
    n_points: int = 200,
    title: str = "Fuzzy Membership Functions for Sentiment",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional[Any]:
    """
    Plot fuzzy membership functions for sentiment classes.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return None

    apply_thesis_style()

    x = np.linspace(x_range[0], x_range[1], n_points)

    # Gaussian membership functions
    def gaussian_mf(x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    negative = gaussian_mf(x, -0.7, 0.3)
    neutral = gaussian_mf(x, 0.0, 0.25)
    positive = gaussian_mf(x, 0.7, 0.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Membership functions
    ax1.plot(x, negative, color=THESIS_COLORS['negative'],
             linewidth=2, label='Negative')
    ax1.plot(x, neutral, color=THESIS_COLORS['neutral'],
             linewidth=2, label='Neutral')
    ax1.plot(x, positive, color=THESIS_COLORS['positive'],
             linewidth=2, label='Positive')
    ax1.fill_between(x, negative, alpha=0.2, color=THESIS_COLORS['negative'])
    ax1.fill_between(x, neutral, alpha=0.2, color=THESIS_COLORS['neutral'])
    ax1.fill_between(x, positive, alpha=0.2, color=THESIS_COLORS['positive'])
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Membership Degree')
    ax1.set_title('Gaussian Membership Functions')
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Triangular membership functions
    def triangular_mf(x, a, b, c):
        return np.maximum(0, np.minimum((x - a) / (b - a + 1e-10),
                                         (c - x) / (c - b + 1e-10)))

    neg_tri = triangular_mf(x, -1.0, -0.7, -0.2)
    neu_tri = triangular_mf(x, -0.4, 0.0, 0.4)
    pos_tri = triangular_mf(x, 0.2, 0.7, 1.0)

    ax2.plot(x, neg_tri, color=THESIS_COLORS['negative'],
             linewidth=2, label='Negative')
    ax2.plot(x, neu_tri, color=THESIS_COLORS['neutral'],
             linewidth=2, label='Neutral')
    ax2.plot(x, pos_tri, color=THESIS_COLORS['positive'],
             linewidth=2, label='Positive')
    ax2.fill_between(x, neg_tri, alpha=0.2, color=THESIS_COLORS['negative'])
    ax2.fill_between(x, neu_tri, alpha=0.2, color=THESIS_COLORS['neutral'])
    ax2.fill_between(x, pos_tri, alpha=0.2, color=THESIS_COLORS['positive'])
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Membership Degree')
    ax2.set_title('Triangular Membership Functions')
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Thesis Figure Generator Class
# =============================================================================

@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    output_dir: str = "./figures"
    format: str = "png"  # png, pdf, svg
    dpi: int = 300
    style: str = "thesis"


class ThesisFigureGenerator:
    """
    Comprehensive thesis figure generator.

    Generates all figures needed for thesis from experiment results.

    Example
    -------
    >>> generator = ThesisFigureGenerator(result, output_dir="./thesis/figures")
    >>> generator.generate_all()
    """

    def __init__(
        self,
        result: Any = None,
        output_dir: str = "./figures",
        format: str = "png",
    ):
        self.result = result
        self.output_dir = Path(output_dir)
        self.format = format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_figures: List[str] = []

    def generate_all(self) -> List[str]:
        """Generate all thesis figures."""
        print("\n" + "=" * 50)
        print("GENERATING THESIS FIGURES")
        print("=" * 50)

        if not MATPLOTLIB_AVAILABLE:
            print("ERROR: matplotlib not installed. Run: pip install matplotlib seaborn")
            return []

        # Generate each figure type
        self._generate_convergence_plot()
        self._generate_confusion_matrices()
        self._generate_model_comparison()
        self._generate_ensemble_weights()
        self._generate_per_class_f1()
        self._generate_fuzzy_membership()

        if self.result and hasattr(self.result, 'cross_domain_results') and self.result.cross_domain_results:
            self._generate_cross_domain_heatmap()

        print(f"\nGenerated {len(self.generated_figures)} figures in {self.output_dir}/")
        return self.generated_figures

    def _save_path(self, name: str) -> str:
        """Get save path for figure."""
        return str(self.output_dir / f"{name}.{self.format}")

    def _generate_convergence_plot(self) -> None:
        """Generate PSO convergence plot."""
        if self.result and hasattr(self.result, 'optimization_results'):
            for opt_name, opt_result in self.result.optimization_results.items():
                if hasattr(opt_result, 'convergence_history') and opt_result.convergence_history:
                    path = self._save_path(f"convergence_{opt_name}")
                    plot_convergence(
                        opt_result.convergence_history,
                        title=f"{opt_result.algorithm} Convergence",
                        save_path=path
                    )
                    self.generated_figures.append(path)
                    plt.close()
        else:
            # Demo convergence plot
            demo_history = [0.7, 0.75, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.935]
            path = self._save_path("convergence_demo")
            plot_convergence(demo_history, title="PSO Convergence (Demo)", save_path=path)
            self.generated_figures.append(path)
            plt.close()

    def _generate_confusion_matrices(self) -> None:
        """Generate confusion matrix heatmaps."""
        if self.result and hasattr(self.result, 'dataset_results'):
            for ds_name, ds_result in self.result.dataset_results.items():
                for model_name, model_result in ds_result.model_results.items():
                    if hasattr(model_result, 'confusion_matrix'):
                        cm = model_result.confusion_matrix
                        if isinstance(cm, np.ndarray) and cm.size > 0:
                            path = self._save_path(f"confusion_{model_name}_{ds_name}")
                            plot_confusion_matrix(
                                cm,
                                title=f"Confusion Matrix: {model_name} on {ds_name}",
                                save_path=path
                            )
                            self.generated_figures.append(path)
                            plt.close()
        else:
            # Demo confusion matrix
            demo_cm = np.array([[45, 3, 2], [5, 38, 7], [2, 4, 44]])
            path = self._save_path("confusion_demo")
            plot_confusion_matrix(demo_cm, title="Confusion Matrix (Demo)", save_path=path)
            self.generated_figures.append(path)
            plt.close()

    def _generate_model_comparison(self) -> None:
        """Generate model comparison chart."""
        if self.result and hasattr(self.result, 'dataset_results'):
            # Collect metrics
            model_names = []
            accuracies = []
            f1_scores = []
            precisions = []

            for ds_name, ds_result in self.result.dataset_results.items():
                for model_name, model_result in ds_result.model_results.items():
                    if model_name not in model_names:
                        model_names.append(model_name)
                        accuracies.append(model_result.accuracy)
                        f1_scores.append(model_result.f1_macro)
                        precisions.append(model_result.precision_macro)

            if model_names:
                path = self._save_path("model_comparison")
                plot_model_comparison(
                    model_names,
                    {'Accuracy': accuracies, 'F1 Score': f1_scores, 'Precision': precisions},
                    save_path=path
                )
                self.generated_figures.append(path)
                plt.close()
        else:
            # Demo comparison
            path = self._save_path("model_comparison_demo")
            plot_model_comparison(
                ['LogReg', 'SVM', 'TF-IDF', 'Fuzzy', 'Ensemble'],
                {'Accuracy': [0.85, 0.87, 0.83, 0.80, 0.91],
                 'F1 Score': [0.84, 0.86, 0.82, 0.78, 0.90]},
                save_path=path
            )
            self.generated_figures.append(path)
            plt.close()

    def _generate_ensemble_weights(self) -> None:
        """Generate ensemble weights visualization."""
        if self.result and hasattr(self.result, 'ensemble_weights') and self.result.ensemble_weights is not None:
            weights = self.result.ensemble_weights
            # Get model names
            model_names = []
            for ds_result in self.result.dataset_results.values():
                model_names = list(ds_result.model_results.keys())
                break

            if len(model_names) == len(weights):
                path = self._save_path("ensemble_weights")
                plot_ensemble_weights(model_names, list(weights), save_path=path)
                self.generated_figures.append(path)
                plt.close()
        else:
            # Demo weights
            path = self._save_path("ensemble_weights_demo")
            plot_ensemble_weights(
                ['LogReg', 'SVM', 'TF-IDF', 'Fuzzy'],
                [0.25, 0.35, 0.22, 0.18],
                save_path=path
            )
            self.generated_figures.append(path)
            plt.close()

    def _generate_per_class_f1(self) -> None:
        """Generate per-class F1 radar chart."""
        if self.result and hasattr(self.result, 'dataset_results'):
            per_class_data = {}
            model_names = []

            for ds_result in self.result.dataset_results.values():
                for model_name, model_result in ds_result.model_results.items():
                    if hasattr(model_result, 'per_class_f1'):
                        per_class_data[model_name] = model_result.per_class_f1
                        if model_name not in model_names:
                            model_names.append(model_name)
                break  # First dataset only

            if per_class_data:
                path = self._save_path("per_class_f1")
                plot_per_class_f1(model_names, per_class_data, save_path=path)
                self.generated_figures.append(path)
                plt.close()
        else:
            # Demo per-class
            path = self._save_path("per_class_f1_demo")
            plot_per_class_f1(
                ['LogReg', 'SVM', 'Ensemble'],
                {
                    'LogReg': {'Negative': 0.85, 'Neutral': 0.72, 'Positive': 0.88},
                    'SVM': {'Negative': 0.87, 'Neutral': 0.75, 'Positive': 0.86},
                    'Ensemble': {'Negative': 0.90, 'Neutral': 0.80, 'Positive': 0.91},
                },
                save_path=path
            )
            self.generated_figures.append(path)
            plt.close()

    def _generate_cross_domain_heatmap(self) -> None:
        """Generate cross-domain heatmap."""
        if self.result and hasattr(self.result, 'cross_domain_results') and self.result.cross_domain_results:
            domains = list(self.result.cross_domain_results.keys())
            n = len(domains)
            matrix = np.zeros((n, n))

            for i, train_ds in enumerate(domains):
                for j, test_ds in enumerate(domains):
                    if test_ds in self.result.cross_domain_results.get(train_ds, {}):
                        matrix[i, j] = self.result.cross_domain_results[train_ds][test_ds].f1_macro

            path = self._save_path("cross_domain_heatmap")
            plot_cross_domain_heatmap(matrix, domains, save_path=path)
            self.generated_figures.append(path)
            plt.close()

    def _generate_fuzzy_membership(self) -> None:
        """Generate fuzzy membership function plot."""
        path = self._save_path("fuzzy_membership")
        plot_fuzzy_membership(save_path=path)
        self.generated_figures.append(path)
        plt.close()

    def get_latex_includes(self) -> str:
        """Generate LaTeX include statements for all figures."""
        lines = ["% Auto-generated figure includes", ""]
        for fig_path in self.generated_figures:
            fig_name = Path(fig_path).stem
            lines.append(f"\\begin{{figure}}[htbp]")
            lines.append(f"  \\centering")
            lines.append(f"  \\includegraphics[width=0.8\\textwidth]{{{fig_path}}}")
            lines.append(f"  \\caption{{{fig_name.replace('_', ' ').title()}}}")
            lines.append(f"  \\label{{fig:{fig_name}}}")
            lines.append(f"\\end{{figure}}")
            lines.append("")
        return '\n'.join(lines)
