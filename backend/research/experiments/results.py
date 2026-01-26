"""
Results Aggregation and Thesis Report Generation

Provides tools for aggregating experiment results and generating
publication-ready LaTeX tables and figures.

Author: [Your Name]
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
from datetime import datetime


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float  # 95% CI
    ci_upper: float
    n_runs: int

    def to_latex(self, precision: int = 4) -> str:
        """Format as LaTeX with std deviation."""
        return f"${self.mean:.{precision}f} \\pm {self.std:.{precision}f}$"

    def to_latex_ci(self, precision: int = 4) -> str:
        """Format as LaTeX with confidence interval."""
        return f"${self.mean:.{precision}f}$ [{self.ci_lower:.{precision}f}, {self.ci_upper:.{precision}f}]"


class ResultsAggregator:
    """
    Aggregates results from multiple experiment runs.

    Computes statistics like mean, std, confidence intervals
    for publication-quality reporting.

    Example
    -------
    >>> aggregator = ResultsAggregator()
    >>> aggregator.add_result("run1", result1)
    >>> aggregator.add_result("run2", result2)
    >>> summary = aggregator.get_summary()
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.results: Dict[str, Any] = {}
        self.run_count = 0

    def add_result(self, run_name: str, result: Any) -> None:
        """Add experiment result."""
        self.results[run_name] = result
        self.run_count += 1

    def aggregate_metric(self, values: List[float]) -> AggregatedMetrics:
        """Aggregate a list of metric values."""
        arr = np.array(values)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if n > 1 else 0

        # Confidence interval (using t-distribution approximation)
        if n > 1:
            t_value = 1.96  # Approximate for 95% CI
            margin = t_value * std / np.sqrt(n)
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = ci_upper = mean

        return AggregatedMetrics(
            mean=float(mean),
            std=float(std),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            n_runs=n,
        )

    def get_model_aggregates(self) -> Dict[str, Dict[str, AggregatedMetrics]]:
        """Get aggregated metrics by model."""
        model_metrics: Dict[str, Dict[str, List[float]]] = {}

        for run_name, result in self.results.items():
            if not hasattr(result, 'dataset_results'):
                continue

            for ds_name, ds_result in result.dataset_results.items():
                for model_name, model_result in ds_result.model_results.items():
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {
                            'accuracy': [],
                            'f1_macro': [],
                            'f1_weighted': [],
                            'precision_macro': [],
                            'recall_macro': [],
                        }

                    model_metrics[model_name]['accuracy'].append(model_result.accuracy)
                    model_metrics[model_name]['f1_macro'].append(model_result.f1_macro)
                    model_metrics[model_name]['f1_weighted'].append(model_result.f1_weighted)
                    model_metrics[model_name]['precision_macro'].append(model_result.precision_macro)
                    model_metrics[model_name]['recall_macro'].append(model_result.recall_macro)

        # Aggregate
        aggregated = {}
        for model_name, metrics in model_metrics.items():
            aggregated[model_name] = {}
            for metric_name, values in metrics.items():
                aggregated[model_name][metric_name] = self.aggregate_metric(values)

        return aggregated

    def get_summary(self) -> Dict[str, Any]:
        """Get complete summary."""
        return {
            'n_runs': self.run_count,
            'model_aggregates': {
                model: {
                    metric: {
                        'mean': agg.mean,
                        'std': agg.std,
                        'ci_lower': agg.ci_lower,
                        'ci_upper': agg.ci_upper,
                    }
                    for metric, agg in metrics.items()
                }
                for model, metrics in self.get_model_aggregates().items()
            }
        }


class ThesisReportGenerator:
    """
    Generates thesis-ready LaTeX reports from experiment results.

    Produces:
    - Results tables (tabular format)
    - Cross-domain matrices
    - Ablation study tables
    - Statistical comparison tables

    Example
    -------
    >>> generator = ThesisReportGenerator(result)
    >>> latex = generator.generate_main_results_table()
    >>> generator.save_all("./thesis/tables/")
    """

    def __init__(self, result: Any = None, aggregator: ResultsAggregator = None):
        self.result = result
        self.aggregator = aggregator
        self.tables: Dict[str, str] = {}

    def generate_main_results_table(self) -> str:
        """Generate main results table."""
        if self.result is None:
            return ""

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Sentiment Analysis Results on Benchmark Datasets}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Dataset} & \\textbf{Accuracy} & "
            "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Time (ms)} \\\\",
            "\\midrule",
        ]

        for ds_name, ds_result in self.result.dataset_results.items():
            first_row = True
            for model_name, m in ds_result.model_results.items():
                ds_col = ds_name if first_row else ""
                lines.append(
                    f"{model_name} & {ds_col} & {m.accuracy:.4f} & "
                    f"{m.precision_macro:.4f} & {m.recall_macro:.4f} & "
                    f"{m.f1_macro:.4f} & {m.inference_time_ms:.2f} \\\\"
                )
                first_row = False
            lines.append("\\midrule")

        # Remove last midrule and add bottomrule
        lines[-1] = "\\bottomrule"

        lines.extend([
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['main_results'] = table
        return table

    def generate_model_comparison_table(self) -> str:
        """Generate model comparison table with statistical measures."""
        if self.aggregator is None:
            # Use single result
            if self.result is None:
                return ""

            lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Model Performance Comparison}",
                "\\label{tab:model_comparison}",
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1 (Macro)} & "
                "\\textbf{F1 (Weighted)} & \\textbf{Time (ms)} \\\\",
                "\\midrule",
            ]

            # Collect all model results
            model_scores: Dict[str, List[Dict]] = {}
            for ds_name, ds_result in self.result.dataset_results.items():
                for model_name, m in ds_result.model_results.items():
                    if model_name not in model_scores:
                        model_scores[model_name] = []
                    model_scores[model_name].append({
                        'accuracy': m.accuracy,
                        'f1_macro': m.f1_macro,
                        'f1_weighted': m.f1_weighted,
                        'time': m.inference_time_ms,
                    })

            # Average across datasets
            for model_name, scores in model_scores.items():
                avg_acc = np.mean([s['accuracy'] for s in scores])
                avg_f1m = np.mean([s['f1_macro'] for s in scores])
                avg_f1w = np.mean([s['f1_weighted'] for s in scores])
                avg_time = np.mean([s['time'] for s in scores])

                lines.append(
                    f"{model_name} & {avg_acc:.4f} & {avg_f1m:.4f} & "
                    f"{avg_f1w:.4f} & {avg_time:.2f} \\\\"
                )

            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])

            table = '\n'.join(lines)
            self.tables['model_comparison'] = table
            return table

        # Use aggregator for multiple runs
        aggregates = self.aggregator.get_model_aggregates()

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Performance Comparison (Mean $\\pm$ Std)}",
            "\\label{tab:model_comparison}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1 (Macro)} & "
            "\\textbf{F1 (Weighted)} \\\\",
            "\\midrule",
        ]

        for model_name, metrics in aggregates.items():
            acc = metrics['accuracy']
            f1m = metrics['f1_macro']
            f1w = metrics['f1_weighted']

            lines.append(
                f"{model_name} & {acc.to_latex()} & {f1m.to_latex()} & {f1w.to_latex()} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['model_comparison'] = table
        return table

    def generate_cross_domain_table(self) -> str:
        """Generate cross-domain evaluation matrix."""
        if self.result is None or not self.result.cross_domain_results:
            return ""

        datasets = list(self.result.cross_domain_results.keys())

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Cross-Domain Evaluation Results (F1 Score)}",
            "\\label{tab:cross_domain}",
            f"\\begin{{tabular}}{{l{'c' * len(datasets)}}}",
            "\\toprule",
            "\\textbf{Train $\\downarrow$ / Test $\\rightarrow$} & " +
            " & ".join([f"\\textbf{{{d}}}" for d in datasets]) + " \\\\",
            "\\midrule",
        ]

        for train_ds in datasets:
            row = [train_ds]
            for test_ds in datasets:
                if test_ds in self.result.cross_domain_results.get(train_ds, {}):
                    f1 = self.result.cross_domain_results[train_ds][test_ds].f1_macro
                    # Highlight diagonal (in-domain)
                    if train_ds == test_ds:
                        row.append(f"\\textbf{{{f1:.4f}}}")
                    else:
                        row.append(f"{f1:.4f}")
                else:
                    row.append("-")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['cross_domain'] = table
        return table

    def generate_ensemble_weights_table(self) -> str:
        """Generate ensemble weights table."""
        if self.result is None or self.result.ensemble_weights is None:
            return ""

        model_names = list(self.result.dataset_results.values())[0].model_results.keys() \
            if self.result.dataset_results else []
        model_names = list(model_names)

        if len(model_names) != len(self.result.ensemble_weights):
            model_names = [f"Model {i+1}" for i in range(len(self.result.ensemble_weights))]

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{PSO-Optimized Ensemble Weights}",
            "\\label{tab:ensemble_weights}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Weight} \\\\",
            "\\midrule",
        ]

        for name, weight in zip(model_names, self.result.ensemble_weights):
            lines.append(f"{name} & {weight:.4f} \\\\")

        lines.extend([
            "\\midrule",
            f"\\textbf{{Ensemble Accuracy}} & \\textbf{{{self.result.ensemble_accuracy:.4f}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['ensemble_weights'] = table
        return table

    def generate_per_class_table(self) -> str:
        """Generate per-class F1 scores table."""
        if self.result is None:
            return ""

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Per-Class F1 Scores}",
            "\\label{tab:per_class_f1}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Negative} & \\textbf{Neutral} & \\textbf{Positive} \\\\",
            "\\midrule",
        ]

        # Get first dataset's results
        for ds_name, ds_result in self.result.dataset_results.items():
            for model_name, m in ds_result.model_results.items():
                neg = m.per_class_f1.get('Negative', 0)
                neu = m.per_class_f1.get('Neutral', 0)
                pos = m.per_class_f1.get('Positive', 0)
                lines.append(f"{model_name} & {neg:.4f} & {neu:.4f} & {pos:.4f} \\\\")
            break  # Only first dataset

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['per_class_f1'] = table
        return table

    def generate_optimization_convergence(self) -> str:
        """Generate optimization convergence description for thesis."""
        if self.result is None or not self.result.optimization_results:
            return ""

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{PSO Optimization Statistics}",
            "\\label{tab:pso_stats}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "\\textbf{Parameter} & \\textbf{Value} \\\\",
            "\\midrule",
        ]

        for opt_name, opt_result in self.result.optimization_results.items():
            lines.extend([
                f"Algorithm & {opt_result.algorithm} \\\\",
                f"Iterations & {opt_result.n_iterations} \\\\",
                f"Evaluations & {opt_result.n_evaluations} \\\\",
                f"Best Fitness & {opt_result.best_fitness:.4f} \\\\",
                f"Runtime & {opt_result.runtime_seconds:.2f}s \\\\",
            ])

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        table = '\n'.join(lines)
        self.tables['pso_stats'] = table
        return table

    def generate_all_tables(self) -> Dict[str, str]:
        """Generate all tables."""
        self.generate_main_results_table()
        self.generate_model_comparison_table()
        self.generate_cross_domain_table()
        self.generate_ensemble_weights_table()
        self.generate_per_class_table()
        self.generate_optimization_convergence()
        return self.tables

    def save_all(self, output_dir: str) -> None:
        """Save all tables to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.generate_all_tables()

        for table_name, latex in self.tables.items():
            if latex:
                filepath = output_path / f"{table_name}.tex"
                with open(filepath, 'w') as f:
                    f.write(latex)

        # Save combined file
        combined_path = output_path / "all_tables.tex"
        with open(combined_path, 'w') as f:
            f.write("% Auto-generated thesis tables\n")
            f.write(f"% Generated: {datetime.now().isoformat()}\n\n")
            for table_name, latex in self.tables.items():
                if latex:
                    f.write(f"% === {table_name} ===\n")
                    f.write(latex)
                    f.write("\n\n")

    def get_results_summary(self) -> str:
        """Generate text summary of results."""
        if self.result is None:
            return "No results available."

        lines = [
            "=" * 60,
            "THESIS EXPERIMENT RESULTS SUMMARY",
            "=" * 60,
            "",
        ]

        # Best model per dataset
        lines.append("BEST MODELS BY DATASET:")
        lines.append("-" * 40)
        for ds_name, ds_result in self.result.dataset_results.items():
            lines.append(f"  {ds_name}: {ds_result.best_model} (F1={ds_result.best_score:.4f})")

        # Ensemble results
        if self.result.ensemble_weights is not None:
            lines.extend([
                "",
                "ENSEMBLE OPTIMIZATION:",
                "-" * 40,
                f"  Optimized Ensemble Accuracy: {self.result.ensemble_accuracy:.4f}",
                f"  Weights: {[f'{w:.3f}' for w in self.result.ensemble_weights]}",
            ])

        # Runtime
        lines.extend([
            "",
            f"Total Runtime: {self.result.total_runtime_seconds:.2f}s",
            "=" * 60,
        ])

        return '\n'.join(lines)
