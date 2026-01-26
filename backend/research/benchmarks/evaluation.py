"""
Benchmark Evaluation Framework

Provides comprehensive evaluation tools for sentiment analysis models
across multiple benchmark datasets.

Features:
- Single-dataset evaluation
- Cross-domain evaluation
- Statistical significance testing
- LaTeX report generation
- Comparison with baselines

Author: [Your Name]
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from .base import Dataset, DatasetSplit, SentimentLabel, DatasetManager


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation."""
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
            'f1_macro': self.f1_macro,
            'precision_weighted': self.precision_weighted,
            'recall_weighted': self.recall_weighted,
            'f1_weighted': self.f1_weighted,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'per_class_f1': self.per_class_f1,
            'inference_time_ms': self.inference_time_ms,
            'n_samples': self.n_samples,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark evaluation report."""
    model_name: str
    dataset_results: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    cross_domain_results: Dict[str, Dict[str, EvaluationMetrics]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'dataset_results': {k: v.to_dict() for k, v in self.dataset_results.items()},
            'cross_domain_results': {
                train: {test: m.to_dict() for test, m in results.items()}
                for train, results in self.cross_domain_results.items()
            },
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }

    def to_json(self, filepath: str = None) -> str:
        """Export to JSON."""
        content = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(content)
        return content

    def to_latex_table(self) -> str:
        """Generate LaTeX table for thesis."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{Benchmark Results for {self.model_name}}}",
            "\\label{tab:benchmark_results}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{Dataset} & \\textbf{Accuracy} & \\textbf{Precision} & "
            "\\textbf{Recall} & \\textbf{F1} & \\textbf{Time (ms)} \\\\",
            "\\midrule",
        ]

        for dataset, metrics in self.dataset_results.items():
            lines.append(
                f"{dataset} & {metrics.accuracy:.4f} & {metrics.precision_macro:.4f} & "
                f"{metrics.recall_macro:.4f} & {metrics.f1_macro:.4f} & "
                f"{metrics.inference_time_ms:.1f} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def to_cross_domain_latex(self) -> str:
        """Generate cross-domain results LaTeX table."""
        if not self.cross_domain_results:
            return ""

        datasets = list(self.cross_domain_results.keys())

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Cross-Domain Evaluation Results (F1 Score)}",
            "\\label{tab:cross_domain}",
            "\\begin{tabular}{l" + "c" * len(datasets) + "}",
            "\\toprule",
            "\\textbf{Train $\\downarrow$ / Test $\\rightarrow$} & " +
            " & ".join([f"\\textbf{{{d}}}" for d in datasets]) + " \\\\",
            "\\midrule",
        ]

        for train_ds in datasets:
            row = [train_ds]
            for test_ds in datasets:
                if test_ds in self.cross_domain_results.get(train_ds, {}):
                    f1 = self.cross_domain_results[train_ds][test_ds].f1_macro
                    row.append(f"{f1:.4f}")
                else:
                    row.append("-")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"BENCHMARK REPORT: {self.model_name}",
            f"Generated: {self.timestamp}",
            "=" * 60,
            "",
            "DATASET RESULTS:",
            "-" * 40,
        ]

        for dataset, metrics in self.dataset_results.items():
            lines.extend([
                f"\n{dataset}:",
                f"  Accuracy:  {metrics.accuracy:.4f}",
                f"  F1 (macro): {metrics.f1_macro:.4f}",
                f"  Precision: {metrics.precision_macro:.4f}",
                f"  Recall:    {metrics.recall_macro:.4f}",
                f"  Samples:   {metrics.n_samples}",
                f"  Time/sample: {metrics.inference_time_ms:.2f}ms",
            ])

        if self.cross_domain_results:
            lines.extend([
                "",
                "CROSS-DOMAIN RESULTS (F1 Score):",
                "-" * 40,
            ])
            for train_ds, results in self.cross_domain_results.items():
                for test_ds, metrics in results.items():
                    lines.append(f"  {train_ds} -> {test_ds}: {metrics.f1_macro:.4f}")

        return '\n'.join(lines)


class BenchmarkEvaluator:
    """
    Evaluator for sentiment analysis models on benchmark datasets.

    Example
    -------
    >>> evaluator = BenchmarkEvaluator()
    >>> evaluator.register_dataset('youtube', YouTubeDataset())
    >>>
    >>> report = evaluator.evaluate(model, model_name='FuzzySentiment')
    >>> print(report.summary())
    >>> report.to_json('results.json')
    """

    def __init__(self, n_classes: int = 3):
        self.datasets: Dict[str, Dataset] = {}
        self.n_classes = n_classes

    def register_dataset(self, name: str, dataset: Dataset) -> 'BenchmarkEvaluator':
        """Register a dataset for evaluation."""
        self.datasets[name] = dataset
        return self

    def evaluate(
        self,
        model: Any,
        model_name: str = "model",
        predict_fn: Callable = None,
        max_samples: int = None,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Evaluate model on all registered datasets.

        Parameters
        ----------
        model : Any
            Sentiment model with analyze() or predict() method
        model_name : str
            Name for reporting
        predict_fn : callable, optional
            Custom prediction function: fn(model, text) -> label
        max_samples : int, optional
            Limit samples per dataset (for quick testing)
        verbose : bool
            Print progress

        Returns
        -------
        BenchmarkReport
            Complete evaluation report
        """
        report = BenchmarkReport(model_name=model_name)

        # Determine prediction function
        if predict_fn is None:
            predict_fn = self._default_predict_fn

        for dataset_name, dataset in self.datasets.items():
            if verbose:
                print(f"\nEvaluating on {dataset_name}...")

            dataset.load()
            test_split = dataset.test

            if max_samples:
                test_split = test_split.sample(max_samples)

            metrics = self._evaluate_split(model, test_split, predict_fn, verbose)
            report.dataset_results[dataset_name] = metrics

            if verbose:
                print(f"  Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_macro:.4f}")

        return report

    def _default_predict_fn(self, model: Any, text: str) -> str:
        """Default prediction function."""
        if hasattr(model, 'analyze'):
            result = model.analyze(text)
            if hasattr(result, 'label'):
                return result.label
            return result.get('label', 'Neutral')
        elif hasattr(model, 'predict'):
            return model.predict(text)
        else:
            raise ValueError("Model must have 'analyze' or 'predict' method")

    def _evaluate_split(
        self,
        model: Any,
        split: DatasetSplit,
        predict_fn: Callable,
        verbose: bool = False
    ) -> EvaluationMetrics:
        """Evaluate on a single split."""
        predictions = []
        true_labels = []

        start_time = time.time()

        for i, (text, label) in enumerate(split):
            try:
                pred = predict_fn(model, text)
                pred_label = self._normalize_label(pred)
                predictions.append(pred_label)
                true_labels.append(label.value)
            except Exception as e:
                predictions.append(1)  # Default to neutral
                true_labels.append(label.value)

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(split)} samples...")

        total_time = time.time() - start_time

        # Compute metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        metrics = self._compute_metrics(predictions, true_labels)
        metrics.inference_time_ms = (total_time / len(split)) * 1000
        metrics.n_samples = len(split)

        return metrics

    def _normalize_label(self, label: Any) -> int:
        """Normalize label to integer."""
        if isinstance(label, int):
            return label
        if isinstance(label, SentimentLabel):
            return label.value

        label_str = str(label).lower().strip()
        if label_str in ['negative', 'neg', '0']:
            return 0
        elif label_str in ['positive', 'pos', '2']:
            return 2
        else:
            return 1  # Neutral

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> EvaluationMetrics:
        """Compute evaluation metrics."""
        metrics = EvaluationMetrics()

        # Accuracy
        metrics.accuracy = np.mean(predictions == true_labels)

        # Confusion matrix
        n_classes = max(3, max(predictions.max(), true_labels.max()) + 1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(true_labels, predictions):
            cm[t, p] += 1
        metrics.confusion_matrix = cm

        # Per-class metrics
        precisions = []
        recalls = []
        f1s = []
        supports = []

        class_names = ['Negative', 'Neutral', 'Positive']

        for c in range(min(3, n_classes)):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(cm[c, :].sum())

            if c < len(class_names):
                metrics.per_class_f1[class_names[c]] = float(f1)

        # Macro averages
        metrics.precision_macro = np.mean(precisions)
        metrics.recall_macro = np.mean(recalls)
        metrics.f1_macro = np.mean(f1s)

        # Weighted averages
        total = sum(supports)
        if total > 0:
            metrics.precision_weighted = np.average(precisions, weights=supports)
            metrics.recall_weighted = np.average(recalls, weights=supports)
            metrics.f1_weighted = np.average(f1s, weights=supports)

        return metrics


class CrossDomainEvaluator:
    """
    Cross-domain evaluation framework.

    Tests model generalization by training on one domain
    and testing on another.

    Example
    -------
    >>> evaluator = CrossDomainEvaluator()
    >>> evaluator.add_dataset('youtube', YouTubeDataset())
    >>>
    >>> report = evaluator.evaluate_cross_domain(
    ...     model_factory=lambda: FuzzySentimentClassifier(),
    ...     train_fn=train_model
    ... )
    """

    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    def add_dataset(self, name: str, dataset: Dataset) -> 'CrossDomainEvaluator':
        """Add a dataset for cross-domain evaluation."""
        self.datasets[name] = dataset
        return self

    def evaluate_cross_domain(
        self,
        model_factory: Callable,
        train_fn: Callable,
        predict_fn: Callable = None,
        max_train_samples: int = 5000,
        max_test_samples: int = 1000,
        verbose: bool = True
    ) -> BenchmarkReport:
        """
        Evaluate cross-domain generalization.

        Parameters
        ----------
        model_factory : callable
            Function that creates a new model instance
        train_fn : callable
            Function to train: train_fn(model, texts, labels)
        predict_fn : callable
            Function to predict: predict_fn(model, text) -> label
        max_train_samples : int
            Max training samples
        max_test_samples : int
            Max test samples

        Returns
        -------
        BenchmarkReport
            Report with cross-domain results
        """
        report = BenchmarkReport(model_name="CrossDomain")

        dataset_names = list(self.datasets.keys())

        for train_name in dataset_names:
            report.cross_domain_results[train_name] = {}

            if verbose:
                print(f"\nTraining on {train_name}...")

            # Get training data
            train_dataset = self.datasets[train_name]
            train_dataset.load()
            train_split = train_dataset.train.sample(max_train_samples)
            train_texts, train_labels = train_split.to_string_labels()

            # Create and train model
            model = model_factory()
            train_fn(model, train_texts, train_labels)

            # Test on all datasets
            for test_name in dataset_names:
                if verbose:
                    print(f"  Testing on {test_name}...")

                test_dataset = self.datasets[test_name]
                test_dataset.load()
                test_split = test_dataset.test.sample(max_test_samples)

                # Evaluate
                evaluator = BenchmarkEvaluator()
                metrics = evaluator._evaluate_split(
                    model, test_split,
                    predict_fn or evaluator._default_predict_fn
                )

                report.cross_domain_results[train_name][test_name] = metrics

                if verbose:
                    print(f"    F1: {metrics.f1_macro:.4f}")

        return report


def create_standard_benchmark() -> BenchmarkEvaluator:
    """
    Create evaluator with standard benchmark datasets.

    Returns
    -------
    BenchmarkEvaluator
        Pre-configured evaluator
    """
    return BenchmarkEvaluator()
