"""
Fuzzy Evaluation Metrics for Thesis-Grade Analysis

This module provides comprehensive evaluation metrics specifically
designed for fuzzy sentiment classification systems.

Standard ML metrics (accuracy, F1) don't capture the nuances of
fuzzy classification, such as:
- Uncertainty quantification quality
- Calibration of fuzzy outputs
- Handling of ambiguous samples
- Inter-model agreement analysis

Metrics Provided:
    1. Standard Classification Metrics (with fuzzy adaptations)
    2. Uncertainty Quality Metrics
    3. Fuzzy-Specific Metrics
    4. Calibration Metrics
    5. Statistical Comparison Tests

Reference:
    - Hullermeier, E., & Waegeman, W. (2021). "Aleatoric and epistemic
      uncertainty in machine learning: An introduction to concepts and methods"
    - Pedrycz, W. (2013). "Granular Computing: Analysis and Design of
      Intelligent Systems"

Author: [Your Name]
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json


@dataclass
class FuzzyEvaluationResult:
    """
    Comprehensive evaluation results for fuzzy sentiment classification.

    Attributes:
        standard_metrics: Traditional ML metrics (accuracy, precision, recall, F1)
        fuzzy_metrics: Fuzzy-specific metrics (fuzziness, specificity, etc.)
        uncertainty_metrics: Uncertainty quality assessment
        calibration_metrics: Probability calibration measures
        per_class_metrics: Metrics broken down by class
        confusion_matrix: Standard confusion matrix
        fuzzy_confusion_matrix: Soft confusion matrix with partial memberships
        sample_analysis: Per-sample analysis results
    """
    standard_metrics: Dict[str, float] = field(default_factory=dict)
    fuzzy_metrics: Dict[str, float] = field(default_factory=dict)
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    fuzzy_confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    sample_analysis: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'standard_metrics': self.standard_metrics,
            'fuzzy_metrics': self.fuzzy_metrics,
            'uncertainty_metrics': self.uncertainty_metrics,
            'calibration_metrics': self.calibration_metrics,
            'per_class_metrics': self.per_class_metrics,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'fuzzy_confusion_matrix': self.fuzzy_confusion_matrix.tolist(),
        }

    def to_latex_table(self) -> str:
        """Generate LaTeX table for thesis."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Fuzzy Sentiment Classification Results}",
            "\\label{tab:fuzzy_results}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Value} \\\\",
            "\\midrule",
        ]

        # Add standard metrics
        for metric, value in self.standard_metrics.items():
            lines.append(f"{metric.replace('_', ' ').title()} & {value:.4f} \\\\")

        lines.append("\\midrule")

        # Add fuzzy metrics
        for metric, value in self.fuzzy_metrics.items():
            lines.append(f"{metric.replace('_', ' ').title()} & {value:.4f} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = ["=" * 50, "FUZZY EVALUATION RESULTS", "=" * 50, ""]

        lines.append("Standard Metrics:")
        for k, v in self.standard_metrics.items():
            lines.append(f"  {k}: {v:.4f}")

        lines.append("\nFuzzy Metrics:")
        for k, v in self.fuzzy_metrics.items():
            lines.append(f"  {k}: {v:.4f}")

        lines.append("\nUncertainty Metrics:")
        for k, v in self.uncertainty_metrics.items():
            lines.append(f"  {k}: {v:.4f}")

        lines.append("\nCalibration Metrics:")
        for k, v in self.calibration_metrics.items():
            lines.append(f"  {k}: {v:.4f}")

        return '\n'.join(lines)


class FuzzyEvaluator:
    """
    Comprehensive evaluator for fuzzy sentiment classification.

    This evaluator computes both standard ML metrics and fuzzy-specific
    metrics for thorough thesis-level analysis.

    Example
    -------
    >>> evaluator = FuzzyEvaluator(classes=['Negative', 'Neutral', 'Positive'])
    >>>
    >>> # Collect predictions and true labels
    >>> for text, true_label in test_data:
    ...     result = classifier.classify(model_outputs)
    ...     evaluator.add_sample(
    ...         true_label=true_label,
    ...         predicted_label=result.label,
    ...         probabilities=result.probabilities,
    ...         uncertainty=result.uncertainty_metrics,
    ...         crisp_score=result.crisp_score
    ...     )
    >>>
    >>> # Compute all metrics
    >>> evaluation = evaluator.compute_metrics()
    >>> print(evaluation.summary())
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        uncertainty_threshold: float = 0.3
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        classes : list, optional
            List of class labels (default: ['Negative', 'Neutral', 'Positive'])
        uncertainty_threshold : float
            Threshold for classifying predictions as "uncertain"
        """
        self.classes = classes or ['Negative', 'Neutral', 'Positive']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.uncertainty_threshold = uncertainty_threshold

        # Storage for samples
        self.true_labels: List[str] = []
        self.predicted_labels: List[str] = []
        self.probabilities: List[Dict[str, float]] = []
        self.uncertainties: List[Dict[str, float]] = []
        self.crisp_scores: List[float] = []

    def reset(self) -> None:
        """Clear all stored samples."""
        self.true_labels = []
        self.predicted_labels = []
        self.probabilities = []
        self.uncertainties = []
        self.crisp_scores = []

    def add_sample(
        self,
        true_label: str,
        predicted_label: str,
        probabilities: Dict[str, float],
        uncertainty: Dict[str, float],
        crisp_score: float
    ) -> None:
        """
        Add a single sample for evaluation.

        Parameters
        ----------
        true_label : str
            Ground truth label
        predicted_label : str
            Predicted label from classifier
        probabilities : dict
            Class probabilities from fuzzy classifier
        uncertainty : dict
            Uncertainty metrics from fuzzy classifier
        crisp_score : float
            Defuzzified sentiment score
        """
        self.true_labels.append(true_label)
        self.predicted_labels.append(predicted_label)
        self.probabilities.append(probabilities)
        self.uncertainties.append(uncertainty)
        self.crisp_scores.append(crisp_score)

    def add_batch(
        self,
        true_labels: List[str],
        results: List[Any]  # List of FuzzySentimentResult
    ) -> None:
        """Add multiple samples from FuzzySentimentResult objects."""
        for true_label, result in zip(true_labels, results):
            self.add_sample(
                true_label=true_label,
                predicted_label=result.label,
                probabilities=result.probabilities,
                uncertainty=result.uncertainty_metrics,
                crisp_score=result.crisp_score
            )

    def compute_metrics(self) -> FuzzyEvaluationResult:
        """
        Compute all evaluation metrics.

        Returns
        -------
        FuzzyEvaluationResult
            Comprehensive evaluation results
        """
        result = FuzzyEvaluationResult()

        if len(self.true_labels) == 0:
            return result

        # Convert to numpy arrays
        y_true = np.array([self.class_to_idx[l] for l in self.true_labels])
        y_pred = np.array([self.class_to_idx[l] for l in self.predicted_labels])

        # Standard metrics
        result.standard_metrics = self._compute_standard_metrics(y_true, y_pred)

        # Confusion matrices
        result.confusion_matrix = self._compute_confusion_matrix(y_true, y_pred)
        result.fuzzy_confusion_matrix = self._compute_fuzzy_confusion_matrix()

        # Per-class metrics
        result.per_class_metrics = self._compute_per_class_metrics(y_true, y_pred)

        # Fuzzy-specific metrics
        result.fuzzy_metrics = self._compute_fuzzy_metrics()

        # Uncertainty metrics
        result.uncertainty_metrics = self._compute_uncertainty_quality_metrics(y_true, y_pred)

        # Calibration metrics
        result.calibration_metrics = self._compute_calibration_metrics(y_true)

        # Sample analysis
        result.sample_analysis = self._analyze_samples(y_true, y_pred)

        return result

    def _compute_standard_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        n_classes = len(self.classes)

        # Accuracy
        accuracy = np.mean(y_true == y_pred)

        # Per-class precision, recall, F1
        precisions = []
        recalls = []
        f1s = []

        for c in range(n_classes):
            true_positive = np.sum((y_true == c) & (y_pred == c))
            false_positive = np.sum((y_true != c) & (y_pred == c))
            false_negative = np.sum((y_true == c) & (y_pred != c))

            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            'accuracy': float(accuracy),
            'precision_macro': float(np.mean(precisions)),
            'recall_macro': float(np.mean(recalls)),
            'f1_macro': float(np.mean(f1s)),
            'precision_weighted': float(np.average(precisions, weights=np.bincount(y_true, minlength=n_classes))),
            'recall_weighted': float(np.average(recalls, weights=np.bincount(y_true, minlength=n_classes))),
            'f1_weighted': float(np.average(f1s, weights=np.bincount(y_true, minlength=n_classes))),
        }

    def _compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute standard confusion matrix."""
        n_classes = len(self.classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        return cm

    def _compute_fuzzy_confusion_matrix(self) -> np.ndarray:
        """
        Compute fuzzy (soft) confusion matrix.

        Instead of hard assignments, uses class probabilities
        to create a soft confusion matrix.

        This captures partial class memberships and provides
        more nuanced error analysis.
        """
        n_classes = len(self.classes)
        fcm = np.zeros((n_classes, n_classes), dtype=float)

        for true_label, probs in zip(self.true_labels, self.probabilities):
            true_idx = self.class_to_idx[true_label]
            for class_name, prob in probs.items():
                pred_idx = self.class_to_idx.get(class_name)
                if pred_idx is not None:
                    fcm[true_idx, pred_idx] += prob

        return fcm

    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class."""
        metrics = {}

        for c, class_name in enumerate(self.classes):
            true_positive = np.sum((y_true == c) & (y_pred == c))
            false_positive = np.sum((y_true != c) & (y_pred == c))
            false_negative = np.sum((y_true == c) & (y_pred != c))
            true_negative = np.sum((y_true != c) & (y_pred != c))

            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            specificity = true_negative / (true_negative + false_positive + 1e-10)

            metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity),
                'support': int(np.sum(y_true == c)),
            }

        return metrics

    def _compute_fuzzy_metrics(self) -> Dict[str, float]:
        """
        Compute fuzzy-specific metrics.

        These metrics capture the quality of fuzzy classification
        beyond standard accuracy measures.
        """
        # Average fuzziness
        fuzziness_values = [u.get('fuzziness', 0) for u in self.uncertainties]
        avg_fuzziness = np.mean(fuzziness_values)

        # Average specificity
        specificity_values = [u.get('specificity', 0) for u in self.uncertainties]
        avg_specificity = np.mean(specificity_values)

        # Average entropy
        entropy_values = [u.get('entropy', 0) for u in self.uncertainties]
        avg_entropy = np.mean(entropy_values)

        # Membership concentration (how concentrated are class memberships)
        max_probs = [max(p.values()) for p in self.probabilities]
        avg_max_prob = np.mean(max_probs)

        # Decision margin (difference between top two class probabilities)
        margins = []
        for probs in self.probabilities:
            sorted_probs = sorted(probs.values(), reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            margins.append(margin)
        avg_margin = np.mean(margins)

        # Uncertain sample ratio
        uncertain_count = sum(1 for u in self.uncertainties
                            if u.get('fuzziness', 0) > self.uncertainty_threshold)
        uncertain_ratio = uncertain_count / len(self.uncertainties)

        return {
            'average_fuzziness': float(avg_fuzziness),
            'average_specificity': float(avg_specificity),
            'average_entropy': float(avg_entropy),
            'average_max_probability': float(avg_max_prob),
            'average_decision_margin': float(avg_margin),
            'uncertain_sample_ratio': float(uncertain_ratio),
            'crisp_score_std': float(np.std(self.crisp_scores)),
        }

    def _compute_uncertainty_quality_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the quality of uncertainty estimates.

        Good uncertainty estimates should:
        - Be high when predictions are wrong
        - Be low when predictions are correct
        - Correlate with actual error rates
        """
        correct = y_true == y_pred
        fuzziness_values = np.array([u.get('fuzziness', 0) for u in self.uncertainties])

        # Uncertainty when correct vs incorrect
        uncertainty_correct = np.mean(fuzziness_values[correct]) if np.any(correct) else 0
        uncertainty_incorrect = np.mean(fuzziness_values[~correct]) if np.any(~correct) else 0

        # Uncertainty discrimination (should be positive if uncertainty is meaningful)
        uncertainty_discrimination = uncertainty_incorrect - uncertainty_correct

        # Correlation between uncertainty and error
        if len(set(correct)) > 1:  # Need both correct and incorrect
            correlation = np.corrcoef(fuzziness_values, ~correct)[0, 1]
        else:
            correlation = 0

        # Selective prediction: accuracy when uncertainty is below threshold
        low_uncertainty_mask = fuzziness_values < self.uncertainty_threshold
        if np.any(low_uncertainty_mask):
            selective_accuracy = np.mean(correct[low_uncertainty_mask])
            coverage = np.mean(low_uncertainty_mask)
        else:
            selective_accuracy = 0
            coverage = 0

        return {
            'uncertainty_when_correct': float(uncertainty_correct),
            'uncertainty_when_incorrect': float(uncertainty_incorrect),
            'uncertainty_discrimination': float(uncertainty_discrimination),
            'uncertainty_error_correlation': float(correlation) if not np.isnan(correlation) else 0,
            'selective_accuracy': float(selective_accuracy),
            'selective_coverage': float(coverage),
        }

    def _compute_calibration_metrics(self, y_true: np.ndarray) -> Dict[str, float]:
        """
        Compute probability calibration metrics.

        Calibration measures how well predicted probabilities
        match actual frequencies.
        """
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(self.probabilities)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = []
            correct_in_bin = []

            for j, probs in enumerate(self.probabilities):
                max_prob = max(probs.values())
                pred_class = max(probs, key=probs.get)

                if bin_lower <= max_prob < bin_upper:
                    in_bin.append(max_prob)
                    correct_in_bin.append(self.true_labels[j] == pred_class)

            if len(in_bin) > 0:
                avg_confidence = np.mean(in_bin)
                avg_accuracy = np.mean(correct_in_bin)
                ece += len(in_bin) / total_samples * abs(avg_confidence - avg_accuracy)

        # Maximum Calibration Error (MCE)
        mce = 0
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]

            in_bin = []
            correct_in_bin = []

            for j, probs in enumerate(self.probabilities):
                max_prob = max(probs.values())
                pred_class = max(probs, key=probs.get)

                if bin_lower <= max_prob < bin_upper:
                    in_bin.append(max_prob)
                    correct_in_bin.append(self.true_labels[j] == pred_class)

            if len(in_bin) > 0:
                avg_confidence = np.mean(in_bin)
                avg_accuracy = np.mean(correct_in_bin)
                mce = max(mce, abs(avg_confidence - avg_accuracy))

        # Brier Score (mean squared error of probability predictions)
        brier_score = 0
        for j, probs in enumerate(self.probabilities):
            true_class = self.true_labels[j]
            for class_name, prob in probs.items():
                target = 1.0 if class_name == true_class else 0.0
                brier_score += (prob - target) ** 2

        brier_score /= (len(self.probabilities) * len(self.classes))

        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'brier_score': float(brier_score),
        }

    def _analyze_samples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Analyze individual samples for error analysis.

        Returns analysis of misclassified and high-uncertainty samples.
        """
        analysis = []

        for i in range(len(self.true_labels)):
            is_correct = y_true[i] == y_pred[i]
            uncertainty = self.uncertainties[i].get('fuzziness', 0)

            # Focus on interesting cases: errors or high uncertainty
            if not is_correct or uncertainty > self.uncertainty_threshold:
                analysis.append({
                    'index': i,
                    'true_label': self.true_labels[i],
                    'predicted_label': self.predicted_labels[i],
                    'probabilities': self.probabilities[i],
                    'uncertainty': uncertainty,
                    'crisp_score': self.crisp_scores[i],
                    'is_correct': is_correct,
                    'is_uncertain': uncertainty > self.uncertainty_threshold,
                })

        return analysis

    def compare_with_baseline(
        self,
        baseline_predictions: List[str],
        baseline_name: str = 'Baseline'
    ) -> Dict[str, Any]:
        """
        Compare fuzzy classifier performance with a baseline.

        Parameters
        ----------
        baseline_predictions : list
            Predictions from baseline classifier
        baseline_name : str
            Name of baseline for reporting

        Returns
        -------
        dict
            Comparison metrics
        """
        y_true = np.array([self.class_to_idx[l] for l in self.true_labels])
        y_fuzzy = np.array([self.class_to_idx[l] for l in self.predicted_labels])
        y_baseline = np.array([self.class_to_idx[l] for l in baseline_predictions])

        fuzzy_accuracy = np.mean(y_true == y_fuzzy)
        baseline_accuracy = np.mean(y_true == y_baseline)

        # McNemar's test data
        # a: both correct, b: fuzzy correct baseline wrong
        # c: fuzzy wrong baseline correct, d: both wrong
        a = np.sum((y_true == y_fuzzy) & (y_true == y_baseline))
        b = np.sum((y_true == y_fuzzy) & (y_true != y_baseline))
        c = np.sum((y_true != y_fuzzy) & (y_true == y_baseline))
        d = np.sum((y_true != y_fuzzy) & (y_true != y_baseline))

        # McNemar's chi-squared
        if b + c > 0:
            mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            mcnemar_chi2 = 0

        return {
            'fuzzy_accuracy': float(fuzzy_accuracy),
            f'{baseline_name.lower()}_accuracy': float(baseline_accuracy),
            'accuracy_improvement': float(fuzzy_accuracy - baseline_accuracy),
            'mcnemar_chi2': float(mcnemar_chi2),
            'contingency_table': {
                'both_correct': int(a),
                'fuzzy_correct_baseline_wrong': int(b),
                'fuzzy_wrong_baseline_correct': int(c),
                'both_wrong': int(d),
            }
        }
