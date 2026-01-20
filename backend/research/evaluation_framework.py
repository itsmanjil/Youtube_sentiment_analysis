"""
Thesis-Grade Evaluation Framework for Sentiment Analysis
Implements k-fold cross-validation, statistical testing, and comprehensive metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, cohen_kappa_score
)
from scipy import stats
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime


class ThesisEvaluationFramework:
    """
    Comprehensive evaluation framework for thesis-level experiments

    Features:
    - K-fold stratified cross-validation
    - Multiple evaluation metrics
    - Statistical significance testing
    - Confidence intervals for all metrics
    - Per-class and macro/micro averages
    - Model comparison with McNemar's test
    """

    def __init__(self, n_folds: int = 10, random_state: int = 42, n_bootstrap: int = 1000):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.results = {}

    def evaluate_with_cross_validation(
        self,
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        groups: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation with comprehensive metrics

        Args:
            model_fn: Function that returns a fitted model
            X: Feature matrix
            y: True labels
            model_name: Name for logging

        Returns:
            Dictionary with all evaluation metrics and fold-wise results
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)
            splitter = GroupKFold(n_splits=self.n_folds)
            split_iter = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y)

        fold_results = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_micro': [],
            'recall_micro': [],
            'f1_micro': [],
            'cohen_kappa': [],
            'per_class_f1': {label: [] for label in np.unique(y)},
            'confusion_matrices': [],
            'predictions': [],
            'true_labels': []
        }

        print(f"\n{'='*80}")
        print(f"Evaluating {model_name} with {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*80}\n")

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            print(f"Fold {fold_idx + 1}/{self.n_folds}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model = model_fn()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Store for later statistical tests
            fold_results['predictions'].extend(y_pred)
            fold_results['true_labels'].extend(y_test)

            # Compute metrics
            fold_results['accuracy'].append(accuracy_score(y_test, y_pred))

            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                y_test, y_pred, average='micro', zero_division=0
            )

            fold_results['precision_macro'].append(precision_macro)
            fold_results['recall_macro'].append(recall_macro)
            fold_results['f1_macro'].append(f1_macro)
            fold_results['precision_micro'].append(precision_micro)
            fold_results['recall_micro'].append(recall_micro)
            fold_results['f1_micro'].append(f1_micro)

            # Cohen's Kappa (agreement beyond chance)
            fold_results['cohen_kappa'].append(cohen_kappa_score(y_test, y_pred))

            # Per-class F1 scores
            _, _, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            for label_idx, label in enumerate(np.unique(y)):
                if label_idx < len(f1_per_class):
                    fold_results['per_class_f1'][label].append(f1_per_class[label_idx])

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fold_results['confusion_matrices'].append(cm)

            print(f"  Accuracy: {fold_results['accuracy'][-1]:.4f}, "
                  f"F1-Macro: {fold_results['f1_macro'][-1]:.4f}, "
                  f"Kappa: {fold_results['cohen_kappa'][-1]:.4f}")

        # Aggregate results
        aggregated = self._aggregate_fold_results(fold_results, model_name)

        # Compute confidence intervals
        aggregated['confidence_intervals'] = self._compute_bootstrap_ci(
            np.array(fold_results['true_labels']),
            np.array(fold_results['predictions'])
        )

        self.results[model_name] = aggregated

        return aggregated

    def _aggregate_fold_results(self, fold_results: Dict, model_name: str) -> Dict[str, Any]:
        """Aggregate fold-wise results into summary statistics"""

        aggregated = {
            'model_name': model_name,
            'n_folds': self.n_folds,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        # Compute mean and std for each metric
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                       'precision_micro', 'recall_micro', 'f1_micro', 'cohen_kappa']:
            values = fold_results[metric]
            aggregated['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'fold_values': [float(v) for v in values]
            }

        # Per-class F1 scores
        aggregated['metrics']['per_class_f1'] = {}
        for label, values in fold_results['per_class_f1'].items():
            if values:
                aggregated['metrics']['per_class_f1'][str(label)] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }

        # Average confusion matrix
        avg_cm = np.mean(fold_results['confusion_matrices'], axis=0)
        aggregated['confusion_matrix'] = {
            'matrix': avg_cm.tolist(),
            'normalized': (avg_cm / avg_cm.sum(axis=1, keepdims=True)).tolist()
        }

        # Store predictions for statistical tests
        aggregated['_predictions'] = fold_results['predictions']
        aggregated['_true_labels'] = fold_results['true_labels']

        return aggregated

    def _compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dictionary with confidence intervals for each metric
        """
        n_samples = len(y_true)
        bootstrap_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'precision_macro': [],
            'recall_macro': []
        }

        np.random.seed(self.random_state)

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute metrics
            bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))

            p, r, f1, _ = precision_recall_fscore_support(
                y_true_boot, y_pred_boot, average='macro', zero_division=0
            )
            bootstrap_metrics['f1_macro'].append(f1)
            bootstrap_metrics['precision_macro'].append(p)
            bootstrap_metrics['recall_macro'].append(r)

        # Compute percentile-based confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            confidence_intervals[metric] = (float(lower), float(upper))

        return confidence_intervals

    def compare_models_mcnemar(
        self,
        model_name_1: str,
        model_name_2: str
    ) -> Dict[str, Any]:
        """
        Compare two models using McNemar's test for statistical significance

        McNemar's test checks if one model is significantly better than another
        on the same test instances (paired test).

        Args:
            model_name_1: Name of first model
            model_name_2: Name of second model

        Returns:
            Dictionary with test statistic, p-value, and interpretation
        """
        if model_name_1 not in self.results or model_name_2 not in self.results:
            raise ValueError("Both models must be evaluated first")

        y_true = np.array(self.results[model_name_1]['_true_labels'])
        y_pred_1 = np.array(self.results[model_name_1]['_predictions'])
        y_pred_2 = np.array(self.results[model_name_2]['_predictions'])

        # Build contingency table
        # n01: model 1 wrong, model 2 correct
        # n10: model 1 correct, model 2 wrong
        n01 = np.sum((y_pred_1 != y_true) & (y_pred_2 == y_true))
        n10 = np.sum((y_pred_1 == y_true) & (y_pred_2 != y_true))

        # McNemar's test statistic (with continuity correction)
        if n01 + n10 == 0:
            statistic = 0
            p_value = 1.0
        else:
            statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

        result = {
            'model_1': model_name_1,
            'model_2': model_name_2,
            'n01_m1_wrong_m2_correct': int(n01),
            'n10_m1_correct_m2_wrong': int(n10),
            'mcnemar_statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'interpretation': self._interpret_mcnemar(p_value, n01, n10, model_name_1, model_name_2)
        }

        return result

    def _interpret_mcnemar(
        self,
        p_value: float,
        n01: int,
        n10: int,
        model_1: str,
        model_2: str
    ) -> str:
        """Generate human-readable interpretation of McNemar's test"""

        if p_value >= 0.05:
            return (f"No significant difference between {model_1} and {model_2} "
                   f"(p={p_value:.4f} >= 0.05). Performance is statistically equivalent.")

        better_model = model_2 if n01 > n10 else model_1
        worse_model = model_1 if n01 > n10 else model_2

        significance = "highly significant" if p_value < 0.01 else "significant"

        return (f"{better_model} is {significance}ly better than {worse_model} "
               f"(p={p_value:.4f}). {better_model} correctly classifies "
               f"{max(n01, n10)} more instances than {worse_model}.")

    def paired_t_test(
        self,
        model_name_1: str,
        model_name_2: str
    ) -> Dict[str, Any]:
        """
        Perform paired t-test on fold-wise F1 scores

        Tests if the mean difference in F1 scores across folds is significant

        Args:
            model_name_1: Name of first model
            model_name_2: Name of second model

        Returns:
            Dictionary with t-statistic, p-value, and interpretation
        """
        if model_name_1 not in self.results or model_name_2 not in self.results:
            raise ValueError("Both models must be evaluated first")

        f1_scores_1 = self.results[model_name_1]['metrics']['f1_macro']['fold_values']
        f1_scores_2 = self.results[model_name_2]['metrics']['f1_macro']['fold_values']

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(f1_scores_1, f1_scores_2)

        mean_diff = np.mean(f1_scores_1) - np.mean(f1_scores_2)

        result = {
            'model_1': model_name_1,
            'model_2': model_name_2,
            'mean_f1_model_1': float(np.mean(f1_scores_1)),
            'mean_f1_model_2': float(np.mean(f1_scores_2)),
            'mean_difference': float(mean_diff),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'interpretation': self._interpret_t_test(
                p_value, mean_diff, model_name_1, model_name_2
            )
        }

        return result

    def _interpret_t_test(
        self,
        p_value: float,
        mean_diff: float,
        model_1: str,
        model_2: str
    ) -> str:
        """Generate human-readable interpretation of paired t-test"""

        if p_value >= 0.05:
            return (f"No significant difference in F1 scores between {model_1} and {model_2} "
                   f"(p={p_value:.4f} >= 0.05, mean diff={mean_diff:.4f}).")

        better_model = model_1 if mean_diff > 0 else model_2
        worse_model = model_2 if mean_diff > 0 else model_1

        significance = "highly significant" if p_value < 0.01 else "significant"

        return (f"{better_model} achieves {significance}ly higher F1 scores than {worse_model} "
               f"(p={p_value:.4f}, mean diff={abs(mean_diff):.4f}).")

    def generate_thesis_report(self, output_path: str = "evaluation_report.json"):
        """
        Generate comprehensive thesis-quality evaluation report

        Includes:
        - All model results
        - Statistical comparisons
        - Confidence intervals
        - Detailed metrics tables

        Args:
            output_path: Path to save JSON report
        """
        report = {
            'metadata': {
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'n_bootstrap': self.n_bootstrap,
                'generated_at': datetime.now().isoformat()
            },
            'models': {},
            'statistical_comparisons': {}
        }

        # Add all model results (remove internal predictions)
        for model_name, results in self.results.items():
            clean_results = {k: v for k, v in results.items()
                           if not k.startswith('_')}
            report['models'][model_name] = clean_results

        # Pairwise model comparisons
        model_names = list(self.results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_1 = model_names[i]
                model_2 = model_names[j]

                comparison_key = f"{model_1}_vs_{model_2}"

                report['statistical_comparisons'][comparison_key] = {
                    'mcnemar_test': self.compare_models_mcnemar(model_1, model_2),
                    'paired_t_test': self.paired_t_test(model_1, model_2)
                }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Thesis Evaluation Report saved to: {output_path}")
        print(f"{'='*80}\n")

        # Print summary table
        self._print_summary_table()

        return report

    def _print_summary_table(self):
        """Print beautiful ASCII table with all model results"""

        print("\n" + "="*120)
        print("THESIS EVALUATION RESULTS SUMMARY".center(120))
        print("="*120 + "\n")

        # Header
        print(f"{'Model':<25} {'Acc':<12} {'P-Macro':<12} {'R-Macro':<12} {'F1-Macro':<12} {'Kappa':<12}")
        print("-" * 120)

        # Rows
        for model_name, results in self.results.items():
            metrics = results['metrics']

            acc = metrics['accuracy']
            p_macro = metrics['precision_macro']
            r_macro = metrics['recall_macro']
            f1_macro = metrics['f1_macro']
            kappa = metrics['cohen_kappa']

            print(f"{model_name:<25} "
                  f"{acc['mean']:.4f}±{acc['std']:.4f}  "
                  f"{p_macro['mean']:.4f}±{p_macro['std']:.4f}  "
                  f"{r_macro['mean']:.4f}±{r_macro['std']:.4f}  "
                  f"{f1_macro['mean']:.4f}±{f1_macro['std']:.4f}  "
                  f"{kappa['mean']:.4f}±{kappa['std']:.4f}")

        print("="*120 + "\n")


# Example usage
if __name__ == "__main__":
    """
    Example demonstrating thesis-level evaluation

    Usage:
        python evaluation_framework.py
    """

    # Mock data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    print("Generating synthetic sentiment data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=50,
        n_classes=3,
        random_state=42
    )

    # Initialize evaluation framework
    evaluator = ThesisEvaluationFramework(n_folds=10, random_state=42, n_bootstrap=1000)

    # Evaluate multiple models
    print("\n" + "="*80)
    print("EVALUATING BASELINE: Logistic Regression")
    print("="*80)
    evaluator.evaluate_with_cross_validation(
        model_fn=lambda: LogisticRegression(max_iter=1000, random_state=42),
        X=X,
        y=y,
        model_name="Logistic_Regression"
    )

    print("\n" + "="*80)
    print("EVALUATING ADVANCED: Random Forest")
    print("="*80)
    evaluator.evaluate_with_cross_validation(
        model_fn=lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        X=X,
        y=y,
        model_name="Random_Forest"
    )

    # Generate thesis report
    evaluator.generate_thesis_report(output_path="thesis_evaluation_report.json")

    print("\n✅ Evaluation complete! Check 'thesis_evaluation_report.json' for full results.")
