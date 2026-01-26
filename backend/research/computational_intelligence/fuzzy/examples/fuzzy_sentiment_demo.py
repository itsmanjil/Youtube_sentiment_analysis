#!/usr/bin/env python3
"""
Fuzzy Sentiment Classification Demo

This script demonstrates the complete fuzzy sentiment classification
system for thesis-level experiments.

Run this script to:
1. See fuzzy membership functions in action
2. Understand fuzzy inference with sentiment rules
3. Compare defuzzification methods
4. Evaluate uncertainty quantification

Usage:
    python fuzzy_sentiment_demo.py

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root / 'backend'))

import numpy as np
import json

# Import fuzzy modules
from research.computational_intelligence.fuzzy import (
    FuzzySentimentClassifier,
    FuzzyEvaluator,
    TriangularMF,
    GaussianMF,
    TrapezoidalMF,
    create_three_class_mfs,
)
from research.computational_intelligence.fuzzy.defuzzification import (
    compare_defuzzification_methods,
    compute_uncertainty_metrics,
)


def demo_membership_functions():
    """Demonstrate different membership function types."""
    print("\n" + "=" * 60)
    print("DEMO 1: Membership Functions")
    print("=" * 60)

    # Create sample sentiment score
    test_scores = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\nTriangular Membership Functions:")
    print("-" * 40)
    tri_mfs = create_three_class_mfs(mf_type='triangular')
    for score in test_scores:
        memberships = {name: f"{mf(score):.3f}" for name, mf in tri_mfs.items()}
        print(f"  Score {score:.2f}: {memberships}")

    print("\nGaussian Membership Functions:")
    print("-" * 40)
    gauss_mfs = create_three_class_mfs(mf_type='gaussian')
    for score in test_scores:
        memberships = {name: f"{mf(score):.3f}" for name, mf in gauss_mfs.items()}
        print(f"  Score {score:.2f}: {memberships}")

    print("\nKey Insight: Gaussian MFs have smoother transitions and")
    print("overlapping regions, better for handling uncertainty.")


def demo_fuzzy_inference():
    """Demonstrate fuzzy inference with multiple model inputs."""
    print("\n" + "=" * 60)
    print("DEMO 2: Fuzzy Inference System")
    print("=" * 60)

    # Create classifier
    classifier = FuzzySentimentClassifier(
        base_models=['model_a', 'model_b', 'model_c'],
        mf_type='gaussian',
        defuzz_method='centroid'
    )

    # Scenario 1: All models agree (positive)
    print("\nScenario 1: All models agree on POSITIVE")
    print("-" * 40)
    outputs = {
        'model_a': {'positive': 0.85, 'neutral': 0.10, 'negative': 0.05},
        'model_b': {'positive': 0.80, 'neutral': 0.15, 'negative': 0.05},
        'model_c': {'positive': 0.90, 'neutral': 0.08, 'negative': 0.02},
    }
    result = classifier.classify(outputs)
    print(f"  Label: {result.label}")
    print(f"  Score: {result.crisp_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Fuzziness: {result.uncertainty_metrics['fuzziness']:.4f}")

    # Scenario 2: Models disagree
    print("\nScenario 2: Models DISAGREE (conflict)")
    print("-" * 40)
    outputs = {
        'model_a': {'positive': 0.70, 'neutral': 0.20, 'negative': 0.10},
        'model_b': {'positive': 0.20, 'neutral': 0.30, 'negative': 0.50},
        'model_c': {'positive': 0.40, 'neutral': 0.50, 'negative': 0.10},
    }
    result = classifier.classify(outputs)
    print(f"  Label: {result.label}")
    print(f"  Score: {result.crisp_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Fuzziness: {result.uncertainty_metrics['fuzziness']:.4f}")
    print(f"  -> Higher fuzziness indicates model disagreement!")

    # Scenario 3: Ambiguous sentiment
    print("\nScenario 3: Ambiguous sentiment (near boundary)")
    print("-" * 40)
    outputs = {
        'model_a': {'positive': 0.40, 'neutral': 0.35, 'negative': 0.25},
        'model_b': {'positive': 0.35, 'neutral': 0.40, 'negative': 0.25},
        'model_c': {'positive': 0.45, 'neutral': 0.30, 'negative': 0.25},
    }
    result = classifier.classify(outputs)
    print(f"  Label: {result.label}")
    print(f"  Score: {result.crisp_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Fuzziness: {result.uncertainty_metrics['fuzziness']:.4f}")


def demo_defuzzification_comparison():
    """Compare different defuzzification methods."""
    print("\n" + "=" * 60)
    print("DEMO 3: Defuzzification Methods Comparison")
    print("=" * 60)

    # Create a sample aggregated fuzzy output
    universe = np.linspace(0, 1, 100)

    # Bimodal membership (simulating model disagreement)
    membership = (
        0.7 * np.exp(-((universe - 0.3) ** 2) / 0.02) +  # Negative peak
        0.5 * np.exp(-((universe - 0.75) ** 2) / 0.02)   # Positive peak
    )

    print("\nBimodal fuzzy output (simulating model disagreement):")
    print(f"  Peak 1: score=0.30 (negative region)")
    print(f"  Peak 2: score=0.75 (positive region)")

    print("\nDefuzzification Results:")
    print("-" * 40)
    results = compare_defuzzification_methods(universe, membership)
    for method, value in results.items():
        print(f"  {method:20s}: {value:.4f}")

    print("\nInterpretation:")
    print("  - Centroid: Weighted average (considers both peaks)")
    print("  - MOM: Mean of maximum (focuses on highest peak)")
    print("  - SOM/LOM: Extreme values of maximum region")

    # Uncertainty metrics
    print("\nUncertainty Metrics:")
    print("-" * 40)
    uncertainty = compute_uncertainty_metrics(universe, membership)
    for metric, value in uncertainty.items():
        print(f"  {metric:20s}: {value:.4f}")


def demo_evaluation_metrics():
    """Demonstrate fuzzy evaluation metrics."""
    print("\n" + "=" * 60)
    print("DEMO 4: Fuzzy Evaluation Metrics")
    print("=" * 60)

    # Create classifier and evaluator
    classifier = FuzzySentimentClassifier(
        base_models=['model_a', 'model_b'],
        mf_type='gaussian',
        defuzz_method='centroid'
    )
    evaluator = FuzzyEvaluator()

    # Simulate some predictions
    test_cases = [
        # (true_label, model_outputs)
        ('Positive', {'model_a': {'positive': 0.9, 'neutral': 0.08, 'negative': 0.02},
                      'model_b': {'positive': 0.85, 'neutral': 0.10, 'negative': 0.05}}),
        ('Positive', {'model_a': {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1},
                      'model_b': {'positive': 0.75, 'neutral': 0.15, 'negative': 0.1}}),
        ('Negative', {'model_a': {'positive': 0.1, 'neutral': 0.2, 'negative': 0.7},
                      'model_b': {'positive': 0.05, 'neutral': 0.15, 'negative': 0.8}}),
        ('Neutral', {'model_a': {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2},
                      'model_b': {'positive': 0.25, 'neutral': 0.55, 'negative': 0.2}}),
        ('Positive', {'model_a': {'positive': 0.4, 'neutral': 0.4, 'negative': 0.2},
                      'model_b': {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}}),  # Ambiguous
    ]

    print("\nProcessing test samples...")
    for true_label, outputs in test_cases:
        result = classifier.classify(outputs)
        evaluator.add_sample(
            true_label=true_label,
            predicted_label=result.label,
            probabilities=result.probabilities,
            uncertainty=result.uncertainty_metrics,
            crisp_score=result.crisp_score
        )

    # Compute metrics
    evaluation = evaluator.compute_metrics()

    print("\nStandard Metrics:")
    print("-" * 40)
    for metric, value in evaluation.standard_metrics.items():
        print(f"  {metric:25s}: {value:.4f}")

    print("\nFuzzy-Specific Metrics:")
    print("-" * 40)
    for metric, value in evaluation.fuzzy_metrics.items():
        print(f"  {metric:25s}: {value:.4f}")

    print("\nUncertainty Quality Metrics:")
    print("-" * 40)
    for metric, value in evaluation.uncertainty_metrics.items():
        print(f"  {metric:30s}: {value:.4f}")


def demo_thesis_experiment():
    """Demonstrate a complete thesis experiment workflow."""
    print("\n" + "=" * 60)
    print("DEMO 5: Complete Thesis Experiment Workflow")
    print("=" * 60)

    print("""
    This demonstrates the experimental workflow for your thesis:

    1. HYPOTHESIS:
       "Fuzzy sentiment classification provides better uncertainty
       quantification than traditional ML approaches."

    2. EXPERIMENTAL SETUP:
       - Base models: LogReg, SVM, TF-IDF (existing engines)
       - Fuzzy configuration: Gaussian MFs, Centroid defuzzification
       - Dataset: YouTube comments (collected via API)
       - Evaluation: Standard metrics + Fuzzy metrics

    3. METRICS TO REPORT:
       Standard:
       - Accuracy, Precision, Recall, F1 (macro/weighted)
       - Confusion matrix

       Fuzzy-Specific:
       - Average fuzziness index
       - Uncertainty discrimination
       - Selective accuracy at confidence threshold
       - Expected Calibration Error (ECE)

    4. STATISTICAL TESTS:
       - McNemar's test (vs baseline)
       - Wilcoxon signed-rank test (fold comparison)
       - Friedman test (multiple model comparison)

    5. VISUALIZATION:
       - Membership function plots
       - Fuzzy confusion matrix
       - Uncertainty vs accuracy scatter
       - Calibration plots
    """)

    # Generate sample LaTeX table
    print("\nSample LaTeX Output for Thesis:")
    print("-" * 40)

    # Create a mock evaluation result
    classifier = FuzzySentimentClassifier(
        base_models=['logreg', 'svm'],
        mf_type='gaussian'
    )
    evaluator = FuzzyEvaluator()

    # Add mock data
    for _ in range(20):
        outputs = {
            'logreg': {'positive': np.random.random(), 'neutral': np.random.random(), 'negative': np.random.random()},
            'svm': {'positive': np.random.random(), 'neutral': np.random.random(), 'negative': np.random.random()},
        }
        # Normalize
        for model in outputs:
            total = sum(outputs[model].values())
            outputs[model] = {k: v/total for k, v in outputs[model].items()}

        result = classifier.classify(outputs)
        true_label = np.random.choice(['Positive', 'Neutral', 'Negative'])
        evaluator.add_sample(
            true_label=true_label,
            predicted_label=result.label,
            probabilities=result.probabilities,
            uncertainty=result.uncertainty_metrics,
            crisp_score=result.crisp_score
        )

    evaluation = evaluator.compute_metrics()
    print(evaluation.to_latex_table())


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("FUZZY SENTIMENT CLASSIFICATION DEMONSTRATION")
    print("For Master's Thesis in Computational Intelligence")
    print("=" * 60)

    demo_membership_functions()
    demo_fuzzy_inference()
    demo_defuzzification_comparison()
    demo_evaluation_metrics()
    demo_thesis_experiment()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
    Next Steps for Your Thesis:

    1. Run experiments on full YouTube dataset
    2. Compare with baseline models (without fuzzy layer)
    3. Perform statistical significance testing
    4. Generate visualizations for thesis
    5. Write up methodology and results chapters

    Files created:
    - membership_functions.py: MF implementations
    - fuzzy_inference.py: Fuzzy inference system
    - fuzzy_sentiment.py: Main classifier
    - defuzzification.py: Defuzzification methods
    - fuzzy_evaluation.py: Evaluation metrics
    - engine_integration.py: Integration with existing engines
    """)


if __name__ == '__main__':
    main()
