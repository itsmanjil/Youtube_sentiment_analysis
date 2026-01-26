#!/usr/bin/env python3
"""
Benchmark Datasets Demo for Thesis

Demonstrates the benchmark evaluation framework on standard
sentiment analysis datasets.

Usage:
    python -m research.benchmarks.demo

Author: [Your Name]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict, Any

from research.benchmarks.base import (
    Dataset,
    DatasetSplit,
    DatasetManager,
    SentimentLabel,
)
from research.benchmarks.evaluation import (
    BenchmarkEvaluator,
    CrossDomainEvaluator,
    EvaluationMetrics,
    BenchmarkReport,
)


# =============================================================================
# Mock Model for Testing
# =============================================================================

class MockSentimentModel:
    """
    Simple mock model for testing the benchmark framework.

    Uses keyword matching for basic sentiment classification.
    """

    def __init__(self, name: str = "MockModel"):
        self.name = name
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'loved', 'best', 'perfect', 'beautiful', 'brilliant',
            'awesome', 'outstanding', 'superb', 'incredible', 'masterpiece'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
            'boring', 'waste', 'disappointing', 'poor', 'stupid', 'dumb',
            'garbage', 'trash', 'pathetic', 'disaster', 'failure'
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        words = set(text.lower().split())

        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)

        if pos_count > neg_count:
            label = 'Positive'
            confidence = min(0.9, 0.5 + 0.1 * (pos_count - neg_count))
        elif neg_count > pos_count:
            label = 'Negative'
            confidence = min(0.9, 0.5 + 0.1 * (neg_count - pos_count))
        else:
            label = 'Neutral'
            confidence = 0.5

        return {
            'label': label,
            'confidence': confidence,
            'probs': {
                'Positive': confidence if label == 'Positive' else (1 - confidence) / 2,
                'Negative': confidence if label == 'Negative' else (1 - confidence) / 2,
                'Neutral': confidence if label == 'Neutral' else (1 - confidence) / 2,
            }
        }

    def predict(self, text: str) -> str:
        """Simple predict interface."""
        return self.analyze(text)['label']


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing benchmark framework.

    Generates controlled test data for validation.
    """

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        super().__init__(
            name='synthetic',
            data_dir=None,
            download=False,
            cache=False
        )
        self.n_samples = n_samples
        self.seed = seed
        self.description = "Synthetic dataset for testing"

    def _data_exists(self) -> bool:
        return True

    def _load_data(self):
        """Generate synthetic data."""
        np.random.seed(self.seed)

        positive_templates = [
            "This is an excellent product, highly recommended!",
            "Amazing quality, I love it!",
            "Great experience, wonderful service!",
            "The best purchase I've ever made!",
            "Fantastic movie, brilliant acting!",
        ]

        negative_templates = [
            "Terrible quality, waste of money.",
            "Awful experience, horrible service!",
            "The worst product ever, hate it!",
            "Boring and disappointing, not recommended.",
            "Poor performance, total disaster!",
        ]

        neutral_templates = [
            "It's okay, nothing special.",
            "Average product, does the job.",
            "Neither good nor bad.",
            "Standard quality, as expected.",
            "Typical experience, nothing notable.",
        ]

        texts = []
        labels = []

        for i in range(self.n_samples):
            r = np.random.random()
            if r < 0.4:
                texts.append(np.random.choice(positive_templates))
                labels.append(SentimentLabel.POSITIVE)
            elif r < 0.8:
                texts.append(np.random.choice(negative_templates))
                labels.append(SentimentLabel.NEGATIVE)
            else:
                texts.append(np.random.choice(neutral_templates))
                labels.append(SentimentLabel.NEUTRAL)

        # Split into train/val/test (60/20/20)
        n_train = int(0.6 * self.n_samples)
        n_val = int(0.2 * self.n_samples)

        train_split = DatasetSplit(
            texts=texts[:n_train],
            labels=labels[:n_train],
            name='train'
        )

        val_split = DatasetSplit(
            texts=texts[n_train:n_train + n_val],
            labels=labels[n_train:n_train + n_val],
            name='val'
        )

        test_split = DatasetSplit(
            texts=texts[n_train + n_val:],
            labels=labels[n_train + n_val:],
            name='test'
        )

        return train_split, val_split, test_split


# =============================================================================
# Demo Functions
# =============================================================================

def demo_dataset_basics():
    """Demo basic dataset operations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Dataset Basics")
    print("=" * 60)

    # Create synthetic dataset
    dataset = SyntheticDataset(n_samples=500)
    dataset.load()

    print(f"\nDataset: {dataset.name}")
    print(f"Description: {dataset.description}")

    print(f"\nSplits:")
    print(f"  Train: {len(dataset.train)} samples")
    print(f"  Val:   {len(dataset.val)} samples")
    print(f"  Test:  {len(dataset.test)} samples")

    print(f"\nLabel distribution (train):")
    dist = dataset.train.get_label_distribution()
    for label, count in dist.items():
        pct = 100 * count / len(dataset.train)
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Sample some data
    print(f"\nSample texts from test set:")
    for i, (text, label) in enumerate(dataset.test):
        if i >= 3:
            break
        print(f"  [{label.to_string():8s}] {text[:50]}...")


def demo_dataset_split():
    """Demo DatasetSplit operations."""
    print("\n" + "=" * 60)
    print("DEMO 2: DatasetSplit Operations")
    print("=" * 60)

    dataset = SyntheticDataset(n_samples=1000)
    dataset.load()

    print(f"\nOriginal test size: {len(dataset.test)}")

    # Sample subset
    sampled = dataset.test.sample(n=50, stratified=True)
    print(f"Sampled size: {len(sampled)}")

    print(f"\nStratified sample distribution:")
    dist = sampled.get_label_distribution()
    for label, count in dist.items():
        print(f"  {label}: {count}")

    # Get texts by label
    positive_texts = dataset.test.get_texts_by_label(SentimentLabel.POSITIVE)
    print(f"\nPositive texts in test set: {len(positive_texts)}")

    # Convert to arrays
    texts, labels = dataset.test.to_arrays()
    print(f"\nArray shapes: texts={len(texts)}, labels={labels.shape}")


def demo_single_evaluation():
    """Demo single dataset evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Single Dataset Evaluation")
    print("=" * 60)

    # Create model and dataset
    model = MockSentimentModel()
    dataset = SyntheticDataset(n_samples=500)

    # Create evaluator
    evaluator = BenchmarkEvaluator()
    evaluator.register_dataset('synthetic', dataset)

    print(f"\nEvaluating {model.name} on registered datasets...")

    report = evaluator.evaluate(
        model=model,
        model_name=model.name,
        max_samples=100,  # Limit for demo
        verbose=True
    )

    print(f"\n{report.summary()}")


def demo_cross_domain():
    """Demo cross-domain evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Cross-Domain Evaluation")
    print("=" * 60)

    # Create multiple synthetic datasets with different characteristics
    dataset1 = SyntheticDataset(n_samples=300, seed=42)
    dataset1.name = "domain_a"

    dataset2 = SyntheticDataset(n_samples=300, seed=123)
    dataset2.name = "domain_b"

    print("\nCross-domain evaluation simulates training on one domain")
    print("and testing on another to measure generalization.")

    # For demo, we'll use pre-trained mock model
    model = MockSentimentModel()

    # Create evaluator and register datasets
    evaluator = BenchmarkEvaluator()
    evaluator.register_dataset('domain_a', dataset1)
    evaluator.register_dataset('domain_b', dataset2)

    # Evaluate on both
    report = evaluator.evaluate(
        model=model,
        model_name="MockModel",
        max_samples=50,
        verbose=True
    )

    print(f"\nResults by domain:")
    for domain, metrics in report.dataset_results.items():
        print(f"  {domain}: Accuracy={metrics.accuracy:.4f}, F1={metrics.f1_macro:.4f}")


def demo_latex_report():
    """Demo LaTeX report generation."""
    print("\n" + "=" * 60)
    print("DEMO 5: LaTeX Report Generation")
    print("=" * 60)

    # Create mock evaluation results
    model = MockSentimentModel()

    dataset1 = SyntheticDataset(n_samples=200, seed=42)
    dataset1.name = "twitter"

    dataset2 = SyntheticDataset(n_samples=200, seed=123)
    dataset2.name = "imdb"

    evaluator = BenchmarkEvaluator()
    evaluator.register_dataset('twitter', dataset1)
    evaluator.register_dataset('imdb', dataset2)

    report = evaluator.evaluate(
        model=model,
        model_name="FuzzySentiment",
        max_samples=50,
        verbose=False
    )

    print("\nGenerated LaTeX table for thesis:")
    print("-" * 50)
    print(report.to_latex_table())
    print("-" * 50)

    print("\nThis can be directly included in your thesis document!")


def demo_metrics_detail():
    """Demo detailed metrics computation."""
    print("\n" + "=" * 60)
    print("DEMO 6: Detailed Metrics")
    print("=" * 60)

    model = MockSentimentModel()
    dataset = SyntheticDataset(n_samples=300)

    evaluator = BenchmarkEvaluator()
    evaluator.register_dataset('test', dataset)

    report = evaluator.evaluate(
        model=model,
        model_name="MockModel",
        max_samples=100,
        verbose=False
    )

    metrics = report.dataset_results['test']

    print(f"\nDetailed Metrics for Test Dataset:")
    print(f"-" * 40)
    print(f"Accuracy:           {metrics.accuracy:.4f}")
    print(f"")
    print(f"Macro Averages:")
    print(f"  Precision:        {metrics.precision_macro:.4f}")
    print(f"  Recall:           {metrics.recall_macro:.4f}")
    print(f"  F1:               {metrics.f1_macro:.4f}")
    print(f"")
    print(f"Weighted Averages:")
    print(f"  Precision:        {metrics.precision_weighted:.4f}")
    print(f"  Recall:           {metrics.recall_weighted:.4f}")
    print(f"  F1:               {metrics.f1_weighted:.4f}")
    print(f"")
    print(f"Per-Class F1:")
    for class_name, f1 in metrics.per_class_f1.items():
        print(f"  {class_name:10s}: {f1:.4f}")
    print(f"")
    print(f"Inference Time:     {metrics.inference_time_ms:.2f} ms/sample")
    print(f"Total Samples:      {metrics.n_samples}")

    print(f"\nConfusion Matrix:")
    cm = metrics.confusion_matrix
    print(f"              Pred_Neg  Pred_Neu  Pred_Pos")
    labels = ['Negative', 'Neutral ', 'Positive']
    for i, label in enumerate(labels):
        row = [f"{int(cm[i, j]):8d}" for j in range(min(3, cm.shape[1]))]
        print(f"  True_{label} {' '.join(row)}")


def demo_dataset_manager():
    """Demo DatasetManager for managing multiple datasets."""
    print("\n" + "=" * 60)
    print("DEMO 7: Dataset Manager")
    print("=" * 60)

    manager = DatasetManager()

    # Register multiple datasets
    manager.register('synthetic_1', SyntheticDataset(n_samples=200, seed=1))
    manager.register('synthetic_2', SyntheticDataset(n_samples=200, seed=2))
    manager.register('synthetic_3', SyntheticDataset(n_samples=200, seed=3))

    print(f"\nRegistered datasets: {manager.list_datasets()}")

    # Load all
    print("\nLoading all datasets...")
    manager.load_all()

    # Get summary
    print("\nDataset Summary:")
    summary = manager.get_summary()
    for name, info in summary.items():
        print(f"\n  {name}:")
        print(f"    Train size: {info.get('train_size', 'N/A')}")
        print(f"    Test size:  {info.get('test_size', 'N/A')}")
        if 'train_distribution' in info:
            print(f"    Distribution: {info['train_distribution']}")


def demo_json_export():
    """Demo JSON export for results archiving."""
    print("\n" + "=" * 60)
    print("DEMO 8: JSON Export")
    print("=" * 60)

    model = MockSentimentModel()
    dataset = SyntheticDataset(n_samples=200)

    evaluator = BenchmarkEvaluator()
    evaluator.register_dataset('test', dataset)

    report = evaluator.evaluate(
        model=model,
        model_name="MockModel",
        max_samples=50,
        verbose=False
    )

    # Export to JSON
    json_str = report.to_json()

    print("\nJSON Export (truncated):")
    print("-" * 50)
    # Show first 800 chars
    print(json_str[:800] + "...")
    print("-" * 50)

    print("\nThis can be saved for reproducibility and comparison!")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("BENCHMARK FRAMEWORK DEMONSTRATION")
    print("Standard Datasets for Sentiment Analysis Evaluation")
    print("=" * 60)

    demo_dataset_basics()
    demo_dataset_split()
    demo_single_evaluation()
    demo_cross_domain()
    demo_latex_report()
    demo_metrics_detail()
    demo_dataset_manager()
    demo_json_export()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
    Summary of Benchmark Framework:

    1. Dataset Classes
       - Dataset: Abstract base for sentiment datasets
       - DatasetSplit: Train/val/test split with utilities
       - DatasetManager: Manage multiple datasets
       - SentimentLabel: Standardized labels (Neg/Neu/Pos)

    2. Standard Datasets (when downloaded):
       - Sentiment140: 1.6M Twitter tweets
       - IMDB: 50K movie reviews
       - Amazon Reviews: Product reviews
       - SST: Stanford Sentiment Treebank

    3. Evaluation Tools:
       - BenchmarkEvaluator: Single/multi-dataset evaluation
       - CrossDomainEvaluator: Domain transfer evaluation
       - BenchmarkReport: Results with LaTeX export

    4. Metrics Computed:
       - Accuracy
       - Precision (macro/weighted)
       - Recall (macro/weighted)
       - F1 Score (macro/weighted/per-class)
       - Confusion Matrix
       - Inference Time

    5. Thesis Integration:
       - LaTeX table generation
       - JSON export for reproducibility
       - Cross-domain analysis
       - Statistical comparison support

    Usage for Your Thesis:
    ----------------------
    1. Load standard benchmarks: Sentiment140, IMDB, Amazon, SST
    2. Evaluate your FuzzySentiment model against baselines
    3. Run cross-domain tests (train on Twitter, test on IMDB)
    4. Generate LaTeX tables for results chapter
    5. Export JSON for supplementary materials
    """)


if __name__ == '__main__':
    main()
