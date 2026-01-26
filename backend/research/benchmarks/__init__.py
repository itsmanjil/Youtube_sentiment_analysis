"""
Benchmark Datasets Module for Sentiment Analysis Thesis

This module provides standardized access to benchmark datasets
for evaluating sentiment analysis models.

Features:
- Unified loading interface
- Automatic download and caching
- Train/val/test splits
- Cross-domain evaluation support
- Stratified sampling

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from .base import (
    Dataset,
    DatasetSplit,
    DatasetManager,
    SentimentLabel,
)
from .evaluation import (
    BenchmarkEvaluator,
    CrossDomainEvaluator,
    BenchmarkReport,
)

__all__ = [
    # Base classes
    'Dataset',
    'DatasetSplit',
    'DatasetManager',
    'SentimentLabel',
    # Evaluation
    'BenchmarkEvaluator',
    'CrossDomainEvaluator',
    'BenchmarkReport',
]
