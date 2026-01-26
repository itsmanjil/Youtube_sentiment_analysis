"""
Benchmark Datasets Module for Sentiment Analysis Thesis

This module provides standardized access to benchmark datasets
for evaluating sentiment analysis models.

Supported Datasets:
1. Sentiment140 (Twitter) - 1.6M tweets
2. IMDB Reviews - 50K movie reviews
3. Amazon Reviews - Product reviews
4. SST-2 (Stanford Sentiment Treebank) - Movie phrases

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
from .datasets import (
    Sentiment140Dataset,
    IMDBDataset,
    AmazonReviewsDataset,
    SSTDataset,
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
    # Datasets
    'Sentiment140Dataset',
    'IMDBDataset',
    'AmazonReviewsDataset',
    'SSTDataset',
    # Evaluation
    'BenchmarkEvaluator',
    'CrossDomainEvaluator',
    'BenchmarkReport',
]
