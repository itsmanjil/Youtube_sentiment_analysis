"""
Transformer Architectures for Sentiment Analysis.

This package provides transformer-based models for sentiment classification:
- BERTSentimentClassifier: BERT-based classifier
- RoBERTaSentimentClassifier: RoBERTa-based classifier (planned)

These models leverage pre-trained language models and fine-tune them
for sentiment classification on YouTube comments.
"""

from .bert_classifier import BERTSentimentClassifier

__all__ = [
    "BERTSentimentClassifier",
]
