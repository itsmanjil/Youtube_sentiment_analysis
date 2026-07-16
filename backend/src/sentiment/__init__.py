"""
Sentiment Analysis Engines Package.

This package provides a unified interface for sentiment analysis using
multiple model architectures:

- Classical ML: TF-IDF + Naive Bayes, Logistic Regression, SVM
- Deep Learning: Hybrid CNN-BiLSTM-Attention
- Transformers: BERT-based classifiers
- Ensemble Methods: Weighted voting, Meta-learner stacking

Usage
-----
>>> from src.sentiment import get_sentiment_engine, SentimentResult
>>> engine = get_sentiment_engine('logreg')
>>> result = engine.analyze("This video is amazing!")
>>> print(result.label, result.score)
Positive 0.92

Available Engines
-----------------
- 'tfidf': TF-IDF + Multinomial Naive Bayes
- 'logreg': TF-IDF + Logistic Regression
- 'svm': TF-IDF + Linear SVM
- 'hybrid_dl': CNN-BiLSTM-Attention (requires PyTorch)
- 'ensemble': Weighted soft voting ensemble
- 'meta_learner': Stacked ensemble with meta-classifier
- 'bert': BERT-based transformer (requires transformers library)
"""

from .base import (
    SentimentResult,
    normalize_label,
    coerce_sentiment_result,
)
from .factory import (
    get_sentiment_engine,
    get_base_engine,
    list_available_engines,
)

__all__ = [
    # Core classes
    "SentimentResult",
    # Utility functions
    "normalize_label",
    "coerce_sentiment_result",
    # Factory functions
    "get_sentiment_engine",
    "get_base_engine",
    "list_available_engines",
]
