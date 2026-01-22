"""
Sentiment Analysis Engine Implementations.

This package contains individual implementations of sentiment analysis engines:

- tfidf_engine: TF-IDF + Multinomial Naive Bayes
- logreg_engine: TF-IDF + Logistic Regression
- svm_engine: TF-IDF + Linear SVM
- ensemble_engine: Weighted soft voting
- meta_learner_engine: Stacked ensemble
- hybrid_dl_engine: CNN-BiLSTM-Attention
- transformer_engine: BERT-based classifier
"""

from .tfidf_engine import TFIDFSentimentEngine
from .logreg_engine import LogRegSentimentEngine
from .svm_engine import SVMSentimentEngine
from .ensemble_engine import EnsembleSentimentEngine
from .meta_learner_engine import MetaLearnerSentimentEngine

__all__ = [
    "TFIDFSentimentEngine",
    "LogRegSentimentEngine",
    "SVMSentimentEngine",
    "EnsembleSentimentEngine",
    "MetaLearnerSentimentEngine",
]

# Lazy imports for optional dependencies
def get_hybrid_dl_engine():
    """Get HybridDLSentimentEngine (requires PyTorch)."""
    from .hybrid_dl_engine import HybridDLSentimentEngine
    return HybridDLSentimentEngine

def get_transformer_engine():
    """Get TransformerSentimentEngine (requires transformers)."""
    from .transformer_engine import TransformerSentimentEngine
    return TransformerSentimentEngine
