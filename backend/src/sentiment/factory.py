"""
Sentiment Engine Factory.

This module provides factory functions for creating sentiment analysis
engines.

Usage
-----
>>> from src.sentiment import get_sentiment_engine
>>> engine = get_sentiment_engine('logreg')
>>> result = engine.analyze("Great video!")

Available Engines
-----------------
- 'tfidf': TF-IDF + Multinomial Naive Bayes (fastest)
- 'logreg': TF-IDF + Logistic Regression (good calibration)
- 'svm': TF-IDF + Linear SVM (highest accuracy among classical)
- 'ensemble': Weighted soft voting (combines classical models)
- 'meta_learner': Stacked ensemble (learns combination rules)
- 'fuzzy_ensemble': Fuzzy inference ensemble (uncertainty-aware)
"""

from typing import Any, List

from .engines.tfidf_engine import TFIDFSentimentEngine
from .engines.logreg_engine import LogRegSentimentEngine
from .engines.svm_engine import SVMSentimentEngine
from .engines.ensemble_engine import EnsembleSentimentEngine
from .engines.meta_learner_engine import MetaLearnerSentimentEngine
from .engines.fuzzy_engine import FuzzyEnsembleSentimentEngine

# Registry of available engines
_ENGINE_REGISTRY = {
    "tfidf": TFIDFSentimentEngine,
    "logreg": LogRegSentimentEngine,
    "svm": SVMSentimentEngine,
    "ensemble": EnsembleSentimentEngine,
    "ci_ensemble": EnsembleSentimentEngine,  # Alias
    "meta_learner": MetaLearnerSentimentEngine,
    "stacking": MetaLearnerSentimentEngine,  # Alias
    "fuzzy_ensemble": FuzzyEnsembleSentimentEngine,
    "fuzzy": FuzzyEnsembleSentimentEngine,  # Alias
}

# Base engines (exclude ensemble methods)
_BASE_ENGINE_REGISTRY = {
    "tfidf": TFIDFSentimentEngine,
    "logreg": LogRegSentimentEngine,
    "svm": SVMSentimentEngine,
}


def list_available_engines() -> List[str]:
    """
    List all available sentiment engine types.

    Returns
    -------
    List[str]
        List of engine type names.

    Examples
    --------
    >>> list_available_engines()
    ['ensemble', 'fuzzy_ensemble', 'logreg', 'meta_learner', 'svm', 'tfidf']
    """
    engines = list(_ENGINE_REGISTRY.keys())

    # Remove aliases for cleaner output
    engines = [e for e in engines if e not in ("ci_ensemble", "stacking", "fuzzy")]
    return sorted(set(engines))


def get_sentiment_engine(engine_type: str = "logreg", **kwargs) -> Any:
    """
    Create a sentiment analysis engine of the specified type.

    Parameters
    ----------
    engine_type : str, optional
        Type of engine to create. Options:
        - 'tfidf': TF-IDF + Naive Bayes
        - 'logreg': TF-IDF + Logistic Regression (default)
        - 'svm': TF-IDF + Linear SVM
        - 'ensemble': Weighted soft voting
        - 'meta_learner': Stacked ensemble
        - 'fuzzy_ensemble': Fuzzy inference ensemble (uncertainty-aware)
            **kwargs
        Additional arguments passed to the engine constructor.

    Returns
    -------
    BaseSentimentEngine
        Initialized sentiment engine.

    Raises
    ------
    ValueError
        If engine_type is not recognized.
    ImportError
        If required dependencies are not installed.

    Examples
    --------
    >>> # Default Logistic Regression engine
    >>> engine = get_sentiment_engine()

    >>> # SVM with custom model path
    >>> engine = get_sentiment_engine('svm', model_path='./custom/model.sav')

    >>> # Ensemble with custom weights
    >>> engine = get_sentiment_engine('ensemble', weights={'logreg': 0.5, 'svm': 0.5})

    >>> # Fuzzy ensemble (uncertainty-aware)
    >>> engine = get_sentiment_engine(
    ...     'fuzzy_ensemble',
    ...     base_models=['logreg', 'svm'],
    ...     mf_type='gaussian',
    ...     defuzz_method='centroid',
    ... )

    """
    engine_type = engine_type.lower().strip()

    # Check standard engines
    if engine_type in _ENGINE_REGISTRY:
        return _ENGINE_REGISTRY[engine_type](**kwargs)

    # Unknown engine type
    available = list_available_engines()
    raise ValueError(
        f"Invalid engine type: '{engine_type}'. "
        f"Available engines: {available}"
    )


def get_base_engine(engine_type: str = "logreg", **kwargs) -> Any:
    """
    Create a base sentiment engine (excludes ensemble methods).

    This function is used internally by ensemble methods to avoid
    circular dependencies and to ensure only base-level models
    are used as components.

    Parameters
    ----------
    engine_type : str, optional
        Type of base engine to create. Options:
        - 'tfidf': TF-IDF + Naive Bayes
        - 'logreg': TF-IDF + Logistic Regression (default)
        - 'svm': TF-IDF + Linear SVM
    **kwargs
        Additional arguments passed to the engine constructor.

    Returns
    -------
    BaseSentimentEngine
        Initialized base sentiment engine.

    Raises
    ------
    ValueError
        If engine_type is not a valid base engine.

    Notes
    -----
    Ensemble methods ('ensemble', 'meta_learner') are not available
    through this function to prevent nested ensembles.
    """
    engine_type = engine_type.lower().strip()

    # Check standard base engines
    if engine_type in _BASE_ENGINE_REGISTRY:
        return _BASE_ENGINE_REGISTRY[engine_type](**kwargs)

    # Reject ensemble types
    if engine_type in ("ensemble", "ci_ensemble", "meta_learner", "stacking"):
        raise ValueError(
            f"'{engine_type}' is not a base engine. "
            "Base engines are: tfidf, logreg, svm"
        )

    # Unknown engine type
    available = list(_BASE_ENGINE_REGISTRY.keys())
    raise ValueError(
        f"Invalid base engine type: '{engine_type}'. "
        f"Available base engines: {available}"
    )
