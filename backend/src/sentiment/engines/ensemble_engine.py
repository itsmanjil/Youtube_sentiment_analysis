"""
Ensemble Sentiment Engine (Weighted Soft Voting).

This module implements an ensemble approach to sentiment analysis
using weighted soft voting across multiple base models.

Mathematical Foundation
-----------------------
Given K base models with weights w_k (where sum(w_k) = 1), the
ensemble prediction is computed as:

    P_ensemble(c|x) = sum_k w_k * P_k(c|x)

Where P_k(c|x) is the probability of class c from model k.

The final prediction is:

    y_hat = argmax_c P_ensemble(c|x)

This approach, known as weighted soft voting, often outperforms
individual models by combining their complementary strengths.

Weight Optimization
-------------------
Weights can be optimized using various methods:
- Manual tuning based on validation performance
- Grid search over weight combinations
- Particle Swarm Optimization (PSO) - see research/optimize_ensemble.py
- Bayesian optimization
"""

from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.sentiment.base import SentimentResult, coerce_sentiment_result, BaseSentimentEngine


class EnsembleSentimentEngine(BaseSentimentEngine):
    """
    Ensemble sentiment analysis using weighted soft voting.

    This engine combines predictions from multiple base models using
    weighted averaging of probability distributions.

    Parameters
    ----------
    base_models : List[str], optional
        List of base model types to include in the ensemble.
        Default: ['logreg', 'svm', 'tfidf']
    weights : Dict[str, float] or List[float], optional
        Weights for each base model. Can be a dictionary mapping
        model names to weights, or a list in the same order as base_models.
        Default: {'logreg': 0.4, 'svm': 0.4, 'tfidf': 0.2}

    Attributes
    ----------
    engines : Dict[str, BaseSentimentEngine]
        Initialized base model engines.
    weights : Dict[str, float]
        Normalized weights for each model.
    model_errors : Dict[str, str]
        Error messages for models that failed to initialize.

    Examples
    --------
    >>> # Default ensemble with all classical models
    >>> engine = EnsembleSentimentEngine()
    >>> result = engine.analyze("This video is great!")
    >>> print(result.label, result.score)

    >>> # Custom weights optimized via PSO
    >>> engine = EnsembleSentimentEngine(
    ...     base_models=['logreg', 'svm'],
    ...     weights={'logreg': 0.45, 'svm': 0.55}
    ... )

    Notes
    -----
    The ensemble approach provides several benefits:
    - Reduced variance through averaging
    - Better handling of different types of comments
    - More robust confidence estimates

    For best results, use weights optimized on a validation set.
    See research/optimize_ensemble.py for PSO-based optimization.
    """

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        weights: Optional[Union[Dict[str, float], List[float]]] = None,
    ):
        if base_models is None:
            base_models = ["logreg", "svm", "tfidf"]

        self.requested_models = base_models
        self.engines = {}
        self.model_errors = {}

        # Import factory function to get base engines
        from src.sentiment.factory import get_base_engine

        for model in base_models:
            try:
                self.engines[model] = get_base_engine(model)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        if not self.engines:
            raise RuntimeError(
                "No ensemble base models could be initialized. "
                f"Errors: {self.model_errors}"
            )

        self.base_models = list(self.engines.keys())
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(
        self, weights: Optional[Union[Dict[str, float], List[float]]]
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Parameters
        ----------
        weights : Optional[Union[Dict[str, float], List[float]]]
            Raw weights from initialization.

        Returns
        -------
        Dict[str, float]
            Normalized weights that sum to 1.0.
        """
        # Default weights based on empirical performance
        if weights is None:
            default_weights = {
                "logreg": 0.4,
                "svm": 0.4,
                "tfidf": 0.2,
            }
            weights = {
                model: default_weights.get(model, 1.0) for model in self.base_models
            }

        # Convert list to dict
        if isinstance(weights, (list, tuple)):
            weights = {
                model: float(weights[idx])
                for idx, model in enumerate(self.base_models)
                if idx < len(weights)
            }

        # Normalize dict weights
        if isinstance(weights, dict):
            normalized = {
                model: float(weights.get(model, 0.0)) for model in self.base_models
            }
        else:
            normalized = {model: 1.0 for model in self.base_models}

        # Ensure positive and normalized
        total = sum(max(value, 0.0) for value in normalized.values())
        if total <= 0:
            return {model: 1.0 / len(self.base_models) for model in self.base_models}

        return {
            model: max(value, 0.0) / total for model, value in normalized.items()
        }

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a single text using ensemble voting.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        SentimentResult
            Ensemble sentiment prediction.
        """
        model_results = {}
        for model_name, engine in self.engines.items():
            result = coerce_sentiment_result(engine.analyze(text), model_name)
            model_results[model_name] = result

        # Weighted combination of probabilities
        combined = {label: 0.0 for label in SENTIMENT_LABELS}
        for model_name, result in model_results.items():
            weight = self.weights.get(model_name, 0.0)
            for label in SENTIMENT_LABELS:
                combined[label] += weight * result.probs.get(label, 0.0)

        combined = normalize_probs(combined)
        sentiment = max(combined, key=combined.get)

        return SentimentResult(
            label=sentiment,
            score=float(combined.get(sentiment, 0.0)),
            probs=combined,
            model="ensemble",
            raw={
                "weights": self.weights,
                "models": {name: result.to_dict() for name, result in model_results.items()},
                "model_errors": self.model_errors,
            },
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts using ensemble voting.

        Parameters
        ----------
        texts : List[str]
            List of texts to analyze.

        Returns
        -------
        List[SentimentResult]
            List of ensemble predictions.
        """
        # Get predictions from all base models
        model_outputs = {}
        for model_name, engine in self.engines.items():
            if hasattr(engine, "batch_analyze"):
                results = engine.batch_analyze(texts)
            else:
                results = [engine.analyze(text) for text in texts]
            model_outputs[model_name] = [
                coerce_sentiment_result(result, model_name) for result in results
            ]

        # Combine predictions for each text
        combined_results = []
        for idx in range(len(texts)):
            combined = {label: 0.0 for label in SENTIMENT_LABELS}
            for model_name, results in model_outputs.items():
                weight = self.weights.get(model_name, 0.0)
                result = results[idx]
                for label in SENTIMENT_LABELS:
                    combined[label] += weight * result.probs.get(label, 0.0)

            combined = normalize_probs(combined)
            sentiment = max(combined, key=combined.get)

            combined_results.append(
                SentimentResult(
                    label=sentiment,
                    score=float(combined.get(sentiment, 0.0)),
                    probs=combined,
                    model="ensemble",
                    raw={
                        "weights": self.weights,
                        "models": {
                            name: model_outputs[name][idx].to_dict()
                            for name in model_outputs
                        },
                        "model_errors": self.model_errors,
                    },
                )
            )

        return combined_results
