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

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.runtime_artifacts import load_runtime_artifact_json
from src.preprocessing import ClassicalPreprocessConfig
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
        weights_optimization: Optional[str] = None,
        preprocess: bool = False,
        preprocess_config: Optional[ClassicalPreprocessConfig] = None,
        calibrate: bool = True,
        allow_degraded: bool = False,
    ):
        self._weights_optimization = weights_optimization  # "pso" | "nsga2" | None
        if base_models is None:
            base_models = ["logreg", "svm", "tfidf"]

        self.preprocess = bool(preprocess)
        self.preprocess_config = preprocess_config

        self.requested_models = base_models
        self.engines = {}
        self.model_errors = {}

        # Import factory function to get base engines
        from src.sentiment.factory import get_base_engine

        classical_models = {"tfidf", "logreg", "svm"}

        for model in base_models:
            try:
                engine_kwargs = {}
                if model in classical_models and self.preprocess:
                    engine_kwargs["preprocess"] = True
                    if self.preprocess_config is not None:
                        engine_kwargs["preprocess_config"] = self.preprocess_config
                if model in classical_models:
                    # Ensemble weights (PSO/NSGA-II) are fitted against raw,
                    # uncalibrated base-model probabilities (see
                    # research/ci/multi_objective_ensemble.py /
                    # research/analysis/pso_convergence_analysis.py). Applying a
                    # per-base-model temperature here would feed the ensemble a
                    # distribution it was never optimized on, silently shifting
                    # the served blend away from the fitted weights. Calibration
                    # is applied once, at the ensemble output, below.
                    engine_kwargs["calibrate"] = False
                self.engines[model] = get_base_engine(model, **engine_kwargs)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        if not self.engines:
            raise RuntimeError(
                "No ensemble base models could be initialized. "
                f"Errors: {self.model_errors}"
            )

        if self.model_errors and not allow_degraded:
            # A partially-initialized ensemble silently renormalizes weights over
            # the surviving models — e.g. a corrupted svm/model.sav quietly turns
            # a 3-model PSO-optimized ensemble into a differently-weighted 2-model
            # one with no caller-visible signal. Fail loudly by default; callers
            # that explicitly want best-effort degraded behavior can opt in.
            raise RuntimeError(
                "Ensemble base models failed to initialize: "
                f"{self.model_errors}. Requested: {self.requested_models}. "
                "Pass allow_degraded=True to proceed with only the surviving "
                "models (weights will be renormalized over them)."
            )

        self.base_models = list(self.engines.keys())
        self.weights, self.weights_source = self._normalize_weights(weights)
        self.calibration_enabled = bool(calibrate)
        # Temperature is fitted per served ensemble *variant*
        # (results/temperature_scaling.json rows "ensemble_pso" /
        # "ensemble_nsga2" — see research/ci/temperature_scaling.py). A PSO-fitted
        # T applied to the NSGA-II blend (or vice versa) would rescale
        # probabilities from a distribution it was never fit on, silently
        # mis-calibrating them while still reporting calibration_applied=True.
        # Request-supplied or default weights have no matching fitted artifact,
        # so they are served uncalibrated (T=1.0).
        if self.calibration_enabled and self.weights_source in ("pso", "nsga2"):
            self.temperature, self.calibration_applied = self._load_temperature(
                f"ensemble_{self.weights_source}"
            )
        else:
            self.temperature, self.calibration_applied = 1.0, False

    def _load_temperature(self, model_name: str):
        """Load fitted temperature from research results; return (T, applied)."""
        try:
            data = load_runtime_artifact_json("temperature_scaling") or {}
            for entry in data.get("models", []):
                if entry.get("model") == model_name:
                    return float(entry["temperature"]), True
        except Exception:
            pass
        return 1.0, False

    def _apply_temperature(self, probs):
        """Apply temperature T via p_new[c] = p[c]^(1/T) / sum(...)."""
        if self.temperature == 1.0:
            return probs
        scaled = {k: max(v, 1e-10) ** (1.0 / self.temperature) for k, v in probs.items()}
        total = sum(scaled.values())
        return {k: v / total for k, v in scaled.items()}

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
        # Default weights: try optimized weights (pso/nsga2), fall back to empirical
        source = "default"
        # A caller-supplied dict (e.g. inline JSON `ensemble_weights` from the
        # API) must also be attributed to "request" — previously only the
        # list/tuple form below set this, so dict-supplied weights were
        # silently reported as weights_source="default" even though they came
        # from the request.
        if isinstance(weights, dict):
            source = "request"
        if weights is None:
            opt = getattr(self, "_weights_optimization", None) or "pso"

            loaded_w = None
            if opt == "nsga2":
                try:
                    _data = load_runtime_artifact_json("multi_objective_ensemble") or {}
                    loaded_w = _data.get("knee_point", {}).get("weights", {})
                    if loaded_w and set(self.base_models).issubset(loaded_w):
                        source = "nsga2"
                    else:
                        loaded_w = None
                except Exception:
                    pass

            if loaded_w is None:  # try PSO (default)
                try:
                    loaded_w = (
                        load_runtime_artifact_json("pso_ensemble_weights") or {}
                    ).get("weights", {})
                    if loaded_w and set(self.base_models).issubset(loaded_w):
                        source = "pso"
                    else:
                        loaded_w = None
                except Exception:
                    pass

            if loaded_w:
                weights = {model: float(loaded_w.get(model, 0.0)) for model in self.base_models}
            else:
                default_weights = {"logreg": 0.4, "svm": 0.4, "tfidf": 0.2}
                weights = {model: default_weights.get(model, 1.0) for model in self.base_models}

        # Convert list to dict
        if isinstance(weights, (list, tuple)):
            weights = {
                model: float(weights[idx])
                for idx, model in enumerate(self.base_models)
                if idx < len(weights)
            }
            source = "request"

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
            return {model: 1.0 / len(self.base_models) for model in self.base_models}, source

        return {
            model: max(value, 0.0) / total for model, value in normalized.items()
        }, source

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

        combined = self._apply_temperature(normalize_probs(combined))
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

            combined = self._apply_temperature(normalize_probs(combined))
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
