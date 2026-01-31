"""
Fuzzy Ensemble Sentiment Engine (Uncertainty-Aware Inference).

This engine integrates the project's fuzzy inference system into the
standard `src.sentiment` engine interface, making it usable anywhere
`get_sentiment_engine()` is used (API, scripts, etc.).

It combines probabilities from multiple base sentiment engines using a
fuzzy inference system (FIS) to produce a final probability distribution
and an uncertainty-aware sentiment label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.sentiment.base import BaseSentimentEngine, SentimentResult, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs


class FuzzyEnsembleSentimentEngine(BaseSentimentEngine):
    """
    Fuzzy ensemble sentiment analysis using fuzzy inference over base model probabilities.

    Parameters
    ----------
    base_models : List[str], optional
        List of base model types to include in the fuzzy ensemble.
        Default: ['logreg', 'svm', 'tfidf'].
    mf_type : str, optional
        Membership function type: 'triangular', 'trapezoidal', 'gaussian'.
        Default: 'gaussian'.
    defuzz_method : str, optional
        Defuzzification method: 'centroid', 'bisector', 'mom', 'som', 'lom',
        or 'weighted_average' (depends on implementation).
        Default: 'centroid'.
    t_norm : str, optional
        T-norm for fuzzy AND operations: e.g., 'min', 'product', 'bounded_product'.
        Default: 'min'.
    t_conorm : str, optional
        T-conorm for fuzzy OR operations: e.g., 'max', 'prob_sum', 'bounded_sum'.
        Default: 'max'.
    alpha_cut : float, optional
        Alpha-cut threshold (0.0 disables).
        Default: 0.0.
    resolution : int, optional
        Numeric resolution used in defuzzification.
        Default: 100.
    confidence_threshold : float, optional
        Threshold used by the fuzzy system to flag low-confidence cases.
        Default: 0.6.
    enable_logging : bool, optional
        Enable fuzzy engine logging.
        Default: False.
    """

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        mf_type: str = "gaussian",
        defuzz_method: str = "centroid",
        t_norm: str = "min",
        t_conorm: str = "max",
        alpha_cut: float = 0.0,
        resolution: int = 100,
        confidence_threshold: float = 0.6,
        enable_logging: bool = False,
    ):
        if base_models is None:
            base_models = ["logreg", "svm", "tfidf"]

        requested = []
        for model in base_models:
            key = str(model).strip().lower()
            if key and key not in requested:
                requested.append(key)

        self.requested_models = requested
        self.model_errors: Dict[str, str] = {}

        # Create base engines (avoid nested ensembles by using get_base_engine).
        from src.sentiment.factory import get_base_engine

        base_engines: Dict[str, Any] = {}
        for model in self.requested_models:
            try:
                base_engines[model] = get_base_engine(model)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        if not base_engines:
            raise RuntimeError(
                "No fuzzy ensemble base models could be initialized. "
                f"Errors: {self.model_errors}"
            )

        self.base_models = list(base_engines.keys())

        # Lazy import the fuzzy engine implementation (lives in research module).
        from research.computational_intelligence.fuzzy.engine_integration import (
            FuzzySentimentEngine,
        )

        self._engine = FuzzySentimentEngine(
            base_engines=base_engines,
            mf_type=mf_type,
            defuzz_method=defuzz_method,
            t_norm=t_norm,
            t_conorm=t_conorm,
            alpha_cut=float(alpha_cut or 0.0),
            resolution=int(resolution or 100),
            confidence_threshold=float(confidence_threshold or 0.0),
            enable_logging=bool(enable_logging),
        )

        self.mf_type = mf_type
        self.defuzz_method = defuzz_method
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.alpha_cut = float(alpha_cut or 0.0)
        self.resolution = int(resolution or 100)
        self.confidence_threshold = float(confidence_threshold or 0.0)

    def _to_sentiment_result(self, result: Any) -> SentimentResult:
        probs = getattr(result, "probs", None)
        if isinstance(result, dict):
            probs = result.get("probs", probs)
        probs = normalize_probs(probs or {})

        label = getattr(result, "label", None)
        if isinstance(result, dict):
            label = result.get("label", label)
        label = normalize_label(label)

        # Prefer explicit score; otherwise derive from probs.
        score_value = getattr(result, "score", None)
        if isinstance(result, dict):
            score_value = result.get("score", score_value)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = float(probs.get(label, 0.0))

        # Ensure probs include all labels
        probs = {
            sentiment: float(probs.get(sentiment, 0.0)) for sentiment in SENTIMENT_LABELS
        }
        probs = normalize_probs(probs)

        model_name = getattr(result, "model", None) or "fuzzy_ensemble"
        raw = None
        if hasattr(result, "to_dict"):
            try:
                raw = result.to_dict()
            except Exception:
                raw = None
        if raw is None and isinstance(result, dict):
            raw = result

        # Add base model initialization info for transparency.
        if raw is None:
            raw = {}
        if isinstance(raw, dict):
            raw.setdefault("base_models", self.base_models)
            raw.setdefault("requested_models", self.requested_models)
            if self.model_errors:
                raw.setdefault("model_errors", self.model_errors)

        return SentimentResult(
            label=label,
            score=score,
            probs=probs,
            model=str(model_name),
            raw=raw,
        )

    def analyze(self, text: str) -> SentimentResult:
        result = self._engine.analyze(text)
        return self._to_sentiment_result(result)

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        if not texts:
            return []
        results = self._engine.analyze_batch(texts)
        return [self._to_sentiment_result(result) for result in results]

    def get_model_info(self) -> Dict[str, Any]:
        try:
            return self._engine.get_model_info()
        except Exception:
            return {
                "base_models": self.base_models,
                "requested_models": self.requested_models,
                "mf_type": self.mf_type,
                "defuzz_method": self.defuzz_method,
                "t_norm": self.t_norm,
                "t_conorm": self.t_conorm,
                "alpha_cut": self.alpha_cut,
                "resolution": self.resolution,
                "confidence_threshold": self.confidence_threshold,
            }

