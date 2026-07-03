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

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.sentiment.base import BaseSentimentEngine, SentimentResult, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.runtime_artifacts import load_runtime_artifact_json
from src.preprocessing import ClassicalPreprocessConfig


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
        preprocess: bool = False,
        preprocess_config: Optional[ClassicalPreprocessConfig] = None,
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
        self.preprocess = bool(preprocess)
        self.preprocess_config = preprocess_config

        # Create base engines (avoid nested ensembles by using get_base_engine).
        from src.sentiment.factory import get_base_engine

        base_engines: Dict[str, Any] = {}
        classical_models = {"tfidf", "logreg", "svm"}
        for model in self.requested_models:
            try:
                engine_kwargs = {}
                if model in classical_models and self.preprocess:
                    engine_kwargs["preprocess"] = True
                    if self.preprocess_config is not None:
                        engine_kwargs["preprocess_config"] = self.preprocess_config
                base_engines[model] = get_base_engine(model, **engine_kwargs)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        if not base_engines:
            raise RuntimeError(
                "No fuzzy ensemble base models could be initialized. "
                f"Errors: {self.model_errors}"
            )

        self.base_models = list(base_engines.keys())
        self._base_engines = base_engines  # kept for NF gate access

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

        # Load neuro-fuzzy gate params (ANFIS learned MFs from research).
        # Only activates when base models match the trained gate exactly.
        self._nf_mfs: Dict[str, List[Dict]] = {}
        try:
            _nf_data = load_runtime_artifact_json("neuro_fuzzy_gate") or {}
            _nf_models = _nf_data.get("architecture", {}).get("model_names", [])
            if set(_nf_models) == set(self.base_models):
                for _entry in _nf_data.get("learned_mfs", []):
                    _m = _entry["model"]
                    self._nf_mfs.setdefault(_m, []).append({
                        "center": float(_entry["center"]),
                        "width": float(_entry["width"]),
                        "alpha": float(_entry["alpha"]),
                    })
        except Exception:
            self._nf_mfs = {}

    def _nf_gate_blend(self, probs_by_model: Dict[str, Dict[str, float]]):
        """
        Neuro-fuzzy adaptive gating (ANFIS-lite).
        Blends base model probs using Gaussian MF activations → softmax gate weights.
        Returns (blended_probs, gate_weights).
        """
        # Confidence = 1 - normalised Shannon entropy per model
        model_conf = {}
        for m, probs in probs_by_model.items():
            p_list = [max(v, 1e-10) for v in probs.values()]
            H = -sum(p * math.log(p) for p in p_list)
            H_norm = H / math.log(max(len(p_list), 2))
            model_conf[m] = 1.0 - H_norm

        # Gaussian MF activation sum per model.
        # Must match research/ci/neuro_fuzzy_gate.py::_fuzzy_activations exactly:
        #   gate_m = sum_k alpha_k * exp(-0.5 * ((confidence_m - center_k) / width_k) ** 2)
        # (alpha is a linear consequent weight applied *outside* the exponent, not
        # inside it — the two forms coincide only when alpha == 1.)
        model_act = {}
        for m in probs_by_model:
            act = 0.0
            for mf in self._nf_mfs.get(m, []):
                c, w, a = mf["center"], mf["width"], mf["alpha"]
                width = max(w, 1e-3)
                mu = math.exp(-0.5 * ((model_conf[m] - c) / width) ** 2)
                act += a * mu
            model_act[m] = act

        # Softmax → gate weights
        max_act = max(model_act.values()) if model_act else 0.0
        exp_act = {m: math.exp(v - max_act) for m, v in model_act.items()}
        total = sum(exp_act.values()) or 1.0
        gate_w = {m: v / total for m, v in exp_act.items()}

        # Weighted blend of probabilities
        combined = {lbl: 0.0 for lbl in SENTIMENT_LABELS}
        for m, w in gate_w.items():
            for lbl in SENTIMENT_LABELS:
                combined[lbl] += w * probs_by_model[m].get(lbl, 0.0)
        return normalize_probs(combined), gate_w

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
        if self._nf_mfs:
            probs_by_model = {}
            for m, eng in self._base_engines.items():
                r = eng.analyze(text)
                probs_by_model[m] = normalize_probs(getattr(r, "probs", {}) or {})
            combined, gate_w = self._nf_gate_blend(probs_by_model)
            sentiment = max(combined, key=combined.get)
            return SentimentResult(
                label=sentiment,
                score=float(combined.get(sentiment, 0.0)),
                probs=combined,
                model="fuzzy_ensemble",
                raw={"gate_weights": gate_w, "routing": "neuro_fuzzy"},
            )
        result = self._engine.analyze(text)
        return self._to_sentiment_result(result)

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        if not texts:
            return []
        if self._nf_mfs:
            # Collect batch probs from each base engine
            model_batch: Dict[str, List[Dict[str, float]]] = {}
            for m, eng in self._base_engines.items():
                if hasattr(eng, "batch_analyze"):
                    batch_results = eng.batch_analyze(texts)
                else:
                    batch_results = [eng.analyze(t) for t in texts]
                model_batch[m] = [
                    normalize_probs(getattr(r, "probs", {}) or {}) for r in batch_results
                ]
            out = []
            for i in range(len(texts)):
                probs_by_model = {m: model_batch[m][i] for m in model_batch}
                combined, gate_w = self._nf_gate_blend(probs_by_model)
                sentiment = max(combined, key=combined.get)
                out.append(SentimentResult(
                    label=sentiment,
                    score=float(combined.get(sentiment, 0.0)),
                    probs=combined,
                    model="fuzzy_ensemble",
                    raw={"gate_weights": gate_w, "routing": "neuro_fuzzy"},
                ))
            return out
        results = self._engine.analyze_batch(texts)
        return [self._to_sentiment_result(result) for result in results]

    def get_model_info(self) -> Dict[str, Any]:
        try:
            info = self._engine.get_model_info()
        except Exception:
            info = {
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
        info["nf_gate_active"] = bool(self._nf_mfs)
        return info
