"""
Meta-Learner Sentiment Engine (Stacked Ensemble).

This module implements a two-level stacking ensemble for sentiment analysis.
Unlike simple weighted voting, the meta-learner learns optimal combination
rules from data.

Mathematical Foundation
-----------------------
Level 0 (Base Models):
    For each base model k, compute probability distribution:
    z_k = [P_k(neg|x), P_k(neu|x), P_k(pos|x)]

Feature Matrix Construction:
    For K base models, construct feature vector:
    Z = [z_1; z_2; ...; z_K] (concatenation)

    Optionally, include logits:
    Z = [z_1, logit_1; z_2, logit_2; ...; z_K, logit_K]

Level 1 (Meta-Learner):
    Train a classifier on Z to predict final sentiment:
    y_hat = meta_learner(Z)

Training Protocol
-----------------
To prevent overfitting, out-of-fold predictions are used:
1. Split training data into K folds
2. For each fold, train base models on K-1 folds
3. Generate predictions on held-out fold
4. Collect all out-of-fold predictions as Z
5. Train meta-learner on (Z, y)

This ensures meta-learner sees predictions that base models
have never trained on, preventing label leakage.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path
from src.sentiment.base import SentimentResult, normalize_label, coerce_sentiment_result, BaseSentimentEngine


class MetaLearnerSentimentEngine(BaseSentimentEngine):
    """
    Stacked ensemble using a meta-learner for final predictions.

    This engine uses base model predictions as features for a
    second-level classifier (meta-learner), learning optimal
    combination rules from training data.

    Parameters
    ----------
    meta_model_path : str or Path, optional
        Path to the trained meta-learner model.
        Default: './models/meta_learner.pkl'
    base_models : List[str], optional
        List of base model types. If None, uses the models stored
        in the meta-learner file.

    Attributes
    ----------
    meta_learner : sklearn classifier
        Trained meta-learner (typically LogisticRegression or XGBoost).
    engines : Dict[str, BaseSentimentEngine]
        Initialized base model engines.
    base_models : List[str]
        Ordered list of base model names (order must match training).
    feature_type : str
        Type of features used ('probs' or 'probs+logits').

    Examples
    --------
    >>> engine = MetaLearnerSentimentEngine()
    >>> result = engine.analyze("This is a fantastic video!")
    >>> print(result.label, result.score)
    Positive 0.94

    Notes
    -----
    The meta-learner typically outperforms simple weighted voting
    because it can learn:
    - Model-specific reliability for different comment types
    - Non-linear combination rules
    - Correction for systematic model biases

    Training: Use research/meta_learner.py to train the meta-learner
    with proper out-of-fold cross-validation.
    """

    def __init__(
        self,
        meta_model_path: Union[str, Path] = "./models/meta_learner.pkl",
        base_models: Optional[List[str]] = None,
    ):
        self.meta_model_path = get_model_path(meta_model_path)

        if not self.meta_model_path.exists():
            raise FileNotFoundError(
                f"Meta-learner model not found: {self.meta_model_path}. "
                "Train and save using research/meta_learner.py"
            )

        with open(self.meta_model_path, "rb") as f:
            saved = pickle.load(f)

        self.meta_learner = saved.get("meta_learner")
        if self.meta_learner is None:
            raise RuntimeError(
                "Meta-learner file is missing the trained model. "
                "Re-train and save the meta-learner."
            )

        # Load base model configuration
        saved_base_models = saved.get("base_models") or []
        saved_base_models = [
            str(model).strip().lower()
            for model in saved_base_models
            if str(model).strip()
        ]

        if base_models is None:
            base_models = saved_base_models
            self.base_models_source = "model"
        else:
            self.base_models_source = "request"

        self.base_models = [
            str(model).strip().lower()
            for model in base_models
            if str(model).strip()
        ]

        if not self.base_models:
            raise RuntimeError(
                "Meta-learner base models are not specified. "
                "Provide base_models or retrain the meta-learner."
            )

        if saved_base_models and self.base_models != saved_base_models:
            raise RuntimeError(
                "Meta-learner base models do not match the trained configuration. "
                f"Saved order: {saved_base_models}. Requested: {self.base_models}."
            )

        # Load metadata
        self.feature_type = saved.get("feature_type", "probs")
        self.meta_learner_type = saved.get("meta_learner_type")
        self.label2idx = saved.get(
            "label2idx",
            {"Negative": 0, "Neutral": 1, "Positive": 2},
        )
        self.idx2label = saved.get(
            "idx2label",
            {value: key for key, value in self.label2idx.items()},
        )

        # Initialize base engines
        from src.sentiment.factory import get_base_engine

        self.engines = {}
        self.model_errors = {}

        for model in self.base_models:
            if model in ("ensemble", "meta_learner", "stacking"):
                self.model_errors[model] = (
                    "Meta-learner base models cannot include ensemble/stacking."
                )
                continue
            try:
                self.engines[model] = get_base_engine(model)
            except Exception as exc:
                self.model_errors[model] = str(exc)

        missing = [model for model in self.base_models if model not in self.engines]
        if missing:
            raise RuntimeError(
                "Meta-learner base models could not be initialized: "
                f"{missing}. Errors: {self.model_errors}"
            )

    def _label_for_class(self, class_label: Any) -> str:
        """Convert class label from meta-learner to sentiment label."""
        try:
            class_idx = int(class_label)
        except (TypeError, ValueError):
            return normalize_label(class_label)
        return self.idx2label.get(class_idx, normalize_label(class_label))

    def _feature_vector(self, result: SentimentResult) -> List[float]:
        """
        Convert a SentimentResult to feature vector for meta-learner.

        Parameters
        ----------
        result : SentimentResult
            Base model prediction.

        Returns
        -------
        List[float]
            Feature vector [P(neg), P(neu), P(pos)] or with logits.
        """
        probs = normalize_probs(result.probs)
        vector = [
            probs.get("Negative", 0.0),
            probs.get("Neutral", 0.0),
            probs.get("Positive", 0.0),
        ]
        if self.feature_type == "probs+logits":
            vector.append(float(result.score))
        return vector

    def _get_base_predictions(self, texts: List[str]) -> Dict[str, List[SentimentResult]]:
        """Get predictions from all base models."""
        model_outputs = {}
        for model_name, engine in self.engines.items():
            if hasattr(engine, "batch_analyze"):
                results = engine.batch_analyze(texts)
            else:
                results = [engine.analyze(text) for text in texts]
            model_outputs[model_name] = [
                coerce_sentiment_result(result, model_name) for result in results
            ]
        return model_outputs

    def _build_feature_matrix(
        self, base_predictions: Dict[str, List[SentimentResult]]
    ) -> List[List[float]]:
        """
        Build feature matrix from base model predictions.

        Parameters
        ----------
        base_predictions : Dict[str, List[SentimentResult]]
            Predictions from each base model.

        Returns
        -------
        List[List[float]]
            Feature matrix where each row is features for one sample.
        """
        if not base_predictions:
            return []

        n_samples = len(next(iter(base_predictions.values())))
        feature_matrix = []

        for idx in range(n_samples):
            row = []
            for model_name in self.base_models:
                row.extend(self._feature_vector(base_predictions[model_name][idx]))
            feature_matrix.append(row)

        return feature_matrix

    def _predict_proba(self, features: List[List[float]]) -> List[Dict[str, float]]:
        """Get probability distributions from meta-learner."""
        if hasattr(self.meta_learner, "predict_proba"):
            raw_probs = self.meta_learner.predict_proba(features)
            classes = getattr(
                self.meta_learner,
                "classes_",
                list(range(len(raw_probs[0]))),
            )
            probs_list = []
            for row in raw_probs:
                mapped = {}
                for idx, class_label in enumerate(classes):
                    mapped[self._label_for_class(class_label)] = float(row[idx])
                probs_list.append(normalize_probs(mapped))
            return probs_list

        # Fallback for models without predict_proba
        predictions = self.meta_learner.predict(features)
        return [
            normalize_probs({self._label_for_class(pred): 1.0})
            for pred in predictions
        ]

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a single text using meta-learning.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        SentimentResult
            Meta-learner sentiment prediction.
        """
        results = self.batch_analyze([text])
        return results[0] if results else SentimentResult(
            label="Neutral",
            score=0.0,
            probs=normalize_probs({}),
            model="meta_learner",
            raw={"empty_input": True},
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts using meta-learning.

        Parameters
        ----------
        texts : List[str]
            List of texts to analyze.

        Returns
        -------
        List[SentimentResult]
            List of meta-learner predictions.
        """
        if not texts:
            return []

        base_predictions = self._get_base_predictions(texts)
        features = self._build_feature_matrix(base_predictions)
        probs_list = self._predict_proba(features)

        results = []
        for probs in probs_list:
            label = max(probs, key=probs.get)
            results.append(
                SentimentResult(
                    label=label,
                    score=float(probs.get(label, 0.0)),
                    probs=probs,
                    model="meta_learner",
                    raw={
                        "base_models": self.base_models,
                        "model_errors": self.model_errors,
                    },
                )
            )

        return results
