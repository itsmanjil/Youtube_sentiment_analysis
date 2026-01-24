"""
Support Vector Machine Sentiment Engine.

This module implements sentiment analysis using TF-IDF vectorization
with Linear SVM classification.

Mathematical Foundation
-----------------------
Linear SVM finds the optimal hyperplane w^T x + b = 0 that maximizes
the margin between classes. For multi-class problems, it uses
one-vs-rest (OVR) strategy:

    f_c(x) = w_c^T x + b_c

The predicted class is:

    y_hat = argmax_c f_c(x)

Probability Estimation
----------------------
Since SVM outputs are decision function values (not probabilities),
we convert them to probabilities using the softmax function:

    P(c|x) = exp(f_c(x)) / sum_k exp(f_k(x))

For numerical stability, we use:

    P(c|x) = exp(f_c(x) - max_k f_k(x)) / sum_k exp(f_k(x) - max_k f_k(x))

Note: These are not calibrated probabilities. For better calibration,
use Platt scaling or isotonic regression during training.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path
from src.sentiment.base import SentimentResult, normalize_label, BaseSentimentEngine
from src.sentiment.engines.artifact_utils import format_model_load_error


class SVMSentimentEngine(BaseSentimentEngine):
    """
    Sentiment analysis using TF-IDF and Linear SVM.

    This engine typically achieves the best accuracy among classical ML
    approaches for sentiment analysis tasks.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the trained model file.
        Default: './models/svm/model.sav'
    vectorizer_path : str or Path, optional
        Path to the fitted TF-IDF vectorizer.
        Default: './models/svm/tfidfVectorizer.pickle'

    Attributes
    ----------
    model : LinearSVC
        Trained Linear SVM classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.

    Examples
    --------
    >>> engine = SVMSentimentEngine()
    >>> result = engine.analyze("Absolutely terrible video, waste of time")
    >>> print(f"{result.label}: {result.score:.2f}")
    Negative: 0.91

    Notes
    -----
    Evaluation Results (10-fold CV on YouTube comments):
    - Accuracy: 75.08% (BEST among classical models)
    - F1-Macro: 75.14%

    The SVM achieves the highest accuracy but its probability estimates
    are not as well-calibrated as Logistic Regression. Consider using
    CalibratedClassifierCV during training for better probability estimates.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "./models/svm/model.sav",
        vectorizer_path: Union[str, Path] = "./models/svm/tfidfVectorizer.pickle",
    ):
        model_path = get_model_path(model_path)
        vectorizer_path = get_model_path(vectorizer_path)

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_path}\n"
                f"  - {vectorizer_path}\n"
                f"Train models using: python scripts/train/train_classical.py --model svm"
            )
        except Exception as exc:
            raise RuntimeError(
                format_model_load_error("svm", model_path, vectorizer_path, exc)
            ) from exc

        self._validate_fitted()

    def _validate_fitted(self) -> None:
        """Validate that model and vectorizer are properly fitted."""
        try:
            import sklearn
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            return

        errors = []
        try:
            check_is_fitted(self.vectorizer, "vocabulary_")
            if hasattr(self.vectorizer, "_tfidf"):
                check_is_fitted(self.vectorizer._tfidf, "idf_")
        except Exception as exc:
            errors.append(f"Vectorizer not fitted ({exc.__class__.__name__}).")

        try:
            check_is_fitted(self.model, "classes_")
        except Exception as exc:
            errors.append(f"Classifier not fitted ({exc.__class__.__name__}).")

        if errors:
            model_version = getattr(self.model, "__sklearn_version__", None)
            vectorizer_version = getattr(self.vectorizer, "__sklearn_version__", None)
            version_note = ""
            if model_version or vectorizer_version:
                version_note = (
                    f" Model version: {model_version or 'unknown'},"
                    f" vectorizer version: {vectorizer_version or 'unknown'},"
                    f" runtime version: {sklearn.__version__}."
                )
            raise RuntimeError(
                "SVM model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note}"
            )

    def _predict_probs(self, vector) -> Optional[List[Dict[str, float]]]:
        """
        Get probability distributions from model predictions.

        For SVM, we use the decision function values and apply softmax
        to convert them to pseudo-probabilities.
        """
        # Try predict_proba first (available if CalibratedClassifierCV was used)
        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(vector)
            labels = getattr(self.model, "classes_", [])
            if len(getattr(raw_probs, "shape", [])) == 1:
                raw_probs = [raw_probs]
            mapped_rows = []
            for row in raw_probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows

        # Fall back to decision_function with softmax
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(vector)

            # Handle binary classification case
            if getattr(scores, "ndim", 1) == 1:
                scores = np.vstack([-scores, scores]).T

            # Apply softmax with numerical stability
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            labels = getattr(self.model, "classes_", [])
            mapped_rows = []
            for row in probs:
                mapped = {
                    normalize_label(label): float(row[idx])
                    for idx, label in enumerate(labels)
                }
                mapped_rows.append(normalize_probs(mapped))
            return mapped_rows

        return None

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a single text.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        SentimentResult
            Sentiment prediction with label, score, and probabilities.
        """
        import pandas as pd

        df = pd.DataFrame([{"text": text}])
        text_vec = self.vectorizer.transform(df["text"])
        prediction = self.model.predict(text_vec)[0]

        sentiment = normalize_label(prediction)
        probs = self._predict_probs(text_vec)
        if probs is None:
            probs = normalize_probs({sentiment: 1.0})
        elif isinstance(probs, list):
            probs = probs[0]

        return SentimentResult(
            label=sentiment,
            score=float(probs.get(sentiment, 0.0)),
            probs=probs,
            model="svm",
            raw={"prediction": str(prediction)},
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze the sentiment of multiple texts efficiently.

        Parameters
        ----------
        texts : List[str]
            List of texts to analyze.

        Returns
        -------
        List[SentimentResult]
            List of sentiment predictions.
        """
        import pandas as pd

        df = pd.DataFrame([{"text": text} for text in texts])
        text_vec = self.vectorizer.transform(df["text"])
        predictions = self.model.predict(text_vec)

        probs = self._predict_probs(text_vec)
        results = []
        for idx, pred in enumerate(predictions):
            sentiment = normalize_label(pred)
            if probs:
                row_probs = probs[idx] if isinstance(probs, list) else probs
                sentiment_probs = {
                    label: row_probs.get(label, 0.0) for label in SENTIMENT_LABELS
                }
            else:
                sentiment_probs = normalize_probs({sentiment: 1.0})

            results.append(
                SentimentResult(
                    label=sentiment,
                    score=float(sentiment_probs.get(sentiment, 0.0)),
                    probs=sentiment_probs,
                    model="svm",
                    raw={"prediction": str(pred)},
                )
            )

        return results
