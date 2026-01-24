"""
Logistic Regression Sentiment Engine.

This module implements sentiment analysis using TF-IDF vectorization
with Logistic Regression classification.

Mathematical Foundation
-----------------------
Logistic Regression models the probability of class c as:

    P(y=c|x) = exp(w_c^T x + b_c) / sum_k exp(w_k^T x + b_k)

For multi-class classification, this is known as softmax regression.

The model is trained by minimizing the cross-entropy loss:

    L = -sum_i sum_c y_ic * log(P(y=c|x_i))

With optional L2 regularization (ridge penalty):

    L_reg = L + lambda * sum_c ||w_c||^2

This results in well-calibrated probability estimates, making it
suitable for ensemble methods and confidence-based filtering.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path
from src.sentiment.base import SentimentResult, normalize_label, BaseSentimentEngine
from src.sentiment.engines.artifact_utils import format_model_load_error


class LogRegSentimentEngine(BaseSentimentEngine):
    """
    Sentiment analysis using TF-IDF and Logistic Regression.

    This engine provides well-calibrated probability estimates, making
    it particularly useful for ensemble methods and applications
    requiring confidence scores.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the trained model file.
        Default: './models/logreg/model.sav'
    vectorizer_path : str or Path, optional
        Path to the fitted TF-IDF vectorizer.
        Default: './models/logreg/tfidfVectorizer.pickle'

    Attributes
    ----------
    model : LogisticRegression
        Trained Logistic Regression classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.

    Examples
    --------
    >>> engine = LogRegSentimentEngine()
    >>> result = engine.analyze("This is a great video!")
    >>> print(f"{result.label}: {result.score:.2f}")
    Positive: 0.89

    Notes
    -----
    Advantages over Naive Bayes (TF-IDF engine):
    - Better calibrated probabilities
    - Can learn feature interactions (via polynomial features)
    - More robust to feature redundancy

    Evaluation Results (10-fold CV on YouTube comments):
    - Accuracy: 74.27%
    - F1-Macro: 74.34%
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "./models/logreg/model.sav",
        vectorizer_path: Union[str, Path] = "./models/logreg/tfidfVectorizer.pickle",
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
                f"Train models using: python scripts/train/train_classical.py --model logreg"
            )
        except Exception as exc:
            raise RuntimeError(
                format_model_load_error("logreg", model_path, vectorizer_path, exc)
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
                "LogReg model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note}"
            )

    def _predict_probs(self, vector) -> Optional[List[Dict[str, float]]]:
        """Get probability distributions from model predictions."""
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
            model="logreg",
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
                    model="logreg",
                    raw={"prediction": str(pred)},
                )
            )

        return results
