"""
TF-IDF + Naive Bayes Sentiment Engine.

This module implements a classical machine learning approach to sentiment
analysis using TF-IDF vectorization with Multinomial Naive Bayes classification.

Mathematical Foundation
-----------------------
TF-IDF (Term Frequency-Inverse Document Frequency) is computed as:

    TF-IDF(t, d) = TF(t, d) * IDF(t)

Where:
    TF(t, d) = count of term t in document d
    IDF(t) = log(N / df(t))
    N = total number of documents
    df(t) = number of documents containing term t

The Multinomial Naive Bayes classifier computes:

    P(c|d) proportional to P(c) * product(P(t_i|c)^count(t_i,d))

Where:
    P(c) = prior probability of class c
    P(t_i|c) = probability of term t_i given class c
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path, Config
from src.sentiment.base import SentimentResult, normalize_label, BaseSentimentEngine


class TFIDFSentimentEngine(BaseSentimentEngine):
    """
    Sentiment analysis using TF-IDF vectorization and Naive Bayes.

    This engine uses a pre-trained TF-IDF vectorizer and Multinomial
    Naive Bayes classifier for sentiment classification.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the trained model file.
        Default: './models/tfidf/model.sav'
    vectorizer_path : str or Path, optional
        Path to the fitted TF-IDF vectorizer.
        Default: './models/tfidf/tfidfVectorizer.pickle'

    Attributes
    ----------
    model : sklearn classifier
        Trained Multinomial Naive Bayes classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.

    Examples
    --------
    >>> engine = TFIDFSentimentEngine()
    >>> result = engine.analyze("This video is amazing!")
    >>> print(result.label, result.score)
    Positive 0.85

    >>> results = engine.batch_analyze(["Great!", "Terrible!", "Okay"])
    >>> [r.label for r in results]
    ['Positive', 'Negative', 'Neutral']

    Notes
    -----
    This is the simplest baseline model. While fast, it:
    - Ignores word order (bag-of-words assumption)
    - Cannot capture semantic relationships
    - May struggle with sarcasm and context

    For better accuracy, consider LogRegSentimentEngine or transformer-based models.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "./models/tfidf/model.sav",
        vectorizer_path: Union[str, Path] = "./models/tfidf/tfidfVectorizer.pickle",
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
                f"Train models using: python scripts/train/train_classical.py --model tfidf"
            )

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
                "TF-IDF model/vectorizer is not fitted or is incompatible with "
                f"the current scikit-learn version.{version_note} "
                "Reinstall scikit-learn==0.24.2 or retrain the model."
            )

    def _predict_probs(self, vector) -> Optional[List[Dict[str, float]]]:
        """
        Get probability distributions from model predictions.

        Parameters
        ----------
        vector : sparse matrix
            TF-IDF transformed input.

        Returns
        -------
        Optional[List[Dict[str, float]]]
            List of probability dictionaries, or None if model doesn't support
            probability prediction.
        """
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

        df = pd.DataFrame([{"tweet": text}])
        tweet_vec = self.vectorizer.transform(df["tweet"])
        prediction = self.model.predict(tweet_vec)[0]

        sentiment = normalize_label(prediction)
        probs = self._predict_probs(tweet_vec)
        if probs is None:
            probs = normalize_probs({sentiment: 1.0})
        elif isinstance(probs, list):
            probs = probs[0]

        return SentimentResult(
            label=sentiment,
            score=float(probs.get(sentiment, 0.0)),
            probs=probs,
            model="tfidf",
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

        df = pd.DataFrame([{"tweet": text} for text in texts])
        tweet_vec = self.vectorizer.transform(df["tweet"])
        predictions = self.model.predict(tweet_vec)

        probs = self._predict_probs(tweet_vec)
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
                    model="tfidf",
                    raw={"prediction": str(pred)},
                )
            )

        return results
