"""
Base classes and utilities for sentiment analysis engines.

This module provides the foundational components used by all sentiment
analysis engines, including the SentimentResult dataclass and label
normalization utilities.

Classes
-------
SentimentResult
    Immutable dataclass representing a sentiment analysis result.

Functions
---------
normalize_label
    Normalize sentiment label strings to standard format.
coerce_sentiment_result
    Convert various result formats to SentimentResult.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from src.utils import SENTIMENT_LABELS, normalize_probs


def normalize_label(label: Optional[Union[str, int]]) -> str:
    """
    Normalize a sentiment label to standard format.

    This function handles various input formats and maps them to the
    standard labels: 'Positive', 'Neutral', 'Negative'.

    Parameters
    ----------
    label : Optional[Union[str, int]]
        Input label in various formats:
        - Standard: 'Positive', 'Neutral', 'Negative'
        - Abbreviated: 'pos', 'neg', 'neu'
        - Numeric: 0 (Negative), 1 (Neutral), 2 (Positive)
        - None or empty: Returns 'Neutral'

    Returns
    -------
    str
        Normalized label: 'Positive', 'Neutral', or 'Negative'.

    Examples
    --------
    >>> normalize_label('pos')
    'Positive'

    >>> normalize_label('NEGATIVE')
    'Negative'

    >>> normalize_label(2)
    'Positive'

    >>> normalize_label(None)
    'Neutral'
    """
    if label is None:
        return "Neutral"

    key = str(label).strip()
    if not key:
        return "Neutral"

    normalized = key.lower()
    label_map = {
        "positive": "Positive",
        "pos": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        # Numeric labels (common in ML frameworks)
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive",
        # Alternative numeric mappings
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }
    return label_map.get(normalized, "Neutral")


@dataclass(frozen=True)
class SentimentResult:
    """
    Immutable result from a sentiment analysis prediction.

    This dataclass provides a standardized format for sentiment analysis
    results across all engines, ensuring consistent output regardless
    of the underlying model.

    Parameters
    ----------
    label : str
        Predicted sentiment label ('Positive', 'Neutral', 'Negative').
    score : float
        Confidence score for the predicted label (0.0 to 1.0).
    probs : Dict[str, float]
        Probability distribution over all sentiment classes.
    model : str
        Name of the model that produced this result.
    raw : Optional[Dict]
        Raw output from the underlying model (for debugging).

    Examples
    --------
    >>> result = SentimentResult(
    ...     label='Positive',
    ...     score=0.92,
    ...     probs={'Positive': 0.92, 'Neutral': 0.05, 'Negative': 0.03},
    ...     model='logreg'
    ... )
    >>> result.label
    'Positive'
    >>> result.to_dict()
    {'label': 'Positive', 'score': 0.92, ...}

    Notes
    -----
    The class is immutable (frozen=True) to ensure results cannot be
    accidentally modified after creation, which is important for
    reproducibility and thread safety.
    """

    label: str
    score: float
    probs: Dict[str, float]
    model: str
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the result.
        """
        return {
            "label": self.label,
            "score": self.score,
            "probs": self.probs,
            "model": self.model,
            "raw": self.raw,
        }

    def __repr__(self) -> str:
        return (
            f"SentimentResult(label='{self.label}', score={self.score:.4f}, "
            f"model='{self.model}')"
        )


def coerce_sentiment_result(
    result: Union[SentimentResult, Dict, Tuple], model_name: str
) -> SentimentResult:
    """
    Convert various result formats to a standardized SentimentResult.

    This function handles different output formats from various models
    and normalizes them to the SentimentResult format.

    Parameters
    ----------
    result : Union[SentimentResult, Dict, Tuple]
        Result in one of the following formats:
        - SentimentResult: Returned as-is
        - Dict with 'label' key: Converted to SentimentResult
        - Tuple of (label, score): Converted to SentimentResult

    model_name : str
        Name of the model that produced the result.

    Returns
    -------
    SentimentResult
        Standardized result object.

    Examples
    --------
    >>> result = coerce_sentiment_result({'label': 'pos', 'score': 0.9}, 'test')
    >>> result.label
    'Positive'

    >>> result = coerce_sentiment_result(('neg', 0.8), 'test')
    >>> result.label
    'Negative'
    """
    # Already a SentimentResult
    if isinstance(result, SentimentResult):
        return result

    # Dictionary format (common for most models)
    if isinstance(result, dict) and "label" in result:
        label = normalize_label(result.get("label"))
        probs = normalize_probs(result.get("probs", {label: 1.0}))
        score = float(result.get("score", 0.0))
        return SentimentResult(
            label=label,
            score=score,
            probs=probs,
            model=model_name,
            raw=result,
        )

    # Tuple format (label, score) - legacy support
    if isinstance(result, tuple) and len(result) == 2:
        label, score = result
        normalized_label = normalize_label(label)
        probs = normalize_probs({normalized_label: 1.0})
        return SentimentResult(
            label=normalized_label,
            score=float(score),
            probs=probs,
            model=model_name,
            raw={"legacy": True, "label": label, "score": score},
        )

    # Unknown format - return neutral with warning
    return SentimentResult(
        label="Neutral",
        score=0.0,
        probs=normalize_probs({}),
        model=model_name,
        raw={"unparsed_result": True, "original": str(result)},
    )


class BaseSentimentEngine:
    """
    Abstract base class for sentiment analysis engines.

    All sentiment engines should inherit from this class and implement
    the analyze() and batch_analyze() methods.

    Methods
    -------
    analyze(text)
        Analyze sentiment of a single text.
    batch_analyze(texts)
        Analyze sentiment of multiple texts efficiently.
    """

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
            Sentiment analysis result.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def batch_analyze(self, texts: list) -> list:
        """
        Analyze the sentiment of multiple texts.

        The default implementation calls analyze() for each text.
        Subclasses should override this for more efficient batch processing.

        Parameters
        ----------
        texts : List[str]
            List of texts to analyze.

        Returns
        -------
        List[SentimentResult]
            List of sentiment analysis results.
        """
        return [self.analyze(text) for text in texts]
