"""
Analysis utilities for sentiment analysis.

This module provides utility functions for:
- Probability normalization
- Confidence scoring using entropy
- Bootstrap confidence intervals
- Temporal sentiment aggregation

Mathematical Foundations
------------------------
Confidence is computed as the complement of normalized entropy:

    H(p) = -sum(p_i * log(p_i)) / log(n)
    confidence = 1 - H(p)

Where:
- p_i is the probability of class i
- n is the number of classes
- H(p) is the normalized entropy (0 to 1)

Higher confidence indicates the model is more certain about its prediction.
"""

import math
import random
import statistics
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Standard sentiment labels used throughout the system
SENTIMENT_LABELS: Tuple[str, str, str] = ("Positive", "Neutral", "Negative")


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize probability distribution to sum to 1.0.

    Parameters
    ----------
    probs : Dict[str, float]
        Raw probability values for each sentiment label.

    Returns
    -------
    Dict[str, float]
        Normalized probabilities that sum to 1.0.
        If input is empty or all zeros, returns uniform distribution.

    Examples
    --------
    >>> normalize_probs({"Positive": 0.6, "Neutral": 0.3, "Negative": 0.1})
    {'Positive': 0.6, 'Neutral': 0.3, 'Negative': 0.1}

    >>> normalize_probs({"Positive": 2.0, "Neutral": 1.0})
    {'Positive': 0.666..., 'Neutral': 0.333..., 'Negative': 0.0}
    """
    cleaned = {label: float(probs.get(label, 0.0)) for label in SENTIMENT_LABELS}
    total = sum(max(value, 0.0) for value in cleaned.values())
    if total <= 0.0:
        return {label: 1.0 / len(SENTIMENT_LABELS) for label in SENTIMENT_LABELS}
    return {label: max(value, 0.0) / total for label, value in cleaned.items()}


def entropy_from_probs(probs: Dict[str, float]) -> float:
    """
    Compute normalized entropy from probability distribution.

    The entropy is normalized by log(n) to produce a value in [0, 1],
    where 0 indicates perfect certainty and 1 indicates maximum uncertainty.

    Parameters
    ----------
    probs : Dict[str, float]
        Probability distribution over sentiment labels.

    Returns
    -------
    float
        Normalized entropy value in [0, 1].

    Notes
    -----
    The normalized entropy formula is:

        H_norm(p) = -sum(p_i * log(p_i)) / log(n)

    This normalizes entropy to [0, 1] regardless of the number of classes.
    """
    normalized = normalize_probs(probs)
    if not normalized:
        return 0.0
    log_n = math.log(len(normalized))
    if log_n == 0:
        return 0.0
    entropy = 0.0
    for prob in normalized.values():
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy / log_n


def confidence_from_probs(probs: Dict[str, float]) -> float:
    """
    Compute confidence score from probability distribution.

    Confidence is defined as 1 minus normalized entropy, giving a value
    in [0, 1] where 1 indicates perfect certainty.

    Parameters
    ----------
    probs : Dict[str, float]
        Probability distribution over sentiment labels.

    Returns
    -------
    float
        Confidence score in [0, 1].

    Examples
    --------
    >>> confidence_from_probs({"Positive": 1.0, "Neutral": 0.0, "Negative": 0.0})
    1.0  # Perfect confidence

    >>> confidence_from_probs({"Positive": 0.33, "Neutral": 0.33, "Negative": 0.34})
    ~0.0  # Near-uniform, low confidence
    """
    return 1.0 - entropy_from_probs(probs)


def aggregate_confidence_stats(
    confidences: List[float], threshold: float = 0.6
) -> Dict[str, float]:
    """
    Compute aggregate statistics for confidence scores.

    Parameters
    ----------
    confidences : List[float]
        List of confidence scores.
    threshold : float, optional
        Threshold below which a prediction is considered low-confidence.
        Default is 0.6.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - mean: Mean confidence score
        - median: Median confidence score
        - low_confidence_ratio: Fraction of predictions below threshold
        - threshold: The threshold used
    """
    if not confidences:
        return {
            "mean": 0.0,
            "median": 0.0,
            "low_confidence_ratio": 0.0,
            "threshold": threshold,
        }

    mean_value = statistics.mean(confidences)
    median_value = statistics.median(confidences)
    low_confidence = sum(1 for value in confidences if value < threshold)
    return {
        "mean": round(mean_value, 4),
        "median": round(median_value, 4),
        "low_confidence_ratio": round(low_confidence / len(confidences), 4),
        "threshold": threshold,
    }


def bootstrap_confidence_intervals(
    labels: List[str],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for sentiment proportions.

    This method uses the percentile bootstrap to estimate confidence
    intervals for the proportion of each sentiment class.

    Parameters
    ----------
    labels : List[str]
        List of sentiment labels from predictions.
    n_boot : int, optional
        Number of bootstrap samples. Default is 500.
    alpha : float, optional
        Significance level for confidence interval. Default is 0.05 (95% CI).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Confidence intervals for each sentiment label, with 'lower' and
        'upper' bounds.

    Notes
    -----
    The bootstrap confidence interval is computed using the percentile method:
    1. Resample labels with replacement n_boot times
    2. Compute proportion of each class in each resample
    3. Return the alpha/2 and 1-alpha/2 percentiles

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
    CRC press.

    Examples
    --------
    >>> labels = ["Positive"] * 60 + ["Neutral"] * 30 + ["Negative"] * 10
    >>> intervals = bootstrap_confidence_intervals(labels)
    >>> intervals["Positive"]
    {'lower': 0.52, 'upper': 0.68}  # 95% CI for positive proportion
    """
    if not labels:
        return {label: {"lower": 0.0, "upper": 0.0} for label in SENTIMENT_LABELS}

    n_boot = max(1, int(n_boot))
    rng = random.Random(seed)
    total = len(labels)
    samples = {label: [] for label in SENTIMENT_LABELS}

    for _ in range(n_boot):
        sample = [rng.choice(labels) for _ in range(total)]
        counts = Counter(sample)
        for label in SENTIMENT_LABELS:
            samples[label].append(counts.get(label, 0) / total)

    intervals = {}
    lower_idx = int((alpha / 2) * (n_boot - 1))
    upper_idx = int((1 - alpha / 2) * (n_boot - 1))
    for label, values in samples.items():
        values.sort()
        intervals[label] = {
            "lower": round(values[lower_idx], 4),
            "upper": round(values[upper_idx], 4),
        }
    return intervals


def build_hourly_sentiment(
    comments: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """
    Aggregate sentiment counts by hour.

    Parameters
    ----------
    comments : List[Dict]
        List of comment dictionaries with 'published_at' and 'sentiment' keys.

    Returns
    -------
    Dict[str, Dict[str, int]]
        Dictionary mapping ISO-formatted hour strings to sentiment counts.
        Sorted chronologically.

    Examples
    --------
    >>> comments = [
    ...     {"published_at": "2024-01-15T10:30:00Z", "sentiment": "Positive"},
    ...     {"published_at": "2024-01-15T10:45:00Z", "sentiment": "Positive"},
    ...     {"published_at": "2024-01-15T11:00:00Z", "sentiment": "Negative"},
    ... ]
    >>> build_hourly_sentiment(comments)
    {
        '2024-01-15T10:00:00': {'Positive': 2},
        '2024-01-15T11:00:00': {'Negative': 1}
    }
    """
    from collections import defaultdict
    from datetime import datetime

    def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    buckets = defaultdict(Counter)
    for item in comments:
        timestamp = parse_timestamp(item.get("published_at"))
        if not timestamp:
            continue
        hour_bucket = timestamp.replace(minute=0, second=0, microsecond=0)
        sentiment = item.get("sentiment", "Neutral")
        buckets[hour_bucket.isoformat()][sentiment] += 1

    return {
        hour: dict(counter)
        for hour, counter in sorted(buckets.items(), key=lambda pair: pair[0])
    }
