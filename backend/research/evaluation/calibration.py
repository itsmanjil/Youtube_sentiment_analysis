"""
Calibration metrics for probabilistic classifiers.

Implements Expected Calibration Error (ECE) and multiclass Brier score.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from src.utils import SENTIMENT_LABELS
except Exception:
    SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")


def _normalize_rows(prob_matrix: np.ndarray) -> np.ndarray:
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    if prob_matrix.ndim == 1:
        prob_matrix = prob_matrix.reshape(1, -1)

    prob_matrix = np.clip(prob_matrix, 0.0, None)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return prob_matrix / row_sums


def probs_to_matrix(
    probs_list: Iterable[dict],
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> np.ndarray:
    rows = []
    for probs in probs_list:
        row = [float(probs.get(label, 0.0)) for label in labels]
        rows.append(row)
    return _normalize_rows(np.asarray(rows, dtype=float))


def _label_to_index(labels: Sequence[str]) -> dict:
    return {label: idx for idx, label in enumerate(labels)}


def compute_brier_multiclass(
    y_true: Sequence[str],
    y_prob: np.ndarray,
    labels: Sequence[str],
) -> float:
    y_prob = _normalize_rows(y_prob)
    label_to_index = _label_to_index(labels)
    y_true_idx = np.array([label_to_index[label] for label in y_true], dtype=int)

    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_ece(
    y_true: Sequence[str],
    y_prob: np.ndarray,
    labels: Sequence[str],
    n_bins: int = 15,
) -> float:
    y_prob = _normalize_rows(y_prob)
    label_to_index = _label_to_index(labels)
    y_true_idx = np.array([label_to_index[label] for label in y_true], dtype=int)

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)

    for bin_idx in range(n_bins):
        start = bin_edges[bin_idx]
        end = bin_edges[bin_idx + 1]
        if bin_idx == 0:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences > start) & (confidences <= end)
        if not np.any(mask):
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(accuracies[mask].mean())
        ece += abs(bin_accuracy - bin_confidence) * (mask.sum() / n_samples)
    return float(ece)


def compute_calibration_metrics(
    y_true: Sequence[str],
    y_prob: np.ndarray,
    labels: Sequence[str] = SENTIMENT_LABELS,
    n_bins: int = 15,
) -> Tuple[float, float]:
    ece = compute_ece(y_true, y_prob, labels=labels, n_bins=n_bins)
    brier = compute_brier_multiclass(y_true, y_prob, labels=labels)
    return ece, brier

