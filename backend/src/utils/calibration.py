"""
Calibration helpers shared by research scripts and runtime inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


def _ensure_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = _ensure_2d(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    return logits_to_probs(_ensure_2d(logits) / safe_temperature)


def negative_log_likelihood_from_logits(
    logits: np.ndarray,
    label_ids: Sequence[int],
    temperature: float = 1.0,
) -> float:
    probs = apply_temperature_to_logits(logits, temperature)
    label_ids = np.asarray(label_ids, dtype=int)
    eps = 1e-10
    return float(-np.mean(np.log(probs[np.arange(len(label_ids)), label_ids] + eps)))


def fit_temperature_from_logits(
    logits: np.ndarray,
    label_ids: Sequence[int],
    t_min: float = 0.05,
    t_max: float = 10.0,
) -> float:
    def objective(temperature: float) -> float:
        return negative_log_likelihood_from_logits(logits, label_ids, temperature)

    result = minimize_scalar(objective, bounds=(t_min, t_max), method="bounded")
    return float(result.x)


def load_temperature_artifact(path: Optional[str | Path]) -> Optional[dict]:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def save_temperature_artifact(path: str | Path, payload: dict) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path
