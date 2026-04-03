from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ProbabilityCubeBundle:
    prob_cube: np.ndarray
    model_names: List[str]
    labels: List[str]
    y_true: List[str]
    texts: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    logits_cube: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def _as_string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=str)


def validate_probability_cube(
    prob_cube: np.ndarray,
    *,
    model_names: Sequence[str],
    labels: Sequence[str],
    y_true: Sequence[str],
    texts: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
    logits_cube: Optional[np.ndarray] = None,
) -> None:
    if prob_cube.ndim != 3:
        raise ValueError(
            f"Probability cube must have shape (n_models, n_samples, n_classes); got {prob_cube.shape}"
        )

    n_models, n_samples, n_classes = prob_cube.shape
    if len(model_names) != n_models:
        raise ValueError(
            f"Model name count ({len(model_names)}) does not match probability cube axis 0 ({n_models})"
        )
    if len(labels) != n_classes:
        raise ValueError(
            f"Label count ({len(labels)}) does not match probability cube axis 2 ({n_classes})"
        )
    if len(y_true) != n_samples:
        raise ValueError(
            f"Ground-truth label count ({len(y_true)}) does not match probability cube axis 1 ({n_samples})"
        )
    if texts is not None and len(texts) != n_samples:
        raise ValueError(
            f"Text count ({len(texts)}) does not match probability cube axis 1 ({n_samples})"
        )
    if sample_ids is not None and len(sample_ids) != n_samples:
        raise ValueError(
            f"Sample id count ({len(sample_ids)}) does not match probability cube axis 1 ({n_samples})"
        )
    if logits_cube is not None and logits_cube.shape != prob_cube.shape:
        raise ValueError(
            f"Logit cube shape {logits_cube.shape} must match probability cube shape {prob_cube.shape}"
        )


def save_probability_cube(
    path: Path | str,
    *,
    prob_cube: np.ndarray,
    model_names: Sequence[str],
    labels: Sequence[str],
    y_true: Sequence[str],
    texts: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
    logits_cube: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prob_array = np.asarray(prob_cube, dtype=np.float32)
    logits_array = None if logits_cube is None else np.asarray(logits_cube, dtype=np.float32)

    validate_probability_cube(
        prob_array,
        model_names=model_names,
        labels=labels,
        y_true=y_true,
        texts=texts,
        sample_ids=sample_ids,
        logits_cube=logits_array,
    )

    payload = {
        "prob_cube": prob_array,
        "model_names": _as_string_array(model_names),
        "labels": _as_string_array(labels),
        "y_true": _as_string_array(y_true),
        "metadata_json": np.asarray(
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            dtype=str,
        ),
    }

    if texts is not None:
        payload["texts"] = _as_string_array(texts)
    if sample_ids is not None:
        payload["sample_ids"] = _as_string_array(sample_ids)
    if logits_array is not None:
        payload["logits_cube"] = logits_array

    np.savez_compressed(output_path, **payload)
    return output_path


def load_probability_cube(path: Path | str) -> ProbabilityCubeBundle:
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as payload:
        prob_cube = np.asarray(payload["prob_cube"], dtype=np.float32)
        model_names = payload["model_names"].astype(str).tolist()
        labels = payload["labels"].astype(str).tolist()
        y_true = payload["y_true"].astype(str).tolist()
        texts = payload["texts"].astype(str).tolist() if "texts" in payload else None
        sample_ids = (
            payload["sample_ids"].astype(str).tolist()
            if "sample_ids" in payload else None
        )
        logits_cube = (
            np.asarray(payload["logits_cube"], dtype=np.float32)
            if "logits_cube" in payload else None
        )
        metadata_json = (
            str(payload["metadata_json"].item())
            if "metadata_json" in payload else "{}"
        )

    metadata = json.loads(metadata_json or "{}")
    validate_probability_cube(
        prob_cube,
        model_names=model_names,
        labels=labels,
        y_true=y_true,
        texts=texts,
        sample_ids=sample_ids,
        logits_cube=logits_cube,
    )

    return ProbabilityCubeBundle(
        prob_cube=prob_cube,
        model_names=model_names,
        labels=labels,
        y_true=y_true,
        texts=texts,
        sample_ids=sample_ids,
        logits_cube=logits_cube,
        metadata=metadata,
    )
