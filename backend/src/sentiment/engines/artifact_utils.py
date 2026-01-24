"""
Helpers for loading serialized model artifacts with actionable errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.utils.config import Config


def _read_metadata(model_name: str) -> Optional[dict]:
    metadata_path = Config.MODELS_DIR / model_name / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_expected_sklearn_version(model_name: str) -> Optional[str]:
    metadata = _read_metadata(model_name)
    if not metadata:
        return None
    return metadata.get("sklearn_version")


def format_model_load_error(
    model_name: str,
    model_path: Path,
    vectorizer_path: Optional[Path],
    exc: Exception,
) -> str:
    expected_version = get_expected_sklearn_version(model_name)
    expected_note = f" Expected scikit-learn {expected_version}." if expected_version else ""
    paths = f" Model path: {model_path}."
    if vectorizer_path is not None:
        paths = f"{paths} Vectorizer path: {vectorizer_path}."
    return (
        f"Failed to load {model_name} model artifacts.{expected_note} "
        "This usually means scikit-learn/numpy is missing or incompatible with the "
        f"model files.{paths} Original error: {exc}"
    )
