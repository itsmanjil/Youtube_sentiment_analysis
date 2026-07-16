"""
Text preprocessing utilities shared across training and inference.

This package intentionally lives under `src/` so it can be imported from:
  - sentiment engines (inference)
  - training/evaluation scripts (research + CLI)
without pulling in any web-framework dependencies.
"""

from .classical import (
    ClassicalPreprocessConfig,
    get_fallback_stopwords,
    preprocess_classical_text,
    preprocess_classical_texts,
)

__all__ = [
    "ClassicalPreprocessConfig",
    "get_fallback_stopwords",
    "preprocess_classical_text",
    "preprocess_classical_texts",
]

