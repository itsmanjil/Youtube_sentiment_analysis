"""
Transformer training utilities for Route A.
"""

from .model_registry import (
    EncoderSpec,
    get_encoder_spec,
    list_encoder_presets,
)

__all__ = [
    "EncoderSpec",
    "get_encoder_spec",
    "list_encoder_presets",
]
