from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    hf_name: str
    description: str
    default_max_length: int = 256
    multilingual: bool = False


_ALIASES = {
    "deberta-v3": "deberta_v3",
    "xlm-v": "xlm_v",
    "mdeberta-v3": "mdeberta_v3",
}

_REGISTRY: Dict[str, EncoderSpec] = {
    "bert": EncoderSpec(
        key="bert",
        hf_name="bert-base-uncased",
        description="BERT baseline encoder",
        default_max_length=256,
    ),
    "modernbert": EncoderSpec(
        key="modernbert",
        hf_name="answerdotai/ModernBERT-base",
        description="ModernBERT English encoder baseline",
        default_max_length=256,
    ),
    "deberta_v3": EncoderSpec(
        key="deberta_v3",
        hf_name="microsoft/deberta-v3-base",
        description="DeBERTa-v3 English encoder baseline",
        default_max_length=256,
    ),
    "xlm_v": EncoderSpec(
        key="xlm_v",
        hf_name="facebook/xlm-v-base",
        description="XLM-V multilingual encoder baseline",
        default_max_length=256,
        multilingual=True,
    ),
    "mdeberta_v3": EncoderSpec(
        key="mdeberta_v3",
        hf_name="microsoft/mdeberta-v3-base",
        description="mDeBERTa-v3 multilingual encoder baseline",
        default_max_length=256,
        multilingual=True,
    ),
}


def normalize_encoder_key(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return _ALIASES.get(normalized, normalized)


def get_encoder_spec(value: str) -> EncoderSpec:
    key = normalize_encoder_key(value)
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown encoder preset '{value}'. "
            f"Available presets: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def list_encoder_presets() -> List[str]:
    return sorted(_REGISTRY)
