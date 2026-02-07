"""
Classical-model text preprocessing (TF-IDF baselines and derived ensembles).

Why this exists
---------------
The codebase previously applied extra preprocessing steps (stopword removal,
negation handling, etc.) only inside the Django API layer, while training and
evaluation scripts used raw `text` from CSV. That creates train/inference skew,
which is a thesis-grade validity risk.

This module centralizes the preprocessing so it can be applied consistently
in:
  - `src.sentiment.engines.*` (inference)
  - `backend/train_*.py` (training)
  - `backend/research/*` (evaluation/optimization)

Design goals
------------
1. Deterministic: avoid network downloads at runtime; degrade gracefully if
   NLTK corpora are missing.
2. Compatible with upstream cleaning: inputs are often already cleaned by
   `YouTubePreprocessor` (letters/spaces). Contraction expansion therefore
   targets apostrophe-stripped forms like "dont" and "cant".
3. Sentiment-safe stopwords: keep negators ("not", "no", "never") to avoid
   destroying sentiment cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence


NEGATORS = {"not", "no", "never", "nor"}

# Minimal fallback (enough to reduce noise without requiring NLTK downloads).
_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}

# Apostrophe-stripped, high-signal negation contractions commonly seen after
# aggressive cleaning (e.g., "don't" -> "dont").
_NEGATION_EXPANSIONS = {
    "aint": "am not",
    "arent": "are not",
    "cant": "can not",
    "cannot": "can not",
    "couldnt": "could not",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hasnt": "has not",
    "havent": "have not",
    "isnt": "is not",
    "mightnt": "might not",
    "mustnt": "must not",
    "neednt": "need not",
    "shant": "shall not",
    "shouldnt": "should not",
    "wasnt": "was not",
    "werent": "were not",
    "wont": "will not",
    "wouldnt": "would not",
}


@dataclass(frozen=True)
class ClassicalPreprocessConfig:
    """
    Configuration for classical-model preprocessing.

    Parameters
    ----------
    expand_negation_contractions:
        Expand apostrophe-stripped negation contractions ("dont" -> "do not").
    negation_tag:
        Apply simple negation tagging ("not good" -> "not_good") using a
        1-token window. This is robust and avoids dependence on WordNet.
    remove_stopwords:
        Remove stopwords while preserving negators and negation-tagged tokens.
    """

    expand_negation_contractions: bool = True
    negation_tag: bool = True
    remove_stopwords: bool = True


@lru_cache(maxsize=1)
def _get_stopwords() -> set[str]:
    """Return a stopword set, falling back to a small built-in list."""
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            # Avoid forcing `nltk.download(...)` at runtime.
            return set(_FALLBACK_STOPWORDS)
    except Exception:
        return set(_FALLBACK_STOPWORDS)


def _expand_negation_contractions(tokens: List[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        replacement = _NEGATION_EXPANSIONS.get(token)
        if replacement:
            expanded.extend(replacement.split())
        else:
            expanded.append(token)
    return expanded


def _apply_negation_tag(tokens: List[str]) -> List[str]:
    """
    Negation tagging with a 1-token window.

    Example:
      ["this","is","not","good"] -> ["this","is","not_good"]
    """
    tagged: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in NEGATORS and idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if nxt:
                tagged.append(f"not_{nxt}")
            else:
                tagged.append(token)
            idx += 2
            continue
        tagged.append(token)
        idx += 1
    return tagged


def _remove_stopwords(tokens: List[str], stopwords: set[str]) -> List[str]:
    filtered: List[str] = []
    for token in tokens:
        if not token:
            continue
        if token.startswith("not_"):
            filtered.append(token)
            continue
        if token in NEGATORS:
            filtered.append(token)
            continue
        if token in stopwords:
            continue
        filtered.append(token)
    return filtered


def preprocess_classical_text(
    text: str,
    config: ClassicalPreprocessConfig | None = None,
) -> str:
    """
    Preprocess a single text for classical (TF-IDF) models.

    Notes
    -----
    This function assumes upstream cleaning may already have removed punctuation.
    It therefore focuses on robust, post-clean steps that remain useful:
      - negation contraction expansion for apostrophe-stripped tokens
      - negation tagging
      - stopword removal (with negator preservation)
    """
    if config is None:
        config = ClassicalPreprocessConfig()

    if text is None:
        return ""

    # Normalize whitespace and lowercase (cheap + deterministic).
    normalized = " ".join(str(text).lower().split())
    if not normalized:
        return ""

    tokens = normalized.split()

    if config.expand_negation_contractions:
        tokens = _expand_negation_contractions(tokens)

    if config.negation_tag:
        tokens = _apply_negation_tag(tokens)

    if config.remove_stopwords:
        stopwords = _get_stopwords()
        if stopwords:
            tokens = _remove_stopwords(tokens, stopwords)

    return " ".join(tokens)


def preprocess_classical_texts(
    texts: Sequence[str],
    config: ClassicalPreprocessConfig | None = None,
) -> List[str]:
    """Preprocess a batch of texts for classical models."""
    if not texts:
        return []
    return [preprocess_classical_text(text, config=config) for text in texts]
