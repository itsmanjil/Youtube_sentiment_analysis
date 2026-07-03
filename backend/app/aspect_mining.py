"""
Keyword-level sentiment aggregation (aspect proxy).

NOTE: This module does NOT implement full Aspect-Based Sentiment Analysis (ABSA).
It uses token-frequency counting to surface the most common content words and
aggregates the sentiment of comments that contain each word. This is a lightweight
proxy for aspect analysis, not opinion-target extraction or aspect-opinion linking.

For full ABSA (aspect-opinion triplet extraction), see the pyabsa library or
transformer-based models such as ASTE-Transformer (Xu et al., 2021).
"""

import re
from collections import Counter, defaultdict

from src.preprocessing import get_fallback_stopwords


def _get_stopwords():
    # Reuse the vendored, deterministic stopword list from
    # `src.preprocessing.classical` rather than loading `nltk.corpus.stopwords`
    # at runtime: whether that corpus is downloaded varies by machine, which
    # would make the same comment set produce different top-aspect words
    # depending on environment.
    return get_fallback_stopwords()


def _tokenize(text):
    return re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())


def extract_aspect_sentiment(comments, top_n=12, min_freq=3):
    stopwords = _get_stopwords()
    aspect_counts = Counter()
    aspect_sentiments = defaultdict(Counter)

    for item in comments:
        text = (
            item.get("processed_text_classical")
            or item.get("processed_text")
            or item.get("text", "")
        )
        sentiment = item.get("sentiment", "Neutral")
        tokens = [
            token
            for token in _tokenize(text)
            if token not in stopwords and len(token) > 2
        ]
        unique_tokens = set(tokens)
        for token in unique_tokens:
            aspect_counts[token] += 1
            aspect_sentiments[token][sentiment] += 1

    aspects = []
    for token, count in aspect_counts.most_common():
        if count < min_freq:
            continue
        sentiment_counts = aspect_sentiments[token]
        total = sum(sentiment_counts.values()) or 1
        aspect = {
            "aspect": token,
            "count": count,
            "sentiment": dict(sentiment_counts),
            "ratio": {
                label: round(sentiment_counts.get(label, 0) / total, 4)
                for label in ("Positive", "Neutral", "Negative")
            },
        }
        aspects.append(aspect)
        if len(aspects) >= top_n:
            break

    return aspects
