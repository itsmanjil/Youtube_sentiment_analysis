import re
from collections import Counter, defaultdict


def _get_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with",
        }


def _tokenize(text):
    return re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())


def extract_aspect_sentiment(comments, top_n=12, min_freq=3):
    stopwords = _get_stopwords()
    aspect_counts = Counter()
    aspect_sentiments = defaultdict(Counter)

    for item in comments:
        text = item.get("processed_text") or item.get("text", "")
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
