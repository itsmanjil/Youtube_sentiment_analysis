import math
import random
import statistics
from collections import Counter

SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")


def normalize_probs(probs):
    cleaned = {label: float(probs.get(label, 0.0)) for label in SENTIMENT_LABELS}
    total = sum(max(value, 0.0) for value in cleaned.values())
    if total <= 0.0:
        return {label: 1.0 / len(SENTIMENT_LABELS) for label in SENTIMENT_LABELS}
    return {label: max(value, 0.0) / total for label, value in cleaned.items()}


def entropy_from_probs(probs):
    normalized = normalize_probs(probs)
    if not normalized:
        return 0.0
    log_n = math.log(len(normalized))
    if log_n == 0:
        return 0.0
    entropy = 0.0
    for prob in normalized.values():
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy / log_n


def confidence_from_probs(probs):
    return 1.0 - entropy_from_probs(probs)


def aggregate_confidence_stats(confidences, threshold=0.6):
    if not confidences:
        return {
            "mean": 0.0,
            "median": 0.0,
            "low_confidence_ratio": 0.0,
            "threshold": threshold,
        }

    mean_value = statistics.mean(confidences)
    median_value = statistics.median(confidences)
    low_confidence = sum(1 for value in confidences if value < threshold)
    return {
        "mean": round(mean_value, 4),
        "median": round(median_value, 4),
        "low_confidence_ratio": round(low_confidence / len(confidences), 4),
        "threshold": threshold,
    }


def bootstrap_confidence_intervals(labels, n_boot=500, alpha=0.05, seed=42):
    if not labels:
        return {
            label: {"lower": 0.0, "upper": 0.0}
            for label in SENTIMENT_LABELS
        }

    n_boot = max(1, int(n_boot))
    rng = random.Random(seed)
    total = len(labels)
    samples = {label: [] for label in SENTIMENT_LABELS}

    for _ in range(n_boot):
        sample = [rng.choice(labels) for _ in range(total)]
        counts = Counter(sample)
        for label in SENTIMENT_LABELS:
            samples[label].append(counts.get(label, 0) / total)

    intervals = {}
    lower_idx = int((alpha / 2) * (n_boot - 1))
    upper_idx = int((1 - alpha / 2) * (n_boot - 1))
    for label, values in samples.items():
        values.sort()
        intervals[label] = {
            "lower": round(values[lower_idx], 4),
            "upper": round(values[upper_idx], 4),
        }
    return intervals


def build_hourly_sentiment(comments):
    from collections import defaultdict
    from datetime import datetime

    def parse_timestamp(value):
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    buckets = defaultdict(Counter)
    for item in comments:
        timestamp = parse_timestamp(item.get("published_at"))
        if not timestamp:
            continue
        hour_bucket = timestamp.replace(minute=0, second=0, microsecond=0)
        sentiment = item.get("sentiment", "Neutral")
        buckets[hour_bucket.isoformat()][sentiment] += 1

    return {
        hour: dict(counter)
        for hour, counter in sorted(buckets.items(), key=lambda pair: pair[0])
    }
