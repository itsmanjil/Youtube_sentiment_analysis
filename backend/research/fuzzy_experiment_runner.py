import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.sentiment import get_sentiment_engine, normalize_label
from research.computational_intelligence.fuzzy.engine_integration import (
    FuzzySentimentEngine,
)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].apply(normalize_label)
    return df


def build_fuzzy_engine(base_models, mf_type, defuzz_method, confidence_threshold):
    base_engines = {}
    for model in base_models:
        base_engines[model] = get_sentiment_engine(model)

    return FuzzySentimentEngine(
        base_engines=base_engines,
        mf_type=mf_type,
        defuzz_method=defuzz_method,
        confidence_threshold=confidence_threshold,
    )


def evaluate_engine(engine, texts, labels):
    predictions = []
    for text in texts:
        result = engine.analyze(text)
        predictions.append(result.label)

    metrics = {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "macro_f1": round(f1_score(labels, predictions, average="macro"), 4),
        "report": classification_report(labels, predictions, output_dict=True),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run fuzzy sentiment ensemble experiments on a labeled dataset."
    )
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="Evaluate on the full dataset without re-splitting.",
    )
    parser.add_argument(
        "--base-models",
        default="logreg,svm,tfidf",
        help="Comma-separated base models for the fuzzy ensemble.",
    )
    parser.add_argument("--mf-type", default="gaussian")
    parser.add_argument("--defuzz-method", default="centroid")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    if args.use_full:
        test_df = df
    else:
        _, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=df["label"],
        )

    if args.limit:
        test_df = test_df.head(args.limit)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    base_models = [
        name.strip().lower() for name in args.base_models.split(",") if name.strip()
    ]

    engine = build_fuzzy_engine(
        base_models,
        mf_type=args.mf_type,
        defuzz_method=args.defuzz_method,
        confidence_threshold=args.confidence_threshold,
    )

    results = {
        "fuzzy_ensemble": evaluate_engine(engine, texts, labels),
        "config": {
            "base_models": base_models,
            "mf_type": args.mf_type,
            "defuzz_method": args.defuzz_method,
            "confidence_threshold": args.confidence_threshold,
            "use_full": args.use_full,
            "limit": args.limit,
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
