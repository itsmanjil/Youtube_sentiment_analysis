import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.sentiment_engines import coerce_sentiment_result, get_sentiment_engine, normalize_label


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].apply(normalize_label)
    return df


def _load_ensemble_weights(value):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / raw
        if not path.exists():
            alt_path = BASE_DIR / raw
            if alt_path.exists():
                path = alt_path
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            return None

    if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
        return payload["weights"]
    if isinstance(payload, dict):
        return payload
    return None


def evaluate_engine(engine_type, texts, labels, **kwargs):
    engine = get_sentiment_engine(engine_type, **kwargs)
    results = engine.batch_analyze(texts)
    predictions = [
        coerce_sentiment_result(result, engine_type).label
        for result in results
    ]

    metrics = {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "macro_f1": round(f1_score(labels, predictions, average="macro"), 4),
        "report": classification_report(labels, predictions, output_dict=True),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment model experiments on a labeled dataset."
    )
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="logreg,svm,tfidf,ensemble",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--ensemble-models",
        default="logreg,svm,tfidf",
        help="Comma-separated base models for the ensemble.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help="Optional JSON dict or a path to a JSON weights file.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    _, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df["label"],
    )

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    model_list = [
        name.strip().lower() for name in args.models.split(",") if name.strip()
    ]
    ensemble_models = [
        name.strip().lower()
        for name in args.ensemble_models.split(",")
        if name.strip()
    ]
    ensemble_weights = _load_ensemble_weights(args.ensemble_weights)
    if args.ensemble_weights and ensemble_weights is None:
        raise ValueError(
            "Invalid ensemble_weights. Provide JSON or a path to a weights file."
        )

    results = {}
    for model in model_list:
        if model == "ensemble":
            metrics = evaluate_engine(
                "ensemble",
                texts,
                labels,
                base_models=ensemble_models,
                weights=ensemble_weights,
            )
        else:
            metrics = evaluate_engine(model, texts, labels)
        results[model] = metrics

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
