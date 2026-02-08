import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from research.evaluation.calibration import compute_calibration_metrics, probs_to_matrix
from src.utils import SENTIMENT_LABELS


def load_dataset(csv_path):
    # `gold_set_template.csv` intentionally has empty labels. Pandas parses empty
    # CSV cells as NaN by default, which would otherwise drop every row and later
    # crash downstream models with a confusing "0 samples" error. We read with
    # keep_default_na=False and explicitly validate labels.
    df = pd.read_csv(
        csv_path,
        dtype={"text": "string", "label": "string"},
        keep_default_na=False,
    )
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Drop empty texts.
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().astype(bool)]

    # Treat blank/whitespace labels as missing (unlabeled) and drop them.
    df["label"] = df["label"].fillna("").astype(str)
    unlabeled_mask = df["label"].str.strip().eq("")
    unlabeled_count = int(unlabeled_mask.sum())
    if unlabeled_count:
        df = df[~unlabeled_mask]

    # Normalize labels to the repo's standard set.
    df["label"] = df["label"].apply(normalize_label)

    if df.empty:
        hint = (
            "No labeled rows found after filtering.\n\n"
            "If you're evaluating a gold set template, fill the 'label' column first "
            "with: Positive / Neutral / Negative.\n"
        )
        if unlabeled_count:
            hint += f"\nUnlabeled rows detected: {unlabeled_count}"
        raise ValueError(hint)

    return df.reset_index(drop=True)


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


def evaluate_engine(engine_type, texts, labels, calibration_bins=15, **kwargs):
    engine = get_sentiment_engine(engine_type, **kwargs)
    results = engine.batch_analyze(texts)
    predictions = [
        coerce_sentiment_result(result, engine_type).label
        for result in results
    ]
    probs_list = [
        coerce_sentiment_result(result, engine_type).probs
        for result in results
    ]
    prob_matrix = probs_to_matrix(probs_list, labels=SENTIMENT_LABELS)
    ece, brier = compute_calibration_metrics(
        labels,
        prob_matrix,
        labels=SENTIMENT_LABELS,
        n_bins=calibration_bins,
    )

    metrics = {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "macro_f1": round(f1_score(labels, predictions, average="macro"), 4),
        "report": classification_report(labels, predictions, output_dict=True),
        "calibration": {
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "bins": calibration_bins,
        },
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
        "--use-full",
        action="store_true",
        help="Evaluate on the full dataset without re-splitting.",
    )
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
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        help="Number of bins for Expected Calibration Error (default: 15).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Enable the shared classical preprocessing (negation expansion/tagging, "
            "stopword removal) for classical models and derived ensembles. "
            "Use this only if you also trained the underlying models with preprocessing enabled."
        ),
    )
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
        base_kwargs = {}
        if args.preprocess and model in {
            "tfidf",
            "logreg",
            "svm",
            "ensemble",
            "meta_learner",
            "fuzzy_ensemble",
        }:
            base_kwargs["preprocess"] = True
        if model == "ensemble":
            metrics = evaluate_engine(
                "ensemble",
                texts,
                labels,
                calibration_bins=args.calibration_bins,
                base_models=ensemble_models,
                weights=ensemble_weights,
                **base_kwargs,
            )
        else:
            metrics = evaluate_engine(
                model,
                texts,
                labels,
                calibration_bins=args.calibration_bins,
                **base_kwargs,
            )
        results[model] = metrics

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
