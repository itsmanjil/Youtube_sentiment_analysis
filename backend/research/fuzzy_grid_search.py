import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_base_sets(value):
    sets = []
    for block in value.split(";"):
        models = [item.strip().lower() for item in block.split(",") if item.strip()]
        if models:
            sets.append(models)
    return sets


def evaluate_engine(engine, texts, labels):
    predictions = []
    for text in texts:
        result = engine.analyze(text)
        predictions.append(result.label)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "macro_f1": round(f1_score(labels, predictions, average="macro"), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid search fuzzy ensemble configs on a labeled dataset."
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
        "--base-sets",
        default="logreg,svm;logreg,svm,tfidf",
        help="Base model sets separated by ';' (e.g. 'logreg,svm;logreg,svm,tfidf').",
    )
    parser.add_argument(
        "--mf-types",
        default="gaussian,triangular,trapezoidal",
    )
    parser.add_argument(
        "--defuzz-methods",
        default="centroid,bisector,mom,som,lom,weighted_average",
    )
    parser.add_argument(
        "--t-norms",
        default="min,product,lukasiewicz",
    )
    parser.add_argument(
        "--t-conorms",
        default="max,prob_sum,bounded_sum",
    )
    parser.add_argument("--alpha-cuts", default="0.0")
    parser.add_argument("--resolutions", default="100")
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

    base_sets = parse_base_sets(args.base_sets)
    mf_types = parse_list(args.mf_types)
    defuzz_methods = parse_list(args.defuzz_methods)
    t_norms = parse_list(args.t_norms)
    t_conorms = parse_list(args.t_conorms)
    alpha_cuts = [float(x) for x in parse_list(args.alpha_cuts)]
    resolutions = [int(x) for x in parse_list(args.resolutions)]

    results = []
    total = (
        len(base_sets)
        * len(mf_types)
        * len(defuzz_methods)
        * len(t_norms)
        * len(t_conorms)
        * len(alpha_cuts)
        * len(resolutions)
    )

    config_idx = 0
    for base_models in base_sets:
        base_engines = {name: get_sentiment_engine(name) for name in base_models}

        for mf_type, defuzz_method, t_norm, t_conorm, alpha_cut, resolution in product(
            mf_types,
            defuzz_methods,
            t_norms,
            t_conorms,
            alpha_cuts,
            resolutions,
        ):
            config_idx += 1
            engine = FuzzySentimentEngine(
                base_engines=base_engines,
                mf_type=mf_type,
                defuzz_method=defuzz_method,
                t_norm=t_norm,
                t_conorm=t_conorm,
                alpha_cut=alpha_cut,
                resolution=resolution,
                confidence_threshold=args.confidence_threshold,
            )
            metrics = evaluate_engine(engine, texts, labels)
            results.append(
                {
                    "config": {
                        "base_models": base_models,
                        "mf_type": mf_type,
                        "defuzz_method": defuzz_method,
                        "t_norm": t_norm,
                        "t_conorm": t_conorm,
                        "alpha_cut": alpha_cut,
                        "resolution": resolution,
                    },
                    "metrics": metrics,
                }
            )
            print(
                f"[{config_idx}/{total}] f1={metrics['macro_f1']} acc={metrics['accuracy']}"
            )

    results.sort(key=lambda item: item["metrics"]["macro_f1"], reverse=True)

    payload = {
        "results": results,
        "best": results[0] if results else None,
        "meta": {
            "use_full": args.use_full,
            "limit": args.limit,
            "confidence_threshold": args.confidence_threshold,
        },
    }

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
