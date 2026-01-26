import argparse
import inspect
import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


LABELS = ("Negative", "Neutral", "Positive")


def _parse_ngram(value):
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("ngram_range must be in the form '1,2'")
    return int(parts[0]), int(parts[1])


def _parse_class_weight(value):
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("none", ""):
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError("class_weight must be 'balanced' or 'none'")


def load_dataset(
    csv_path,
    text_column="CommentText",
    label_column="Sentiment",
    max_per_class=None,
    random_seed=42,
):
    df = pd.read_csv(
        csv_path,
        usecols=[text_column, label_column],
        dtype={text_column: "string", label_column: "string"},
    )
    df = df.dropna(subset=[text_column, label_column])
    df = df.rename(columns={text_column: "text", label_column: "label"})
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)]

    if max_per_class is not None:
        df = (
        df.groupby("label", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(len(group), max_per_class),
                    random_state=random_seed,
                )
            )
            .reset_index(drop=True)
        )

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def train_model(train_df, test_df, args):
    train_texts = train_df["text"]
    train_labels = train_df["label"]
    test_texts = test_df["text"]
    test_labels = test_df["label"]

    vectorizer = TfidfVectorizer(
        lowercase=not args.no_lowercase,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=_parse_ngram(args.ngram_range),
        strip_accents="unicode" if args.strip_accents else None,
    )

    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    solver = args.solver.lower()
    model_kwargs = {
        "C": args.C,
        "max_iter": args.max_iter,
        "solver": solver,
        "n_jobs": args.n_jobs,
        "class_weight": _parse_class_weight(args.class_weight),
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        if solver in ("lbfgs", "newton-cg", "sag", "saga"):
            model_kwargs["multi_class"] = "multinomial"
        else:
            model_kwargs["multi_class"] = "ovr"

    model = LogisticRegression(**model_kwargs)
    model.fit(train_vectors, train_labels)

    predictions = model.predict(test_vectors)
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "f1_macro": f1_score(test_labels, predictions, average="macro"),
        "report": classification_report(test_labels, predictions, output_dict=True),
    }

    return model, vectorizer, metrics


def save_outputs(output_dir, model, vectorizer, metrics, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.sav"
    vectorizer_path = output_dir / "tfidfVectorizer.pickle"
    metadata_path = output_dir / "logreg_metadata.json"

    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    with open(vectorizer_path, "wb") as handle:
        pickle.dump(vectorizer, handle)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({**metadata, **metrics}, handle, indent=2)

    return model_path, vectorizer_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression sentiment model on YouTube comments.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to youtube_comments_cleaned.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store model.sav and tfidfVectorizer.pickle",
    )
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=75000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--ngram_range", default="1,2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--solver", default="saga")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--class_weight", default="none")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--test_data", default=None, help="Path to test.csv")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = (
        Path(args.data)
        if args.data
        else base_dir / "data" / "train.csv"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base_dir / "models" / "logreg"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Load train data
    train_df = load_dataset(
        data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        max_per_class=args.max_per_class,
        random_seed=args.random_seed,
    )

    # Load test data
    test_path = Path(args.test_data) if args.test_data else base_dir / "data" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    test_df = load_dataset(
        test_path,
        text_column=args.text_column,
        label_column=args.label_column,
        max_per_class=None,
        random_seed=args.random_seed,
    )

    model, vectorizer, metrics = train_model(train_df, test_df, args)

    metadata = {
        "train_dataset": str(data_path),
        "test_dataset": str(test_path),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "label_distribution": train_df["label"].value_counts().to_dict(),
        "max_per_class": args.max_per_class,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "vectorizer": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": args.ngram_range,
            "lowercase": not args.no_lowercase,
            "strip_accents": args.strip_accents,
        },
        "model": {
            "type": "LogisticRegression",
            "C": args.C,
            "max_iter": args.max_iter,
            "solver": args.solver,
            "class_weight": args.class_weight,
            "n_jobs": args.n_jobs,
        },
        "random_seed": args.random_seed,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }

    model_path, vectorizer_path, metadata_path = save_outputs(
        output_dir, model, vectorizer, metrics, metadata
    )

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
