import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].apply(normalize_label)
    return df


def precompute_model_probs(models, texts):
    model_probs = {}
    for model in models:
        engine = get_sentiment_engine(model)
        results = engine.batch_analyze(texts)
        model_probs[model] = [
            normalize_probs(coerce_sentiment_result(result, model).probs)
            for result in results
        ]
    return model_probs


def evaluate_weights(weights, models, labels, model_probs):
    normalized = normalize_weights(weights, models)
    predictions = []
    for idx in range(len(labels)):
        combined = {label: 0.0 for label in SENTIMENT_LABELS}
        for model_name, weight in normalized.items():
            probs = model_probs[model_name][idx]
            for label in SENTIMENT_LABELS:
                combined[label] += weight * probs.get(label, 0.0)
        combined = normalize_probs(combined)
        predictions.append(max(combined, key=combined.get))
    return f1_score(labels, predictions, average="macro")


def normalize_weights(weights, models):
    weights = {model: max(0.0, float(weights.get(model, 0.0))) for model in models}
    total = sum(weights.values())
    if total <= 0:
        return {model: 1.0 / len(models) for model in models}
    return {model: value / total for model, value in weights.items()}


def pso_optimize(models, labels, model_probs, n_particles=20, n_iters=30, seed=42):
    rng = random.Random(seed)
    dim = len(models)

    def random_position():
        return [rng.random() for _ in range(dim)]

    positions = [random_position() for _ in range(n_particles)]
    velocities = [[rng.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(n_particles)]
    personal_best = list(positions)
    personal_scores = [float("-inf")] * n_particles
    global_best = positions[0]
    global_score = float("-inf")

    for _ in range(n_iters):
        for idx, position in enumerate(positions):
            weights = {models[i]: position[i] for i in range(dim)}
            score = evaluate_weights(weights, models, labels, model_probs)
            if score > personal_scores[idx]:
                personal_scores[idx] = score
                personal_best[idx] = list(position)
            if score > global_score:
                global_score = score
                global_best = list(position)

        for idx in range(n_particles):
            for i in range(dim):
                inertia = 0.7 * velocities[idx][i]
                cognitive = 1.4 * rng.random() * (personal_best[idx][i] - positions[idx][i])
                social = 1.4 * rng.random() * (global_best[i] - positions[idx][i])
                velocities[idx][i] = inertia + cognitive + social
                positions[idx][i] = max(0.0, positions[idx][i] + velocities[idx][i])

    best_weights = normalize_weights(
        {models[i]: global_best[i] for i in range(dim)},
        models,
    )
    return best_weights, global_score


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ensemble weights with PSO on a labeled dataset."
    )
    parser.add_argument("--data", required=True, help="Path to labeled CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="logreg,svm,tfidf",
        help="Comma-separated model list.",
    )
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    _, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df["label"],
    )

    models = [
        name.strip().lower() for name in args.models.split(",") if name.strip()
    ]
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    model_probs = precompute_model_probs(models, texts)

    weights, score = pso_optimize(
        models,
        labels,
        model_probs,
        n_particles=args.particles,
        n_iters=args.iterations,
        seed=args.random_seed,
    )

    result = {"weights": weights, "macro_f1": round(score, 4)}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
