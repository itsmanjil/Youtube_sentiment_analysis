#!/usr/bin/env python3
"""
ROC-AUC (One-vs-Rest) evaluation for all runtime sentiment models.

Produces thesis-facing ROC-AUC table with macro-average and per-class
AUC scores for every model in the standard benchmark suite.

Usage
-----
    cd backend
    python research/evaluation/roc_auc.py --test data/test.csv --sample 5000
    python research/evaluation/roc_auc.py --test data/test.csv          # full set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

BASE_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from research.evaluation.calibration import probs_to_matrix
from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS

LABELS = list(SENTIMENT_LABELS)  # ["Positive", "Neutral", "Negative"]

DEFAULT_MODELS: List[Tuple[str, str | None]] = [
    ("logreg",        None),
    ("svm",           None),
    ("tfidf",         None),
    ("ensemble",      "pso"),
    ("ensemble",      "nsga2"),
    ("meta_learner",  None),
    ("fuzzy_ensemble", None),
]


def _display(model_name: str, variant: str | None) -> str:
    if model_name == "ensemble" and variant:
        return f"ensemble_{variant}"
    return model_name


def load_dataset(csv_path: Path, sample: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"text": "string", "label": "string"},
                     keep_default_na=False)
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).map(normalize_label)
    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["label"].str.strip().astype(bool)]
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    return df.reset_index(drop=True)


def _engine_kwargs(model_name: str, variant: str | None) -> dict:
    kwargs: dict = {}
    if model_name == "ensemble":
        kwargs["base_models"] = ["logreg", "svm", "tfidf"]
        if variant in {"pso", "nsga2"}:
            kwargs["weights_optimization"] = variant
    elif model_name == "fuzzy_ensemble":
        kwargs["base_models"] = ["logreg", "svm", "tfidf"]
    return kwargs


def evaluate_roc(
    model_name: str,
    variant: str | None,
    texts: List[str],
    y_true: List[str],
) -> Dict[str, float]:
    engine = get_sentiment_engine(model_name, **_engine_kwargs(model_name, variant))

    probs_list = []
    if hasattr(engine, "batch_analyze"):
        results = engine.batch_analyze(texts)
    else:
        results = [engine.analyze(t) for t in texts]

    for result in results:
        coerced = coerce_sentiment_result(result, model_name)
        probs_list.append(coerced.probs)

    prob_matrix = probs_to_matrix(probs_list, labels=LABELS)

    # Binarize true labels for OvR ROC-AUC
    y_bin = label_binarize(y_true, classes=LABELS)

    per_class: Dict[str, float] = {}
    for i, cls in enumerate(LABELS):
        try:
            per_class[cls] = float(roc_auc_score(y_bin[:, i], prob_matrix[:, i]))
        except ValueError:
            per_class[cls] = float("nan")

    try:
        macro_auc = float(roc_auc_score(y_bin, prob_matrix, average="macro",
                                         multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")

    try:
        weighted_auc = float(roc_auc_score(y_bin, prob_matrix, average="weighted",
                                            multi_class="ovr"))
    except ValueError:
        weighted_auc = float("nan")

    return {
        "model":    _display(model_name, variant),
        "macro":    round(macro_auc, 4),
        "weighted": round(weighted_auc, 4),
        **{f"auc_{cls.lower()}": round(v, 4) for cls, v in per_class.items()},
    }


def build_markdown(rows: List[Dict], n_samples: int, csv_path: str) -> str:
    lines = [
        "# ROC-AUC Evaluation (One-vs-Rest)\n",
        f"- Dataset: `{csv_path}`",
        f"- Samples: `{n_samples}`",
        "- Method: One-vs-Rest (OvR) per class; macro-average across classes",
        "",
        "## Macro and Weighted AUC",
        "",
        "| Model | Macro AUC | Weighted AUC |",
        "|-------|-----------|--------------|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['macro']:.4f} | {r['weighted']:.4f} |")

    lines += [
        "",
        "## Per-Class AUC (OvR)",
        "",
        "| Model | Positive AUC | Neutral AUC | Negative AUC |",
        "|-------|-------------|------------|--------------|",
    ]
    for r in rows:
        p = r.get("auc_positive", float("nan"))
        n = r.get("auc_neutral",  float("nan"))
        g = r.get("auc_negative", float("nan"))
        lines.append(f"| {r['model']} | {p:.4f} | {n:.4f} | {g:.4f} |")

    lines += [
        "",
        "## Thesis Interpretation",
        "",
        "ROC-AUC is threshold-independent and measures how well the model's",
        "probability scores separate each sentiment class from the rest.",
        "A macro AUC of 1.0 is perfect; 0.5 is no better than random.",
        "",
        "Neutral class typically has the lowest per-class AUC, consistent",
        "with its lower F1 scores observed across all models — reflecting",
        "the inherent ambiguity of neutral comments.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="ROC-AUC evaluation for all models")
    parser.add_argument("--test",   default="data/test.csv",    help="Labeled test CSV")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Sample N rows (0 = full set, default 5000)")
    parser.add_argument("--output", default="results/roc_auc",  help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.test)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = args.sample if args.sample > 0 else None
    print(f"Loading dataset from {csv_path} (sample={sample or 'full'})...")
    df = load_dataset(csv_path, sample)
    texts  = df["text"].tolist()
    labels = df["label"].tolist()
    print(f"  {len(labels)} labeled samples loaded.")

    rows = []
    for model_name, variant in DEFAULT_MODELS:
        display = _display(model_name, variant)
        print(f"  Evaluating {display}...", flush=True)
        try:
            row = evaluate_roc(model_name, variant, texts, labels)
            rows.append(row)
            print(f"    macro AUC={row['macro']:.4f}  weighted={row['weighted']:.4f}")
        except Exception as exc:
            print(f"    SKIPPED ({exc})")

    # Sort by macro AUC descending
    rows.sort(key=lambda r: r["macro"], reverse=True)

    md  = build_markdown(rows, len(labels), str(csv_path))
    js  = json.dumps({"n_samples": len(labels), "dataset": str(csv_path),
                       "labels": LABELS, "results": rows}, indent=2)

    (out_dir / "roc_auc.md").write_text(md,  encoding="utf-8")
    (out_dir / "roc_auc.json").write_text(js, encoding="utf-8")
    print(f"\nOutputs written to {out_dir}/")
    print(md)


if __name__ == "__main__":
    main()
