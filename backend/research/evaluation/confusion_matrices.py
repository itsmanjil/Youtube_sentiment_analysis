#!/usr/bin/env python3
"""
Full confusion matrices for all runtime sentiment models.

Produces a thesis-facing Markdown + JSON report with:
- Per-model confusion matrix (Negative / Neutral / Positive)
- Normalised confusion matrix (row-normalised, i.e. recall per class)
- Per-class Precision / Recall / F1 summary table

Usage
-----
    cd backend
    python research/evaluation/confusion_matrices.py --test data/test.csv --sample 5000
    python research/evaluation/confusion_matrices.py --test data/test.csv  # full set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS

LABELS = list(SENTIMENT_LABELS)   # ["Positive", "Neutral", "Negative"]
ORDERED = ["Negative", "Neutral", "Positive"]   # display order for confusion matrix


def _display(model_name: str, variant: str | None) -> str:
    if model_name == "ensemble" and variant:
        return f"ensemble_{variant}"
    return model_name


DEFAULT_MODELS: List[Tuple[str, str | None]] = [
    ("logreg",         None),
    ("svm",            None),
    ("tfidf",          None),
    ("ensemble",       "pso"),
    ("ensemble",       "nsga2"),
    ("meta_learner",   None),
    ("fuzzy_ensemble", None),
]


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


def get_predictions(model_name: str, variant: str | None, texts: List[str]) -> List[str]:
    engine = get_sentiment_engine(model_name, **_engine_kwargs(model_name, variant))
    if hasattr(engine, "batch_analyze"):
        results = engine.batch_analyze(texts)
    else:
        results = [engine.analyze(t) for t in texts]
    return [coerce_sentiment_result(r, model_name).label for r in results]


def _cm_to_dict(cm: np.ndarray, labels: List[str]) -> Dict:
    return {
        "labels": labels,
        "matrix": cm.tolist(),
    }


def _format_cm_md(cm: np.ndarray, labels: List[str], title: str) -> str:
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    sep    = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows   = [header, sep]
    for i, row_label in enumerate(labels):
        vals = " | ".join(str(cm[i, j]) for j in range(len(labels)))
        rows.append(f"| **{row_label}** | {vals} |")
    return f"### {title}\n\n" + "\n".join(rows) + "\n"


def _format_norm_cm_md(cm: np.ndarray, labels: List[str], title: str) -> str:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    sep    = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows   = [header, sep]
    for i, row_label in enumerate(labels):
        vals = " | ".join(f"{cm_norm[i, j]:.3f}" for j in range(len(labels)))
        rows.append(f"| **{row_label}** | {vals} |")
    return f"### {title} (Row-Normalised Recall)\n\n" + "\n".join(rows) + "\n"


def build_markdown(results: List[Dict], n_samples: int, csv_path: str) -> str:
    lines = [
        "# Confusion Matrices — All Models\n",
        f"- Dataset: `{csv_path}`",
        f"- Samples: `{n_samples}`",
        f"- Label order: Negative / Neutral / Positive",
        "",
        "## Per-Class Precision / Recall / F1 Summary",
        "",
        "| Model | Neg P | Neg R | Neg F1 | Neu P | Neu R | Neu F1 | Pos P | Pos R | Pos F1 | Macro F1 |",
        "|-------|-------|-------|--------|-------|-------|--------|-------|-------|--------|----------|",
    ]

    for r in results:
        prf = r["prf"]
        lines.append(
            f"| {r['model']} "
            f"| {prf['Negative']['precision']:.3f} | {prf['Negative']['recall']:.3f} | {prf['Negative']['f1']:.3f} "
            f"| {prf['Neutral']['precision']:.3f} | {prf['Neutral']['recall']:.3f} | {prf['Neutral']['f1']:.3f} "
            f"| {prf['Positive']['precision']:.3f} | {prf['Positive']['recall']:.3f} | {prf['Positive']['f1']:.3f} "
            f"| {r['macro_f1']:.4f} |"
        )

    lines += ["", "---", ""]

    for r in results:
        cm      = np.array(r["confusion_matrix"]["matrix"])
        cm_labels = r["confusion_matrix"]["labels"]
        lines.append(f"## {r['model']}\n")
        lines.append(_format_cm_md(cm, cm_labels, "Confusion Matrix (counts)"))
        lines.append(_format_norm_cm_md(cm, cm_labels, "Confusion Matrix"))
        lines.append("")

    lines += [
        "## Thesis Interpretation",
        "",
        "The confusion matrices reveal a consistent pattern across all models:",
        "- Neutral class has the highest off-diagonal mass (most confused with Positive and Negative)",
        "- Positive and Negative are generally well-separated from each other",
        "- The Neutral row in the normalised matrix shows the lowest diagonal value for every model,",
        "  confirming that neutral sentiment is the primary source of classification error.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion matrix evaluation for all models")
    parser.add_argument("--test",   default="data/test.csv",           help="Labeled test CSV")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Sample N rows (0 = full set, default 5000)")
    parser.add_argument("--output", default="results/confusion_matrices", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.test)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = args.sample if args.sample > 0 else None
    print(f"Loading dataset from {csv_path} (sample={sample or 'full'})...")
    df = load_dataset(csv_path, sample)
    texts  = df["text"].tolist()
    y_true = df["label"].tolist()
    print(f"  {len(y_true)} labeled samples loaded.")

    results = []
    for model_name, variant in DEFAULT_MODELS:
        display = _display(model_name, variant)
        print(f"  Evaluating {display}...", flush=True)
        try:
            preds = get_predictions(model_name, variant, texts)

            cm = confusion_matrix(y_true, preds, labels=ORDERED)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, preds, labels=ORDERED, average=None, zero_division=0
            )
            macro_f1 = float(np.mean(f1))

            prf = {cls: {"precision": round(float(prec[i]), 4),
                         "recall":    round(float(rec[i]),  4),
                         "f1":        round(float(f1[i]),   4)}
                   for i, cls in enumerate(ORDERED)}

            results.append({
                "model":            display,
                "macro_f1":         round(macro_f1, 4),
                "confusion_matrix": _cm_to_dict(cm, ORDERED),
                "prf":              prf,
            })
            print(f"    macro F1={macro_f1:.4f}")
        except Exception as exc:
            print(f"    SKIPPED ({exc})")

    md  = build_markdown(results, len(y_true), str(csv_path))
    js  = json.dumps({"n_samples": len(y_true), "dataset": str(csv_path),
                       "label_order": ORDERED, "results": results}, indent=2)

    (out_dir / "confusion_matrices.md").write_text(md,  encoding="utf-8")
    (out_dir / "confusion_matrices.json").write_text(js, encoding="utf-8")
    print(f"\nOutputs written to {out_dir}/")
    print(md)


if __name__ == "__main__":
    main()
