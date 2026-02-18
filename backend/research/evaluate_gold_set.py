#!/usr/bin/env python3
"""
Evaluate models on a human-labeled gold set and report annotation agreement.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine
from src.utils import SENTIMENT_LABELS
from research.evaluation.calibration import compute_calibration_metrics, probs_to_matrix


def _parse_label(value: object, *, allow_blank: bool) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        if allow_blank:
            return None
        raise ValueError("empty label")

    normalized = text.lower()
    mapping = {
        "positive": "Positive",
        "pos": "Positive",
        "2": "Positive",
        "label_2": "Positive",
        "neutral": "Neutral",
        "neu": "Neutral",
        "1": "Neutral",
        "label_1": "Neutral",
        "negative": "Negative",
        "neg": "Negative",
        "0": "Negative",
        "label_0": "Negative",
    }
    if normalized not in mapping:
        raise ValueError(text)
    return mapping[normalized]


def _normalize_models(raw: str) -> List[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _load_ensemble_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / value
        if not path.exists():
            alt = BASE_DIR / value
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Could not resolve ensemble weights file: {value}")
        payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
        payload = payload["weights"]
    if not isinstance(payload, dict):
        raise ValueError("Ensemble weights must be a JSON object.")

    weights: Dict[str, float] = {}
    for key, val in payload.items():
        weights[str(key).strip().lower()] = float(val)
    return weights


def _load_gold_set(
    csv_path: Path,
    text_column: str,
    label_column: str,
    source_label_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={text_column: "string", label_column: "string"},
        keep_default_na=False,
    )
    missing = [col for col in [text_column, label_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Gold set is missing required columns: {missing}")

    data = df.copy()
    data[text_column] = data[text_column].astype(str).str.strip()
    data = data[data[text_column].astype(bool)].copy()

    labels: List[str] = []
    bad_rows: List[str] = []
    for idx, value in enumerate(data[label_column].tolist(), start=2):
        try:
            parsed = _parse_label(value, allow_blank=False)
            labels.append(parsed or "")
        except ValueError:
            bad_rows.append(f"{idx}:{value}")
    if bad_rows:
        preview = ", ".join(bad_rows[:5])
        raise ValueError(
            f"Invalid labels in column '{label_column}'. "
            f"Examples (row:value): {preview}"
        )
    data[label_column] = labels

    if source_label_column in data.columns:
        source_labels: List[Optional[str]] = []
        source_bad_rows: List[str] = []
        for idx, value in enumerate(data[source_label_column].tolist(), start=2):
            try:
                source_labels.append(_parse_label(value, allow_blank=True))
            except ValueError:
                source_bad_rows.append(f"{idx}:{value}")
        if source_bad_rows:
            preview = ", ".join(source_bad_rows[:5])
            raise ValueError(
                f"Invalid labels in source column '{source_label_column}'. "
                f"Examples (row:value): {preview}"
            )
        data[source_label_column] = source_labels
    else:
        data[source_label_column] = None

    return data.reset_index(drop=True)


def _evaluate_model(
    model_name: str,
    texts: List[str],
    y_true: List[str],
    calibration_bins: int,
    preprocess: bool,
    ensemble_models: List[str],
    ensemble_weights: Optional[Dict[str, float]],
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if preprocess and model_name in {
        "tfidf",
        "logreg",
        "svm",
        "ensemble",
        "meta_learner",
        "fuzzy_ensemble",
    }:
        kwargs["preprocess"] = True
    if model_name == "ensemble":
        kwargs["base_models"] = ensemble_models
        kwargs["weights"] = ensemble_weights

    engine = get_sentiment_engine(model_name, **kwargs)
    raw_results = engine.batch_analyze(texts)
    parsed = [coerce_sentiment_result(item, model_name) for item in raw_results]
    y_pred = [item.label for item in parsed]
    probs = [item.probs for item in parsed]
    prob_matrix = probs_to_matrix(probs, labels=SENTIMENT_LABELS)
    ece, brier = compute_calibration_metrics(
        y_true,
        prob_matrix,
        labels=SENTIMENT_LABELS,
        n_bins=calibration_bins,
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=SENTIMENT_LABELS
        ).tolist(),
        "calibration": {
            "ece": round(float(ece), 6),
            "brier": round(float(brier), 6),
            "bins": int(calibration_bins),
        },
    }


def _agreement_summary(
    data: pd.DataFrame,
    label_column: str,
    source_label_column: str,
) -> Optional[Dict[str, object]]:
    compared = data[data[source_label_column].notna()].copy()
    if compared.empty:
        return None

    y_human = compared[label_column].astype(str).tolist()
    y_source = compared[source_label_column].astype(str).tolist()
    matrix = confusion_matrix(y_human, y_source, labels=SENTIMENT_LABELS).tolist()

    return {
        "label_column": label_column,
        "source_label_column": source_label_column,
        "n_compared": int(len(compared)),
        "agreement_rate": round(float((compared[label_column] == compared[source_label_column]).mean()), 6),
        "cohen_kappa": round(float(cohen_kappa_score(y_human, y_source)), 6),
        "confusion_matrix": matrix,
    }


def _to_markdown(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Gold Set Evaluation")
    lines.append("")
    lines.append(f"- Data: `{result['data']}`")
    lines.append(f"- Samples: {result['n_samples']}")
    lines.append(f"- Generated at (UTC): {result['generated_at_utc']}")
    lines.append("")

    agreement = result.get("human_vs_source_agreement")
    if agreement:
        lines.append("## Human vs Source Agreement")
        lines.append("")
        lines.append(f"- Compared rows: {agreement['n_compared']}")
        lines.append(f"- Agreement rate: {agreement['agreement_rate']:.4f}")
        lines.append(f"- Cohen's kappa: {agreement['cohen_kappa']:.4f}")
        lines.append("")

    lines.append("## Model Metrics")
    lines.append("")
    lines.append("| Model | Accuracy | Macro-F1 | ECE | Brier |")
    lines.append("| --- | --- | --- | --- | --- |")
    for model, metrics in result["metrics"].items():
        cal = metrics.get("calibration", {})
        lines.append(
            "| {m} | {acc:.4f} | {f1:.4f} | {ece:.6f} | {brier:.6f} |".format(
                m=model,
                acc=float(metrics.get("accuracy", 0.0)),
                f1=float(metrics.get("macro_f1", 0.0)),
                ece=float(cal.get("ece", 0.0)),
                brier=float(cal.get("brier", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sentiment models on a human-labeled gold set."
    )
    parser.add_argument("--data", required=True, help="Path to gold-set CSV.")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--source_label_column", default="source_label")
    parser.add_argument(
        "--models",
        default="tfidf,logreg,svm,ensemble,meta_learner",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--ensemble-models",
        default="logreg,svm,tfidf",
        help="Base models for ensemble engine.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default=None,
        help="Optional JSON dict or path to JSON weights file.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        help="Bins for Expected Calibration Error.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable classical preprocessing for classical/ensemble models.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: backend/results/gold_set_evaluation.json).",
    )
    parser.add_argument(
        "--summary_md",
        default=None,
        help="Optional markdown summary path. Defaults to JSON path with .md extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Gold set CSV not found: {data_path}")

    gold = _load_gold_set(
        csv_path=data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        source_label_column=args.source_label_column,
    )

    texts = gold[args.text_column].astype(str).tolist()
    y_true = gold[args.label_column].astype(str).tolist()
    model_list = _normalize_models(args.models)
    ensemble_models = _normalize_models(args.ensemble_models)
    ensemble_weights = _load_ensemble_weights(args.ensemble_weights)

    metrics: Dict[str, Dict[str, object]] = {}
    for model in model_list:
        metrics[model] = _evaluate_model(
            model_name=model,
            texts=texts,
            y_true=y_true,
            calibration_bins=int(args.calibration_bins),
            preprocess=bool(args.preprocess),
            ensemble_models=ensemble_models,
            ensemble_weights=ensemble_weights,
        )

    result: Dict[str, object] = {
        "data": str(data_path),
        "n_samples": int(len(gold)),
        "label_distribution": gold[args.label_column].value_counts().to_dict(),
        "models": model_list,
        "ensemble_models": ensemble_models,
        "ensemble_weights": ensemble_weights,
        "preprocess": bool(args.preprocess),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "human_vs_source_agreement": _agreement_summary(
            data=gold,
            label_column=args.label_column,
            source_label_column=args.source_label_column,
        ),
    }

    out_path = (
        Path(args.output)
        if args.output
        else BASE_DIR / "results" / "gold_set_evaluation.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_path = Path(args.summary_md) if args.summary_md else out_path.with_suffix(".md")
    md_path.write_text(_to_markdown(result), encoding="utf-8")

    print(f"Wrote JSON: {out_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
