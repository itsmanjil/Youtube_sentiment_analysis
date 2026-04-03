#!/usr/bin/env python3
"""
Benchmark the pinned live runtime stack on a held-out labeled split.

This script evaluates the same engines used by the application runtime, using
the currently pinned runtime artifact version from `backend/results/runtime/`.
It is intended to produce thesis-facing evidence for the deployed inference
path rather than for offline research-only scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from research.evaluation.calibration import compute_calibration_metrics, probs_to_matrix
from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, get_runtime_artifact_metadata


DEFAULT_MODEL_SPECS = (
    "tfidf,logreg,svm,ensemble:pso,ensemble:nsga2,meta_learner,fuzzy_ensemble"
)


def parse_model_specs(raw: str) -> List[Tuple[str, str | None]]:
    specs: List[Tuple[str, str | None]] = []
    for token in str(raw or "").split(","):
        item = token.strip().lower()
        if not item:
            continue
        if ":" in item:
            model_name, variant = item.split(":", 1)
            specs.append((model_name.strip(), variant.strip() or None))
        else:
            specs.append((item, None))
    return specs


def display_name(model_name: str, variant: str | None) -> str:
    if model_name == "ensemble" and variant:
        return f"ensemble_{variant}"
    return model_name


def load_dataset(csv_path: Path, text_column: str, label_column: str, max_samples: int | None) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={text_column: "string", label_column: "string"},
        keep_default_na=False,
    )
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Dataset must contain '{text_column}' and '{label_column}' columns."
        )

    frame = df[[text_column, label_column]].copy()
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame[label_column] = frame[label_column].fillna("").astype(str).map(normalize_label)
    frame = frame[frame[text_column].str.strip().astype(bool)]
    frame = frame[frame[label_column].str.strip().astype(bool)]
    if max_samples:
        frame = frame.iloc[: int(max_samples)].copy()
    if frame.empty:
        raise ValueError("No labeled rows available for runtime benchmarking.")
    return frame.reset_index(drop=True)


def iter_chunks(values: List[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def evaluate_runtime_model(
    model_name: str,
    variant: str | None,
    texts: List[str],
    labels: List[str],
    chunk_size: int,
    calibration_bins: int,
) -> Dict[str, object]:
    engine_kwargs: Dict[str, object] = {}
    if model_name == "ensemble":
        engine_kwargs["base_models"] = ["logreg", "svm", "tfidf"]
        if variant in {"pso", "nsga2"}:
            engine_kwargs["weights_optimization"] = variant
    elif model_name == "fuzzy_ensemble":
        engine_kwargs["base_models"] = ["logreg", "svm", "tfidf"]

    engine = get_sentiment_engine(model_name, **engine_kwargs)

    predictions: List[str] = []
    probs_list: List[Dict[str, float]] = []
    start = time.perf_counter()

    for chunk in iter_chunks(texts, chunk_size):
        if hasattr(engine, "batch_analyze"):
            results = engine.batch_analyze(chunk)
        else:
            results = [engine.analyze(text) for text in chunk]

        for result in results:
            coerced = coerce_sentiment_result(result, model_name)
            predictions.append(coerced.label)
            probs_list.append(coerced.probs)

    elapsed = time.perf_counter() - start
    prob_matrix = probs_to_matrix(probs_list, labels=SENTIMENT_LABELS)
    ece, brier = compute_calibration_metrics(
        labels,
        prob_matrix,
        labels=SENTIMENT_LABELS,
        n_bins=calibration_bins,
    )

    record = {
        "model": display_name(model_name, variant),
        "engine_type": model_name,
        "variant": variant,
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
        "ece": round(float(ece), 6),
        "brier": round(float(brier), 6),
        "n_samples": len(labels),
        "runtime_seconds": round(float(elapsed), 3),
        "ms_per_sample": round(float((elapsed / max(len(labels), 1)) * 1000.0), 4),
        "calibration_applied": bool(getattr(engine, "calibration_applied", False)),
        "temperature": getattr(engine, "temperature", None),
        "weights_source": getattr(engine, "weights_source", None),
        "nf_gate_active": bool(getattr(engine, "_nf_mfs", {})),
        "model_artifact": getattr(engine, "model_artifact", None),
    }
    return record


def build_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Live Runtime Benchmark\n",
        f"- Runtime artifact version: `{payload['runtime_artifacts']['version']}`",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Text column: `{payload['text_column']}`",
        f"- Samples: `{payload['n_samples']}`",
        "",
        "| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | Temp | Weights | NF Gate | ms/sample |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |",
    ]

    for result in payload["results"]:
        temperature = result["temperature"]
        lines.append(
            "| {model} | {accuracy:.4f} | {macro_f1:.4f} | {ece:.6f} | {brier:.6f} | {cal} | {temp} | {weights} | {nf_gate} | {ms:.4f} |".format(
                model=result["model"],
                accuracy=result["accuracy"],
                macro_f1=result["macro_f1"],
                ece=result["ece"],
                brier=result["brier"],
                cal="yes" if result["calibration_applied"] else "no",
                temp=f"{float(temperature):.4f}" if isinstance(temperature, (int, float)) and not math.isnan(float(temperature)) else "—",
                weights=result["weights_source"] or "—",
                nf_gate="yes" if result["nf_gate_active"] else "no",
                ms=result["ms_per_sample"],
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the pinned live runtime stack.")
    parser.add_argument("--data", default="data/test.csv", help="Held-out labeled CSV to evaluate.")
    parser.add_argument("--text_column", default="text", help="Text column to evaluate.")
    parser.add_argument("--label_column", default="label", help="Label column to evaluate.")
    parser.add_argument("--models", default=DEFAULT_MODEL_SPECS, help="Comma-separated model specs. Use ensemble:pso or ensemble:nsga2 for ensemble variants.")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Batch size for runtime inference.")
    parser.add_argument("--calibration_bins", type=int, default=15, help="ECE bin count.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--output_json", default=None, help="Optional JSON output path.")
    parser.add_argument("--output_md", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    runtime_metadata = get_runtime_artifact_metadata()
    version = runtime_metadata["version"]
    default_output_root = BASE_DIR / "results" / "runtime" / version
    default_output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = (BASE_DIR / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    frame = load_dataset(
        dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        max_samples=args.max_samples,
    )

    texts = frame[args.text_column].astype(str).tolist()
    labels = frame[args.label_column].astype(str).tolist()

    results = []
    failures = []
    for model_name, variant in parse_model_specs(args.models):
        try:
            result = evaluate_runtime_model(
                model_name=model_name,
                variant=variant,
                texts=texts,
                labels=labels,
                chunk_size=int(args.chunk_size),
                calibration_bins=int(args.calibration_bins),
            )
            print(
                f"[ok] {result['model']}: macro_f1={result['macro_f1']:.4f} "
                f"ece={result['ece']:.6f}"
            )
            results.append(result)
        except Exception as exc:
            failure = {
                "model": display_name(model_name, variant),
                "engine_type": model_name,
                "variant": variant,
                "error": str(exc),
            }
            print(f"[error] {failure['model']}: {failure['error']}")
            failures.append(failure)

    results.sort(key=lambda item: item["macro_f1"], reverse=True)

    payload = {
        "title": "Live Runtime Benchmark",
        "runtime_artifacts": runtime_metadata,
        "dataset_path": str(dataset_path),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "n_samples": len(labels),
        "results": results,
        "failures": failures,
    }

    output_json = Path(args.output_json) if args.output_json else default_output_root / "live_runtime_benchmark_full_test.json"
    output_md = Path(args.output_md) if args.output_md else default_output_root / "live_runtime_benchmark_full_test.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    output_md.write_text(build_markdown(payload))

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
