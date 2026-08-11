#!/usr/bin/env python3
"""
Compare offline probability-cube predictions against the live runtime engines.

The benchmark-level reconciliation proves aggregate metrics line up. This
script closes the stricter gap: for the same rows and same model artifacts, do
the live engines produce the same per-sample labels and probabilities as a
stored offline probability cube?
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.prob_cube_io import (
    load_probability_cube,
    prepare_scoring_frame,
)
from src.sentiment import coerce_sentiment_result, get_sentiment_engine
from src.utils import SENTIMENT_LABELS, get_runtime_artifact_metadata, normalize_probs


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BACKEND_ROOT / path


def _engine_kwargs(model_name: str, calibration_profile: str) -> Dict[str, object]:
    return {}


def _iter_chunks(values: Sequence[str], chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield start, list(values[start : start + chunk_size])


def _matrix_from_results(results, model_name: str, labels: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(results), len(labels)), dtype=np.float64)
    for row_index, raw_result in enumerate(results):
        result = coerce_sentiment_result(raw_result, model_name)
        probs = normalize_probs(result.probs)
        matrix[row_index] = [float(probs.get(label, 0.0)) for label in labels]
    return matrix


def _score_live_model(
    model_name: str,
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    calibration_profile: str,
    chunk_size: int,
) -> np.ndarray:
    engine = get_sentiment_engine(
        model_name,
        **_engine_kwargs(model_name, calibration_profile),
    )
    live_probs = np.zeros((len(texts), len(labels)), dtype=np.float64)

    for start, chunk in _iter_chunks(texts, chunk_size):
        if hasattr(engine, "batch_analyze"):
            results = engine.batch_analyze(chunk)
        else:
            results = [engine.analyze(text) for text in chunk]
        if len(results) != len(chunk):
            raise RuntimeError(
                f"Model '{model_name}' returned {len(results)} results for {len(chunk)} inputs."
            )
        live_probs[start : start + len(chunk)] = _matrix_from_results(
            results,
            model_name,
            labels,
        )

    return live_probs


def _load_texts_for_cube(bundle, metadata: Dict[str, object], sample_seed: int):
    if bundle.texts is not None:
        texts_by_model = {model: list(bundle.texts) for model in bundle.model_names}
        return texts_by_model, {"source": "cube_texts", "source_csv": None}

    source_csv = metadata.get("source_csv")
    if not source_csv:
        raise ValueError(
            "Probability cube does not include texts and metadata has no source_csv."
        )

    source_path = _resolve_backend_path(str(source_csv))
    text_columns_by_model = metadata.get("text_columns_by_model") or {}
    text_column = str(metadata.get("text_column") or "auto")
    label_column = str(metadata.get("label_column") or "label")

    frame, _, resolved_by_model = prepare_scoring_frame(
        source_path,
        model_names=bundle.model_names,
        text_column=text_column,
        label_column=label_column,
    )

    sample_size = int(metadata.get("sample_size") or len(bundle.y_true))
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)
    if len(frame) != len(bundle.y_true):
        raise ValueError(
            f"Reconstructed frame length {len(frame)} does not match cube length {len(bundle.y_true)}."
        )

    texts_by_model = {}
    for model_name in bundle.model_names:
        column = text_columns_by_model.get(model_name) or resolved_by_model[model_name]
        texts_by_model[model_name] = frame[column].astype(str).tolist()

    return texts_by_model, {
        "source": "reconstructed_source_csv",
        "source_csv": str(source_path),
        "sample_seed": sample_seed,
    }


def compare_model(
    *,
    model_name: str,
    offline_probs: np.ndarray,
    live_probs: np.ndarray,
    labels: Sequence[str],
    prob_tolerance: float,
) -> Dict[str, object]:
    offline_pred_idx = offline_probs.argmax(axis=1)
    live_pred_idx = live_probs.argmax(axis=1)
    offline_labels = [labels[index] for index in offline_pred_idx]
    live_labels = [labels[index] for index in live_pred_idx]
    label_matches = offline_pred_idx == live_pred_idx
    abs_diff = np.abs(offline_probs.astype(np.float64) - live_probs.astype(np.float64))
    mismatches = np.where(~label_matches)[0]

    return {
        "model": model_name,
        "n_samples": int(len(offline_labels)),
        "label_matches": int(label_matches.sum()),
        "label_mismatches": int((~label_matches).sum()),
        "label_match_rate": round(float(label_matches.mean()), 8),
        "max_abs_probability_delta": round(float(abs_diff.max()), 10),
        "mean_abs_probability_delta": round(float(abs_diff.mean()), 10),
        "probability_within_tolerance": bool(abs_diff.max() <= prob_tolerance),
        "first_mismatches": [
            {
                "index": int(index),
                "offline_label": offline_labels[index],
                "live_label": live_labels[index],
                "offline_probs": {
                    label: round(float(offline_probs[index, label_index]), 8)
                    for label_index, label in enumerate(labels)
                },
                "live_probs": {
                    label: round(float(live_probs[index, label_index]), 8)
                    for label_index, label in enumerate(labels)
                },
            }
            for index in mismatches[:10]
        ],
    }


def build_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Prediction-Level Offline vs Live Reconciliation\n",
        f"- Created at: `{payload['created_at']}`",
        f"- Runtime artifact version: `{payload['runtime_artifacts']['version']}`",
        f"- Offline probability cube: `{payload['offline_prob_cube']}`",
        f"- Text source: `{payload['text_source']['source']}`",
        f"- Samples: `{payload['n_samples']}`",
        f"- Probability tolerance: `{payload['probability_tolerance']}`",
        f"- Label-equivalence status: `{'PASS' if payload['label_equivalence_passed'] else 'FAIL'}`",
        f"- Strict probability-equivalence status: `{'PASS' if payload['probability_equivalence_passed'] else 'FAIL'}`",
        "",
        "| Model | Samples | Label Match Rate | Mismatches | Max Prob Delta | Mean Prob Delta | Prob Tol Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in payload["model_comparisons"]:
        lines.append(
            "| {model} | {n_samples} | {label_match_rate:.8f} | {label_mismatches} | "
            "{max_abs_probability_delta:.10f} | {mean_abs_probability_delta:.10f} | {tol} |".format(
                model=row["model"],
                n_samples=row["n_samples"],
                label_match_rate=row["label_match_rate"],
                label_mismatches=row["label_mismatches"],
                max_abs_probability_delta=row["max_abs_probability_delta"],
                mean_abs_probability_delta=row["mean_abs_probability_delta"],
                tol="yes" if row["probability_within_tolerance"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation\n",
            payload["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare offline probability-cube predictions with live runtime predictions."
    )
    parser.add_argument(
        "--prob_cube",
        default="results/prob_cubes/route_a_benchmark_cpu_test_logreg_svm.npz",
        help="Offline probability cube to compare.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model subset. Defaults to all cube models.",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--probability_tolerance", type=float, default=1e-6)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    cube_path = _resolve_backend_path(args.prob_cube)
    bundle = load_probability_cube(cube_path)
    metadata = bundle.metadata or {}
    labels = list(bundle.labels or SENTIMENT_LABELS)
    calibration_profile = str(metadata.get("calibration_profile") or "auto")

    requested_models = [
        item.strip().lower().replace("-", "_")
        for item in str(args.models or ",".join(bundle.model_names)).split(",")
        if item.strip()
    ]
    missing = [model for model in requested_models if model not in bundle.model_names]
    if missing:
        raise SystemExit(
            f"Requested models are not present in cube: {missing}. "
            f"Available: {bundle.model_names}"
        )

    texts_by_model, text_source = _load_texts_for_cube(
        bundle,
        metadata,
        sample_seed=args.sample_seed,
    )

    model_comparisons = []
    for model_name in requested_models:
        cube_index = bundle.model_names.index(model_name)
        live_probs = _score_live_model(
            model_name,
            texts_by_model[model_name],
            labels,
            calibration_profile=calibration_profile,
            chunk_size=args.chunk_size,
        )
        model_comparisons.append(
            compare_model(
                model_name=model_name,
                offline_probs=bundle.prob_cube[cube_index],
                live_probs=live_probs,
                labels=labels,
                prob_tolerance=args.probability_tolerance,
            )
        )

    label_equivalence_passed = all(row["label_mismatches"] == 0 for row in model_comparisons)
    probability_equivalence_passed = all(
        row["probability_within_tolerance"] for row in model_comparisons
    )
    interpretation = (
        "The live runtime reproduced every offline probability-cube label. "
        "Probability deltas are reported separately because calibration or "
        "environment-level floating-point differences can change confidence "
        "without changing the predicted class."
        if label_equivalence_passed
        else (
            "At least one model differs at the per-sample label level. Inspect "
            "the mismatch rows before using aggregate reconciliation as "
            "equivalence evidence."
        )
    )

    runtime_metadata = get_runtime_artifact_metadata()
    payload = {
        "title": "Prediction-Level Offline vs Live Reconciliation",
        "created_at": _utcnow(),
        "runtime_artifacts": runtime_metadata,
        "offline_prob_cube": str(cube_path),
        "offline_metadata": metadata,
        "text_source": text_source,
        "labels": labels,
        "n_samples": len(bundle.y_true),
        "probability_tolerance": float(args.probability_tolerance),
        "model_comparisons": model_comparisons,
        "label_equivalence_passed": label_equivalence_passed,
        "probability_equivalence_passed": probability_equivalence_passed,
        "passed": label_equivalence_passed,
        "interpretation": interpretation,
    }

    output_root = BACKEND_ROOT / "results" / "runtime" / runtime_metadata["version"]
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_root / "prediction_level_reconciliation.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else output_root / "prediction_level_reconciliation.md"
    )
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if not label_equivalence_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
