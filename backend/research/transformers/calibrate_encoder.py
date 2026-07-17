#!/usr/bin/env python3
"""
Fit a post-hoc temperature scaling artifact for a trained encoder checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.evaluation.calibration import compute_calibration_metrics
from research.transformers.model_registry import get_encoder_spec, list_encoder_presets
from research.transformers.train_encoder import (
    LABELS as TRAINING_LABELS,
    load_split_frame,
    load_split_metadata,
    summarize_split_provenance,
)
from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import (
    apply_temperature_to_logits,
    fit_temperature_from_logits,
    logits_to_probs,
    save_temperature_artifact,
)
from src.utils.config import Config

LABELS = list(TRAINING_LABELS)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_float_dict(metrics: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _metric_summary(y_true: List[str], logits: np.ndarray, temperature: float = 1.0) -> Dict[str, float]:
    probs = logits_to_probs(logits) if temperature == 1.0 else apply_temperature_to_logits(logits, temperature)
    preds = [LABELS[index] for index in probs.argmax(axis=1)]
    accuracy = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    ece, brier = compute_calibration_metrics(y_true, probs, labels=LABELS)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "ece": float(ece),
        "brier": float(brier),
    }


def _collect_logits(engine, texts: List[str], batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        results = engine.batch_analyze(batch_texts)
        for result in results:
            coerced = coerce_sentiment_result(result, getattr(engine, "engine_name", "transformer"))
            raw = coerced.raw or {}
            logits = raw.get("logits")
            if logits is None:
                raise RuntimeError("Transformer runtime did not return raw logits for calibration.")
            rows.append([float(value) for value in logits])
    return np.asarray(rows, dtype=float)


def resolve_model_label_order(engine) -> List[str]:
    config = getattr(getattr(engine, "model", None), "config", None)
    raw_id2label = getattr(config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        ordered = []
        for index in sorted(raw_id2label):
            label = normalize_label(raw_id2label[index])
            if label not in ordered:
                ordered.append(label)
        if ordered and set(ordered) == set(TRAINING_LABELS):
            return ordered
    return list(TRAINING_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a trained encoder with temperature scaling.")
    parser.add_argument(
        "--model_preset",
        default="modernbert",
        choices=list_encoder_presets(),
        help="Named encoder preset to calibrate.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Optional explicit model path or HF model name override.",
    )
    parser.add_argument(
        "--val_csv",
        default=str(Config.DATA_DIR / "val.csv"),
        help="Validation CSV used to fit the temperature.",
    )
    parser.add_argument(
        "--test_csv",
        default=str(Config.DATA_DIR / "test.csv"),
        help="Test CSV used for reporting before/after metrics.",
    )
    parser.add_argument(
        "--text_column",
        default="auto",
        help="Text column to use. Defaults to the canonical split column.",
    )
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--split_metadata", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--artifact_output",
        default=None,
        help="Calibration artifact path. Defaults to backend/models/transformers/<preset>/temperature_scaling.json.",
    )
    parser.add_argument(
        "--results_output",
        default=None,
        help="Optional JSON report path. Defaults to backend/results/<preset>_temperature_scaling.json.",
    )
    args = parser.parse_args()

    spec = get_encoder_spec(args.model_preset)
    model_name_or_path = args.model_name_or_path or str(
        Config.get_model_path(spec.key) or (Config.MODELS_DIR / "transformers" / spec.key)
    )

    artifact_output = (
        Path(args.artifact_output)
        if args.artifact_output
        else (Config.get_model_path(spec.key, "calibration") or (Config.MODELS_DIR / "transformers" / spec.key / "temperature_scaling.json"))
    )
    results_output = (
        Path(args.results_output)
        if args.results_output
        else (BACKEND_ROOT / "results" / f"{spec.key}_temperature_scaling.json")
    )
    results_output.parent.mkdir(parents=True, exist_ok=True)

    val_df, val_text_column = load_split_frame(
        Path(args.val_csv),
        text_column=args.text_column,
        label_column=args.label_column,
    )
    test_df, test_text_column = load_split_frame(
        Path(args.test_csv),
        text_column=args.text_column,
        label_column=args.label_column,
    )

    split_metadata, split_metadata_path = load_split_metadata(
        Path(args.val_csv),
        explicit_path=Path(args.split_metadata) if args.split_metadata else None,
    )
    split_id = summarize_split_provenance(split_metadata)

    engine = get_sentiment_engine(
        spec.key,
        model_name_or_path=model_name_or_path,
        calibration_profile="off",
    )
    model_labels = resolve_model_label_order(engine)
    if model_labels != LABELS:
        raise RuntimeError(
            f"Unexpected label order {model_labels}; expected {LABELS}. "
            "Re-train the checkpoint or update calibration label handling."
        )

    y_val = val_df["label"].tolist()
    y_test = test_df["label"].tolist()
    y_val_ids = [LABEL_TO_ID[label] for label in y_val]

    val_logits = _collect_logits(engine, val_df[val_text_column].tolist(), args.batch_size)
    test_logits = _collect_logits(engine, test_df[test_text_column].tolist(), args.batch_size)

    temperature = fit_temperature_from_logits(val_logits, y_val_ids)

    val_before = _metric_summary(y_val, val_logits, temperature=1.0)
    val_after = _metric_summary(y_val, val_logits, temperature=temperature)
    test_before = _metric_summary(y_test, test_logits, temperature=1.0)
    test_after = _metric_summary(y_test, test_logits, temperature=temperature)

    artifact_payload = {
        "created_at": _utcnow(),
        "method": "temperature_scaling",
        "temperature": float(temperature),
        "model": {
            "preset": spec.key,
            "hf_name": spec.hf_name,
            "model_name_or_path": model_name_or_path,
            "artifact_dir": str(Path(model_name_or_path)) if Path(model_name_or_path).exists() else None,
            "label_order": model_labels,
        },
        "dataset": {
            "val_csv": str(args.val_csv),
            "test_csv": str(args.test_csv),
            "text_columns": {
                "val": val_text_column,
                "test": test_text_column,
            },
            "label_column": args.label_column,
            "split_metadata_path": str(split_metadata_path) if split_metadata_path else None,
            "split_id": split_id,
        },
        "metrics": {
            "val_before": _to_float_dict(val_before),
            "val_after": _to_float_dict(val_after),
            "test_before": _to_float_dict(test_before),
            "test_after": _to_float_dict(test_after),
        },
    }

    artifact_path = save_temperature_artifact(artifact_output, artifact_payload)
    results_output.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    print(f"Saved calibration artifact to: {artifact_path}")
    print(f"Saved calibration report to: {results_output}")
    print(
        "Calibration summary: "
        f"T={temperature:.4f}, "
        f"test_ece_before={test_before['ece']:.4f}, "
        f"test_ece_after={test_after['ece']:.4f}"
    )


if __name__ == "__main__":
    main()
