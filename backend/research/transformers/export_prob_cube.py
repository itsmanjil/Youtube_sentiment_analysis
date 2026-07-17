#!/usr/bin/env python3
"""
Export calibrated model probability cubes for CI ensemble experiments.

This bridges Route A encoder training with the adaptive CI layer: once a set of
models has been trained and optionally calibrated, this script scores a split
once, stores the resulting probability cube, and lets downstream NSGA-II /
gating experiments reuse those scores without repeated rescoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.model_registry import normalize_encoder_key
from research.transformers.prob_cube_io import save_probability_cube
from research.transformers.train_encoder import (
    load_split_metadata,
    resolve_text_column,
    summarize_split_provenance,
)
from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import Config

LABELS = list(SENTIMENT_LABELS)
CLASSICAL_TEXT_COLUMN_PRIORITY = ["text_classical", "text", "text_raw", "text_transformer"]
TRANSFORMER_TEXT_COLUMN_PRIORITY = ["text_transformer", "text", "text_raw", "text_classical"]
TRANSFORMER_MODELS = {
    "bert",
    "transformer",
    "roberta",
    "modernbert",
    "deberta_v3",
    "xlm_v",
    "mdeberta_v3",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_model_names(raw_value: str) -> List[str]:
    if not raw_value:
        raise ValueError("At least one model must be provided.")

    models = []
    for item in raw_value.split(","):
        name = item.strip()
        if not name:
            continue
        normalized = normalize_encoder_key(name)
        models.append(normalized)

    if not models:
        raise ValueError("At least one model must be provided.")
    return models


def _engine_kwargs(model_name: str, calibration_profile: str) -> dict:
    if model_name in TRANSFORMER_MODELS:
        return {"calibration_profile": calibration_profile}
    return {}


def resolve_text_column_for_model(
    columns: Sequence[str],
    model_name: str,
    *,
    preferred: str = "auto",
) -> str:
    column_names = set(columns)
    if preferred and preferred != "auto":
        if preferred not in column_names:
            raise ValueError(
                f"Requested text column '{preferred}' not found. "
                f"Available columns: {sorted(column_names)}"
            )
        return preferred

    priority = (
        TRANSFORMER_TEXT_COLUMN_PRIORITY
        if model_name in TRANSFORMER_MODELS
        else CLASSICAL_TEXT_COLUMN_PRIORITY
    )
    for candidate in priority:
        if candidate in column_names:
            return candidate

    raise ValueError(
        f"No supported text column found for model '{model_name}'. "
        f"Tried: {priority}. Available columns: {sorted(column_names)}"
    )


def prepare_scoring_frame(
    csv_path: Path,
    *,
    model_names: Sequence[str],
    text_column: str = "auto",
    label_column: str = "label",
) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {csv_path}. "
            f"Available columns: {sorted(df.columns)}"
        )

    canonical_text_column = resolve_text_column(df.columns, preferred=text_column)
    model_text_columns = {
        model_name: resolve_text_column_for_model(
            df.columns,
            model_name,
            preferred=text_column,
        )
        for model_name in model_names
    }

    required_columns = set(model_text_columns.values()) | {label_column}
    if canonical_text_column in df.columns:
        required_columns.add(canonical_text_column)

    df = df.dropna(subset=sorted(required_columns)).copy()
    for column in required_columns:
        if column != label_column:
            df[column] = df[column].astype(str).str.strip()
            df = df[df[column].astype(bool)]

    df["label"] = df[label_column].apply(normalize_label)
    df = df[df["label"].isin(LABELS)]
    return df.reset_index(drop=True), canonical_text_column, model_text_columns


def _score_models(
    model_names: Sequence[str],
    model_texts: Mapping[str, Sequence[str]],
    *,
    batch_size: int,
    calibration_profile: str,
) -> Tuple[np.ndarray, np.ndarray | None, List[str]]:
    sample_count = len(model_texts[model_names[0]]) if model_names else 0
    prob_cube = np.zeros((len(model_names), sample_count, len(LABELS)), dtype=np.float32)
    logits_cube = np.full_like(prob_cube, np.nan, dtype=np.float32)
    logit_models: List[str] = []

    for model_index, model_name in enumerate(model_names):
        print(f"Scoring {model_name} on {sample_count:,} samples …", flush=True)
        engine = get_sentiment_engine(
            model_name,
            **_engine_kwargs(model_name, calibration_profile),
        )

        saw_logits = False
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_texts = list(model_texts[model_name][start:end])
            results = engine.batch_analyze(batch_texts)
            if len(results) != len(batch_texts):
                raise RuntimeError(
                    f"Model '{model_name}' returned {len(results)} results for {len(batch_texts)} inputs."
                )

            for offset, raw_result in enumerate(results):
                sample_index = start + offset
                result = coerce_sentiment_result(raw_result, model_name)
                probs = normalize_probs(result.probs)
                prob_cube[model_index, sample_index] = np.asarray(
                    [float(probs.get(label, 0.0)) for label in LABELS],
                    dtype=np.float32,
                )

                logits = (result.raw or {}).get("logits") if isinstance(result.raw, dict) else None
                if logits is not None and len(logits) == len(LABELS):
                    logits_cube[model_index, sample_index] = np.asarray(logits, dtype=np.float32)
                    saw_logits = True

        if saw_logits:
            logit_models.append(model_name)

    return prob_cube, (logits_cube if logit_models else None), logit_models


def _default_output_path(data_csv: Path, model_names: Sequence[str]) -> Path:
    cube_dir = Config.BACKEND_DIR / "results" / "prob_cubes"
    model_slug = "-".join(model_names)
    return cube_dir / f"{data_csv.stem}_{model_slug}_prob_cube.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a model probability cube for CI experiments.")
    parser.add_argument(
        "--data_csv",
        default=str(Config.DATA_DIR / "test.csv"),
        help="CSV split to score.",
    )
    parser.add_argument(
        "--models",
        default="modernbert,deberta_v3",
        help="Comma-separated model names or encoder presets.",
    )
    parser.add_argument(
        "--text_column",
        default="auto",
        help="Text column to score. Defaults to model-appropriate columns when set to auto.",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        help="Ground-truth label column name.",
    )
    parser.add_argument(
        "--sample_id_column",
        default=None,
        help="Optional column to persist as sample ids.",
    )
    parser.add_argument(
        "--split_metadata",
        default=None,
        help="Optional explicit path to split_metadata.json.",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--calibration_profile",
        default="auto",
        help="Transformer calibration mode: auto, off, raw, none.",
    )
    parser.add_argument(
        "--include_texts",
        action="store_true",
        help="Store input texts in the exported artifact.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Defaults to backend/results/prob_cubes/<split>_<models>_prob_cube.npz.",
    )
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    model_names = parse_model_names(args.models)
    df, resolved_text_column, model_text_columns = prepare_scoring_frame(
        data_csv,
        model_names=model_names,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    split_metadata, split_metadata_path = load_split_metadata(
        data_csv,
        explicit_path=Path(args.split_metadata) if args.split_metadata else None,
    )

    texts = df[resolved_text_column].astype(str).tolist()
    model_texts = {
        model_name: df[text_column_name].astype(str).tolist()
        for model_name, text_column_name in model_text_columns.items()
    }
    y_true = df["label"].astype(str).tolist()
    sample_ids = None
    if args.sample_id_column:
        if args.sample_id_column not in df.columns:
            raise SystemExit(
                f"Sample id column '{args.sample_id_column}' not found in {data_csv}. "
                f"Available columns: {sorted(df.columns)}"
            )
        sample_ids = df[args.sample_id_column].astype(str).tolist()

    prob_cube, logits_cube, logit_models = _score_models(
        model_names,
        model_texts,
        batch_size=args.batch_size,
        calibration_profile=args.calibration_profile,
    )

    output_path = Path(args.output) if args.output else _default_output_path(data_csv, model_names)
    metadata = {
        "created_at": _utcnow(),
        "source_csv": str(data_csv),
        "text_column": resolved_text_column,
        "text_columns_by_model": model_text_columns,
        "text_column_strategy": "per_model_auto" if args.text_column == "auto" else "forced",
        "label_column": args.label_column,
        "sample_size": len(df),
        "models": model_names,
        "labels": LABELS,
        "calibration_profile": args.calibration_profile,
        "split_metadata_path": str(split_metadata_path) if split_metadata_path else None,
        "split_provenance": summarize_split_provenance(split_metadata),
        "sample_id_column": args.sample_id_column,
        "logit_models": logit_models,
    }

    save_probability_cube(
        output_path,
        prob_cube=prob_cube,
        logits_cube=logits_cube,
        model_names=model_names,
        labels=LABELS,
        y_true=y_true,
        texts=texts if args.include_texts else None,
        sample_ids=sample_ids,
        metadata=metadata,
    )

    summary = {
        **metadata,
        "output_npz": str(output_path),
        "output_json": str(output_path.with_suffix(".json")),
        "prob_cube_shape": list(prob_cube.shape),
        "contains_logits": logits_cube is not None,
        "texts_included": bool(args.include_texts),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved probability cube → {output_path}")
    print(f"Saved metadata        → {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
