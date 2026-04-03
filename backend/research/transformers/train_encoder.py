#!/usr/bin/env python3
"""
Train a transformer encoder baseline on the leakage-safe YouTube split.

This script is the Phase 1 training entry point for Route A. It expects the
standard split files produced by `backend/scripts/prepare/prepare_hf_dataset.py`
and writes a Hugging Face sequence-classification artifact directly into the
runtime model directory under `backend/models/transformers/<preset>`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.model_registry import get_encoder_spec, list_encoder_presets
from src.sentiment.base import normalize_label
from src.utils.config import Config

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
TEXT_COLUMN_PRIORITY = ["text", "text_transformer", "text_raw", "text_classical"]


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def resolve_text_column(
    columns: Sequence[str],
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

    for candidate in TEXT_COLUMN_PRIORITY:
        if candidate in column_names:
            return candidate

    raise ValueError(
        "No supported text column found. "
        f"Tried: {TEXT_COLUMN_PRIORITY}. Available columns: {sorted(column_names)}"
    )


def load_split_frame(
    csv_path: Path,
    *,
    text_column: str = "auto",
    label_column: str = "label",
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    resolved_text_column = resolve_text_column(df.columns, preferred=text_column)
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {csv_path}. "
            f"Available columns: {sorted(df.columns)}"
        )

    df = df.dropna(subset=[resolved_text_column, label_column]).copy()
    df[resolved_text_column] = df[resolved_text_column].astype(str).str.strip()
    df = df[df[resolved_text_column].astype(bool)]
    df["label"] = df[label_column].apply(normalize_label)
    df = df[df["label"].isin(LABEL_TO_ID)]
    df = df.reset_index(drop=True)
    return df, resolved_text_column


def load_split_metadata(
    train_csv: Path,
    explicit_path: Optional[Path] = None,
) -> Tuple[Optional[dict], Optional[Path]]:
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.append(train_csv.with_name("split_metadata.json"))
    candidates.append(Config.DATA_DIR / "split_metadata.json")

    for candidate in candidates:
        if candidate and candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
    return None, None


def summarize_split_provenance(split_metadata: Optional[dict]) -> Optional[str]:
    if not split_metadata:
        return None
    split_info = split_metadata.get("split", {})
    strategy = split_info.get("strategy") or "unknown"
    random_state = split_info.get("random_state") or "na"
    rows = split_info.get("rows") or {}
    train_rows = rows.get("train", "na")
    val_rows = rows.get("val", "na")
    test_rows = rows.get("test", "na")
    preprocess = split_metadata.get("youtube_preprocess", {})
    primary_text_profile = preprocess.get("primary_text_profile", "text")
    return (
        f"{strategy}_seed{random_state}_"
        f"train{train_rows}_val{val_rows}_test{test_rows}_"
        f"{primary_text_profile}"
    )


def build_run_name(model_preset: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{model_preset}_{timestamp}"


def _import_training_stack():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Transformer training dependencies are missing.\n\n"
            "Install the deep-learning extras first:\n"
            "  pip install -r backend/requirements-dl.txt\n"
            "  pip install -r backend/requirements.txt\n\n"
            f"Import error: {exc}"
        )

    return {
        "torch": torch,
        "AutoTokenizer": AutoTokenizer,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _label_distribution(labels: Iterable[str]) -> Dict[str, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _to_serializable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transformer encoder baseline.")
    parser.add_argument(
        "--model_preset",
        default="modernbert",
        choices=list_encoder_presets(),
        help="Named encoder preset to fine-tune.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Optional explicit model name or local path override.",
    )
    parser.add_argument(
        "--train_csv",
        default=str(Config.DATA_DIR / "train.csv"),
        help="Path to the training split CSV.",
    )
    parser.add_argument(
        "--val_csv",
        default=str(Config.DATA_DIR / "val.csv"),
        help="Path to the validation split CSV.",
    )
    parser.add_argument(
        "--test_csv",
        default=str(Config.DATA_DIR / "test.csv"),
        help="Path to the test split CSV.",
    )
    parser.add_argument(
        "--text_column",
        default="auto",
        help="Text column to use. Defaults to the canonical split column.",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        help="Label column name.",
    )
    parser.add_argument(
        "--split_metadata",
        default=None,
        help="Optional explicit path to split_metadata.json.",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Artifact directory. Defaults to backend/models/transformers/<preset>.",
    )
    parser.add_argument(
        "--results_output",
        default=None,
        help="Optional JSON summary path. Defaults to backend/results/<preset>_encoder_metrics.json.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run name stored in metadata.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Allow overwriting an existing artifact directory.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training when supported.",
    )
    args = parser.parse_args()

    spec = get_encoder_spec(args.model_preset)
    model_name_or_path = args.model_name_or_path or spec.hf_name
    max_length = args.max_length or spec.default_max_length
    run_name = args.run_name or build_run_name(spec.key)
    artifact_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Config.get_model_path(spec.key) or (Config.MODELS_DIR / "transformers" / spec.key))
    )
    artifact_dir = Path(artifact_dir)
    results_output = (
        Path(args.results_output)
        if args.results_output
        else (BACKEND_ROOT / "results" / f"{spec.key}_encoder_metrics.json")
    )
    results_output.parent.mkdir(parents=True, exist_ok=True)

    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not args.overwrite_output_dir:
        raise SystemExit(
            f"Output directory '{artifact_dir}' already exists and is not empty.\n"
            "Re-run with --overwrite_output_dir or provide a different --output_dir."
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)

    train_df, train_text_column = load_split_frame(
        train_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    val_df, val_text_column = load_split_frame(
        val_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    test_df, test_text_column = load_split_frame(
        test_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    split_metadata, split_metadata_path = load_split_metadata(
        train_path,
        explicit_path=Path(args.split_metadata) if args.split_metadata else None,
    )
    split_id = summarize_split_provenance(split_metadata)

    stack = _import_training_stack()
    torch = stack["torch"]
    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForSequenceClassification = stack["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = stack["DataCollatorWithPadding"]
    EarlyStoppingCallback = stack["EarlyStoppingCallback"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]
    set_seed = stack["set_seed"]

    set_seed(args.seed)

    class EncodedTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: Sequence[str], labels: Sequence[str], tokenizer, max_length: int):
            self.encodings = tokenizer(
                list(texts),
                truncation=True,
                max_length=max_length,
            )
            self.labels = [LABEL_TO_ID[label] for label in labels]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Dict[str, object]:
            item = {key: value[idx] for key, value in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(LABELS),
        id2label={index: label for label, index in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
        use_safetensors=True,
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = EncodedTextDataset(
        train_df[train_text_column].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = EncodedTextDataset(
        val_df[val_text_column].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = EncodedTextDataset(
        test_df[test_text_column].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "macro_precision": precision,
            "macro_recall": recall,
        }

    training_args_kwargs = {
        "output_dir": str(artifact_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "run_name": run_name,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": max(1, args.logging_steps),
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "fp16": bool(args.fp16),
        "seed": args.seed,
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_args_kwargs["eval_strategy"] = "epoch"
    if "save_safetensors" in training_args_signature.parameters:
        training_args_kwargs["save_safetensors"] = True

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))

    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    train_metrics = train_result.metrics

    summary = {
        "created_at": _utcnow(),
        "run_name": run_name,
        "model": {
            "preset": spec.key,
            "description": spec.description,
            "hf_name": spec.hf_name,
            "resolved_model_name_or_path": model_name_or_path,
            "artifact_dir": str(artifact_dir),
            "max_length": max_length,
            "multilingual": spec.multilingual,
        },
        "dataset": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "text_columns": {
                "train": train_text_column,
                "val": val_text_column,
                "test": test_text_column,
            },
            "label_column": args.label_column,
            "split_metadata_path": str(split_metadata_path) if split_metadata_path else None,
            "split_id": split_id,
            "rows": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "label_distribution": {
                "train": _label_distribution(train_df["label"]),
                "val": _label_distribution(val_df["label"]),
                "test": _label_distribution(test_df["label"]),
            },
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "fp16": bool(args.fp16),
        },
        "metrics": {
            "train": _to_serializable_metrics(train_metrics),
            "val": _to_serializable_metrics(val_metrics),
            "test": _to_serializable_metrics(test_metrics),
        },
        "split_metadata": split_metadata,
    }

    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "encoder_spec.json").write_text(
        json.dumps(asdict(spec), indent=2),
        encoding="utf-8",
    )
    results_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved model artifact to: {artifact_dir}")
    print(f"Saved training summary to: {artifact_dir / 'training_summary.json'}")
    print(f"Saved metrics summary to: {results_output}")
    print(
        "Test metrics: "
        f"macro_f1={test_metrics.get('test_macro_f1', 'n/a')}, "
        f"accuracy={test_metrics.get('test_accuracy', 'n/a')}"
    )


if __name__ == "__main__":
    main()
