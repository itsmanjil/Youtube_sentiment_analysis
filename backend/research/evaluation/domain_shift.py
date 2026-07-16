#!/usr/bin/env python3
"""
Domain-shift and robustness-slice evaluation for sentiment models.

When channel/topic/time metadata exists, this script evaluates model performance
per metadata group. The checked-in benchmark CSVs currently only contain text
and labels, so the default fallback is a text-length robustness analysis. That
fallback is not a replacement for cross-channel validation, but it is a
runnable slice check and documents the missing metadata requirement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label


LABELS = ["Negative", "Neutral", "Positive"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BACKEND_ROOT / path


def parse_models(raw: str) -> List[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def iter_chunks(values: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])


def load_frame(path: Path, text_column: str, label_column: str, sample: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(
            f"Dataset must contain '{text_column}' and '{label_column}'. "
            f"Available columns: {sorted(frame.columns)}"
        )
    frame = frame.copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame[label_column] = frame[label_column].map(normalize_label)
    frame = frame[frame[text_column].astype(bool)]
    frame = frame[frame[label_column].isin(LABELS)]
    if sample and sample < len(frame):
        frame = frame.iloc[:sample].copy()
    if frame.empty:
        raise ValueError("No labeled rows available for domain-shift evaluation.")
    return frame.reset_index(drop=True)


def add_length_slices(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    frame = frame.copy()
    lengths = frame[text_column].astype(str).str.split().map(len)
    frame["_text_length_tokens"] = lengths
    try:
        frame["_domain_slice"] = pd.qcut(
            lengths,
            q=4,
            labels=["very_short", "short", "medium", "long"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        frame["_domain_slice"] = "all"
    return frame


def choose_slice_column(frame: pd.DataFrame, requested: str | None, text_column: str) -> tuple[pd.DataFrame, str, str]:
    if requested and requested in frame.columns:
        return frame, requested, "metadata"

    candidates = [
        "CategoryID",
        "CountryCode",
        "category_id",
        "country_code",
        "category",
        "channel_id",
        "channel",
        "channel_name",
        "published_month",
        "VideoID",
        "video_id",
        "topic",
        "published_at",
        "PublishedAt",
        "date",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            return frame, candidate, "metadata"

    return add_length_slices(frame, text_column), "_domain_slice", "text_length_proxy"


def score_model(model_name: str, texts: Sequence[str], chunk_size: int) -> List[str]:
    engine_kwargs: Dict[str, object] = {}
    if model_name == "ensemble_pso":
        engine_name = "ensemble"
        engine_kwargs["weights_optimization"] = "pso"
    elif model_name == "ensemble_nsga2":
        engine_name = "ensemble"
        engine_kwargs["weights_optimization"] = "nsga2"
    else:
        engine_name = model_name

    engine = get_sentiment_engine(engine_name, **engine_kwargs)
    predictions: List[str] = []
    for chunk in iter_chunks(texts, chunk_size):
        if hasattr(engine, "batch_analyze"):
            results = engine.batch_analyze(chunk)
        else:
            results = [engine.analyze(text) for text in chunk]
        predictions.extend(
            coerce_sentiment_result(result, engine_name).label for result in results
        )
    return predictions


def metric_record(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    return {
        "n_samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }


def evaluate_slices(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    slice_column: str,
    models: Sequence[str],
    chunk_size: int,
    min_slice_size: int,
) -> Dict[str, object]:
    texts = frame[text_column].astype(str).tolist()
    y_true = frame[label_column].astype(str).tolist()
    results: Dict[str, object] = {}

    for model_name in models:
        y_pred = score_model(model_name, texts, chunk_size)
        overall = metric_record(y_true, y_pred)

        slice_rows = []
        pred_series = pd.Series(y_pred, index=frame.index)
        for slice_value, group in frame.groupby(slice_column, dropna=False):
            if len(group) < min_slice_size:
                continue
            idx = list(group.index)
            row = {
                "slice": str(slice_value),
                **metric_record(
                    group[label_column].astype(str).tolist(),
                    pred_series.loc[idx].astype(str).tolist(),
                ),
            }
            slice_rows.append(row)

        slice_rows.sort(key=lambda item: item["macro_f1"])
        worst = slice_rows[0] if slice_rows else None
        best = slice_rows[-1] if slice_rows else None
        results[model_name] = {
            "overall": overall,
            "slices": slice_rows,
            "worst_slice": worst,
            "best_slice": best,
            "macro_f1_spread": round(
                float(best["macro_f1"] - worst["macro_f1"]), 6
            )
            if best and worst
            else None,
        }

    return results


def build_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Domain-Shift / Robustness Slice Evaluation\n",
        f"- Created at: `{payload['created_at']}`",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Slice column: `{payload['slice_column']}`",
        f"- Slice source: `{payload['slice_source']}`",
        f"- Samples: `{payload['n_samples']}`",
        "",
    ]

    if payload["slice_source"] == "text_length_proxy":
        lines.extend(
            [
                "> The input dataset has no channel/topic/time metadata. Results below are",
                "> text-length robustness slices, not a full cross-domain validation.",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall",
            "",
            "| Model | Accuracy | Macro-F1 | Worst Slice | Worst Slice F1 | Spread |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for model_name, result in payload["results"].items():
        worst = result["worst_slice"] or {}
        lines.append(
            f"| {model_name} | {result['overall']['accuracy']:.6f} | "
            f"{result['overall']['macro_f1']:.6f} | {worst.get('slice', 'n/a')} | "
            f"{worst.get('macro_f1', 0):.6f} | {result.get('macro_f1_spread') or 0:.6f} |"
        )

    for model_name, result in payload["results"].items():
        lines.extend(
            [
                "",
                f"## {model_name} Slices",
                "",
                "| Slice | Samples | Accuracy | Macro-F1 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in result["slices"]:
            lines.append(
                f"| {row['slice']} | {row['n_samples']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} |"
            )

    lines.extend(["", "## Interpretation", "", payload["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment models across domain or robustness slices.")
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--slice_column", default=None)
    parser.add_argument("--models", default="logreg,svm,tfidf,ensemble_nsga2,meta_learner")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--min_slice_size", type=int, default=30)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    dataset_path = _resolve_path(args.data)
    frame = load_frame(dataset_path, args.text_column, args.label_column, args.sample)
    frame, slice_column, slice_source = choose_slice_column(
        frame,
        args.slice_column,
        args.text_column,
    )

    models = parse_models(args.models)
    results = evaluate_slices(
        frame,
        text_column=args.text_column,
        label_column=args.label_column,
        slice_column=slice_column,
        models=models,
        chunk_size=args.chunk_size,
        min_slice_size=args.min_slice_size,
    )

    interpretation = (
        "This is a metadata-backed domain-slice evaluation."
        if slice_source == "metadata"
        else (
            "This is a proxy robustness check because the selected dataset has no "
            "channel/topic/time metadata. Add those columns and rerun with "
            "`--slice_column` for a true domain-shift evaluation."
        )
    )

    payload = {
        "title": "Domain-Shift / Robustness Slice Evaluation",
        "created_at": _utcnow(),
        "dataset_path": str(dataset_path),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "slice_column": slice_column,
        "slice_source": slice_source,
        "n_samples": int(len(frame)),
        "min_slice_size": int(args.min_slice_size),
        "models": models,
        "results": results,
        "interpretation": interpretation,
    }

    output_root = BACKEND_ROOT / "results" / "domain_shift"
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_root / "domain_shift_evaluation.json"
    output_md = Path(args.output_md) if args.output_md else output_root / "domain_shift_evaluation.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
