#!/usr/bin/env python3
"""
Prepare a leakage-safe train/val/test split from the HuggingFace YouTube comment sentiment dataset.

Why this exists
---------------
For a thesis-grade evaluation, exact-duplicate texts must not appear across
splits; otherwise reported metrics can be inflated. This script:

- normalizes text + labels
- optionally applies the same YouTube preprocessing pipeline used by the API
- removes texts with conflicting labels
- deduplicates by final model-input text
- splits into disjoint train/val/test
- writes split provenance to split_metadata.json

Default behavior keeps language/spam filters OFF (fast, deterministic). Enable
`--filter_language`/`--filter_spam` to match production filtering behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

try:
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
except ImportError:  # pragma: no cover - depends on user's environment
    raise SystemExit(
        "Missing Python dependencies for dataset preparation.\n\n"
        "Recommended (use the repo's backend venv):\n"
        "  backend/venv/bin/python backend/scripts/prepare/prepare_hf_dataset.py --help\n\n"
        "Alternatively (create/activate your own venv and install deps):\n"
        "  python -m venv backend/venv\n"
        "  source backend/venv/bin/activate\n"
        "  python -m pip install -r backend/requirements.txt\n"
    )


DEFAULT_SOURCE_URI = (
    "hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv"
)
VALID_LABELS = {"Positive", "Neutral", "Negative"}


def _backend_dir() -> Path:
    # .../backend/scripts/prepare/prepare_hf_dataset.py -> parents[2] == backend
    return Path(__file__).resolve().parents[2]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _resolve_schema(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    text_col = "CommentText" if "CommentText" in df.columns else "comment_text"
    label_col = "Sentiment" if "Sentiment" in df.columns else "label"
    group_col = "VideoID" if "VideoID" in df.columns else None
    return text_col, label_col, group_col


def _apply_youtube_preprocessing(
    df: pd.DataFrame,
    *,
    emoji_mode: str,
    filter_spam: bool,
    filter_language: bool,
    min_words: int,
    chunk_size: int,
    primary_text_profile: str,
    metadata_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply the API's `YouTubePreprocessor` to the text column.

    Returns
    -------
    (df_out, stats)
        df_out contains at least ['text', 'label'] and may include
        ['text_raw', 'text_classical', 'text_transformer'].
        The canonical `text` column is the dedupe/split column.
        stats includes filter counts (spam/language/too_short/empty_after_processing).
    """
    # Make backend/ importable for `from app...` imports even when executed from repo root.
    backend_dir = _backend_dir()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.youtube_preprocessor import YouTubePreprocessor

    pre = YouTubePreprocessor()

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    group_vals = None
    group_col = None
    if "__group__" in df.columns:
        group_col = "__group__"
        group_vals = df[group_col].astype(str).tolist()
    metadata_values = {
        column: df[column].astype(str).tolist()
        for column in metadata_columns
        if column in df.columns
    }

    out_texts = []
    out_labels = []
    out_raw_texts = []
    out_classical_texts = []
    out_transformer_texts = []
    out_groups = [] if group_vals is not None else None
    out_metadata = {column: [] for column in metadata_values}

    stats = Counter()

    def handle_one(
        raw_text: str,
        label: str,
        group_value: Optional[str],
        row_metadata: Dict[str, str],
    ) -> None:
        selected_profile = (
            "transformer"
            if primary_text_profile == "normalized"
            else (primary_text_profile or "classical")
        )
        processed, meta = pre.preprocess_youtube_comment(
            raw_text,
            emoji_mode=emoji_mode,
            check_spam=filter_spam,
            check_lang=filter_language,
            min_words=min_words,
            profile=selected_profile,
        )

        if meta.get("filtered"):
            reason = str(meta.get("filter_reason") or "unknown")
            stats[reason] += 1
            return

        processed = "" if processed is None else str(processed).strip()
        if not processed:
            stats["empty_after_processing"] += 1
            return

        classical_text, _ = pre.preprocess_youtube_comment(
            raw_text,
            emoji_mode=emoji_mode,
            check_spam=False,
            check_lang=False,
            min_words=1,
            profile="classical",
        )
        transformer_text, _ = pre.preprocess_youtube_comment(
            raw_text,
            emoji_mode=emoji_mode,
            check_spam=False,
            check_lang=False,
            min_words=1,
            profile="transformer",
        )
        canonical_text = str(raw_text).strip() if primary_text_profile == "normalized" else processed
        out_texts.append(canonical_text)
        out_labels.append(label)
        out_raw_texts.append(str(raw_text).strip())
        out_classical_texts.append(classical_text or processed)
        out_transformer_texts.append(transformer_text or processed)
        if out_groups is not None and group_value is not None:
            out_groups.append(group_value)
        for column, value in row_metadata.items():
            out_metadata[column].append(value)

    n = len(texts)
    if chunk_size <= 0:
        chunk_size = n

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for idx in range(start, end):
            handle_one(
                texts[idx],
                labels[idx],
                group_vals[idx] if group_vals is not None else None,
                {column: values[idx] for column, values in metadata_values.items()},
            )

        # Minimal progress signal for long runs.
        if n >= 50000:
            print(f"Processed {end}/{n} rows")

    out = pd.DataFrame(
        {
            "text": out_texts,
            "label": out_labels,
            "text_raw": out_raw_texts,
            "text_classical": out_classical_texts,
            "text_transformer": out_transformer_texts,
        }
    )
    if out_groups is not None:
        out[group_col] = out_groups
    for column, values in out_metadata.items():
        out[column] = values

    return out, {k: int(v) for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF YouTube sentiment dataset splits.")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_URI,
        help="HF URI or local CSV path.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: backend/data).",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Optional raw-row cap applied before preprocessing/splitting for faster benchmark builds.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument(
        "--youtube_preprocess",
        action="store_true",
        help="Apply the API's YouTubePreprocessor to text before splitting.",
    )
    parser.add_argument(
        "--emoji_mode",
        choices=["remove", "convert", "keep"],
        default="convert",
        help="Emoji handling mode for YouTubePreprocessor.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Minimum word count (used by YouTubePreprocessor).",
    )
    parser.add_argument(
        "--filter_spam",
        action="store_true",
        help="Enable spam filtering (slower; matches production default).",
    )
    parser.add_argument(
        "--filter_language",
        action="store_true",
        help="Enable language detection filtering (slowest; matches production default).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Chunk size for Python-loop preprocessing (lower uses less memory).",
    )
    parser.add_argument(
        "--primary_text_profile",
        choices=["normalized", "classical", "transformer"],
        default=None,
        help=(
            "Canonical text profile used for dedupe/splitting and exported as `text`. "
            "Defaults to `classical` when --youtube_preprocess is enabled, otherwise `normalized`."
        ),
    )
    parser.add_argument(
        "--metadata_columns",
        default="",
        help=(
            "Comma-separated source metadata columns to retain in exported splits, "
            "or 'all' to keep all non-text/non-label source columns. Useful for "
            "domain-shift evaluation by VideoID, PublishedAt, CountryCode, or CategoryID."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (_backend_dir() / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = str(args.source)
    df = pd.read_csv(source)
    raw_rows = int(len(df))
    raw_columns = list(df.columns)

    text_col, label_col, group_col = _resolve_schema(df)
    requested_metadata = [item.strip() for item in str(args.metadata_columns).split(",") if item.strip()]
    if requested_metadata == ["all"]:
        metadata_columns = [
            column for column in raw_columns if column not in {text_col, label_col}
        ]
    else:
        missing_metadata = [column for column in requested_metadata if column not in raw_columns]
        if missing_metadata:
            raise SystemExit(
                "Requested metadata columns not found in source: "
                f"{missing_metadata}. Available columns: {raw_columns}"
            )
        metadata_columns = requested_metadata

    # Reduce memory footprint early.
    use_cols = list(dict.fromkeys([text_col, label_col] + ([group_col] if group_col else []) + metadata_columns))
    df = df[use_cols].copy()

    # Normalize + validate
    df = df.dropna(subset=[text_col, label_col])
    df = df.rename(columns={text_col: "text", label_col: "label"})
    if group_col:
        df = df.rename(columns={group_col: "__group__"})
        if group_col in metadata_columns:
            df[group_col] = df["__group__"]

    df["text"] = _normalize_text(df["text"])
    df["label"] = df["label"].astype(str).str.title().str.strip()

    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].astype(bool)]

    if args.sample_rows and args.sample_rows < len(df):
        sample_rows = int(args.sample_rows)
        stratify_labels = df["label"] if df["label"].nunique() > 1 else None
        df, _ = train_test_split(
            df,
            train_size=sample_rows,
            stratify=stratify_labels,
            random_state=args.random_seed,
        )
        df = df.reset_index(drop=True)
        print(f"Sampled {len(df):,}/{raw_rows:,} raw rows before preprocessing")

    primary_text_profile = args.primary_text_profile
    if primary_text_profile is None:
        primary_text_profile = "classical" if args.youtube_preprocess else "normalized"

    youtube_stats: Dict[str, int] = {}
    if args.youtube_preprocess:
        df, youtube_stats = _apply_youtube_preprocessing(
            df,
            emoji_mode=args.emoji_mode,
            filter_spam=bool(args.filter_spam),
            filter_language=bool(args.filter_language),
            min_words=int(args.min_words),
            chunk_size=int(args.chunk_size),
            primary_text_profile=primary_text_profile,
            metadata_columns=metadata_columns,
        )

    # Drop conflicting labels by (final) text.
    label_nunique = df.groupby("text")["label"].nunique()
    conflicting_texts = label_nunique[label_nunique > 1].index
    conflicting_count = int(len(conflicting_texts))
    if conflicting_count:
        print(f"Dropping {conflicting_count} texts with conflicting labels")
        df = df[~df["text"].isin(conflicting_texts)]

    # Deduplicate by final text, ensuring disjoint splits.
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed_duplicates = int(before - len(df))
    if removed_duplicates:
        print(f"Removed {removed_duplicates} duplicate texts before splitting")

    # Split
    split_meta = {
        "strategy": "group" if "__group__" in df.columns else "stratified",
        "random_state": int(args.random_seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
    }

    if "__group__" in df.columns:
        gss = GroupShuffleSplit(
            test_size=args.test_size,
            n_splits=1,
            random_state=args.random_seed,
        )
        train_idx, test_idx = next(gss.split(df, groups=df["__group__"]))
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        gss2 = GroupShuffleSplit(
            test_size=args.val_size,
            n_splits=1,
            random_state=args.random_seed,
        )
        train_idx, val_idx = next(gss2.split(train_df, groups=train_df["__group__"]))
        final_train, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            stratify=df["label"],
            random_state=args.random_seed,
        )
        final_train, val_df = train_test_split(
            train_df,
            test_size=args.val_size,
            stratify=train_df["label"],
            random_state=args.random_seed,
        )

    # Export the canonical text column plus optional profile columns.
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    export_columns = ["text", "label"]
    for extra_column in ("text_raw", "text_classical", "text_transformer"):
        if extra_column in final_train.columns:
            export_columns.append(extra_column)
    for metadata_column in metadata_columns:
        if metadata_column in final_train.columns and metadata_column not in export_columns:
            export_columns.append(metadata_column)

    final_train[export_columns].to_csv(train_path, index=False)
    val_df[export_columns].to_csv(val_path, index=False)
    test_df[export_columns].to_csv(test_path, index=False)

    metadata = {
        "created_at": _utcnow(),
        "source": {"uri": source, "raw_rows": raw_rows, "raw_columns": raw_columns},
        "schema": {"text_col": text_col, "label_col": label_col, "group_col": group_col},
        "normalization": {
            "text_whitespace_collapse": True,
            "text_strip": True,
            "label_title_case": True,
            "valid_labels": sorted(VALID_LABELS),
        },
        "sampling": {
            "enabled": bool(args.sample_rows),
            "sample_rows": int(args.sample_rows) if args.sample_rows else None,
            "rows_after_sampling": int(len(df)),
            "random_state": int(args.random_seed),
            "stratify_on_label": True,
        },
        "youtube_preprocess": {
            "enabled": bool(args.youtube_preprocess),
            "primary_text_profile": primary_text_profile,
            "emoji_mode": args.emoji_mode,
            "min_words": int(args.min_words),
            "filter_spam": bool(args.filter_spam),
            "filter_language": bool(args.filter_language),
            "chunk_size": int(args.chunk_size),
            "filter_stats": youtube_stats,
            "export_columns": export_columns,
            "metadata_columns": metadata_columns,
        },
        "dedupe": {
            "conflicting_texts_dropped": conflicting_count,
            "duplicate_text_rows_removed": removed_duplicates,
        },
        "split": {
            **split_meta,
            "paths": {"train": str(train_path), "val": str(val_path), "test": str(test_path)},
            "rows": {
                "train": int(len(final_train)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "label_distribution": {
                "train": final_train["label"].value_counts().to_dict(),
                "val": val_df["label"].value_counts().to_dict(),
                "test": test_df["label"].value_counts().to_dict(),
            },
        },
    }

    meta_path = output_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
