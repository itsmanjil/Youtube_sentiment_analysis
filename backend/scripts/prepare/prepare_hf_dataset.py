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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply the API's `YouTubePreprocessor` to the text column.

    Returns
    -------
    (df_out, stats)
        df_out contains at least ['text', 'label'] (and preserves group columns if present).
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

    out_texts = []
    out_labels = []
    out_groups = [] if group_vals is not None else None

    stats = Counter()

    def handle_one(raw_text: str, label: str, group_value: Optional[str]) -> None:
        processed, meta = pre.preprocess_youtube_comment(
            raw_text,
            emoji_mode=emoji_mode,
            check_spam=filter_spam,
            check_lang=filter_language,
            min_words=min_words,
        )

        if meta.get("filtered"):
            reason = str(meta.get("filter_reason") or "unknown")
            stats[reason] += 1
            return

        processed = "" if processed is None else str(processed).strip()
        if not processed:
            stats["empty_after_processing"] += 1
            return

        out_texts.append(processed)
        out_labels.append(label)
        if out_groups is not None and group_value is not None:
            out_groups.append(group_value)

    n = len(texts)
    if chunk_size <= 0:
        chunk_size = n

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for idx in range(start, end):
            handle_one(texts[idx], labels[idx], group_vals[idx] if group_vals is not None else None)

        # Minimal progress signal for long runs.
        if n >= 50000:
            print(f"Processed {end}/{n} rows")

    out = pd.DataFrame({"text": out_texts, "label": out_labels})
    if out_groups is not None:
        out[group_col] = out_groups

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
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (_backend_dir() / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = str(args.source)
    df = pd.read_csv(source)
    raw_rows = int(len(df))
    raw_columns = list(df.columns)

    text_col, label_col, group_col = _resolve_schema(df)

    # Reduce memory footprint early.
    use_cols = [text_col, label_col] + ([group_col] if group_col else [])
    df = df[use_cols].copy()

    # Normalize + validate
    df = df.dropna(subset=[text_col, label_col])
    df = df.rename(columns={text_col: "text", label_col: "label"})
    if group_col:
        df = df.rename(columns={group_col: "__group__"})

    df["text"] = _normalize_text(df["text"])
    df["label"] = df["label"].astype(str).str.title().str.strip()

    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].astype(bool)]

    youtube_stats: Dict[str, int] = {}
    if args.youtube_preprocess:
        df, youtube_stats = _apply_youtube_preprocessing(
            df,
            emoji_mode=args.emoji_mode,
            filter_spam=bool(args.filter_spam),
            filter_language=bool(args.filter_language),
            min_words=int(args.min_words),
            chunk_size=int(args.chunk_size),
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

    # Export (keep only the required columns).
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    final_train[["text", "label"]].to_csv(train_path, index=False)
    val_df[["text", "label"]].to_csv(val_path, index=False)
    test_df[["text", "label"]].to_csv(test_path, index=False)

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
        "youtube_preprocess": {
            "enabled": bool(args.youtube_preprocess),
            "emoji_mode": args.emoji_mode,
            "min_words": int(args.min_words),
            "filter_spam": bool(args.filter_spam),
            "filter_language": bool(args.filter_language),
            "chunk_size": int(args.chunk_size),
            "filter_stats": youtube_stats,
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
