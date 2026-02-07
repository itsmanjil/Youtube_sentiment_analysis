#!/usr/bin/env python3
"""
Check for train/val/test leakage caused by duplicate texts.

For text classification, exact-duplicate texts across splits can inflate reported
metrics (the model can effectively "see" test examples during training).

This script reports:
- row counts and duplicate-within-split counts
- unique-text overlap sizes between splits
- row-level overlap rates (rows in split B whose text also appears in split A)

Example:
  backend/venv/bin/python backend/scripts/prepare/check_split_leakage.py \\
    --train backend/data/train.csv \\
    --val backend/data/val.csv \\
    --test backend/data/test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _read_texts(path: Path, text_column: str) -> pd.Series:
    df = pd.read_csv(path, usecols=[text_column], dtype={text_column: "string"})
    texts = df[text_column].fillna("").astype(str)
    return texts


def _stats(texts: pd.Series) -> Dict[str, int]:
    total = int(len(texts))
    unique = int(texts.nunique(dropna=False))
    dupes = total - unique
    return {"rows": total, "unique_texts": unique, "duplicate_rows": int(dupes)}


def _overlap_rows(a_set: set[str], b_texts: pd.Series) -> Dict[str, float]:
    mask = b_texts.isin(a_set)
    count = int(mask.sum())
    total = int(len(b_texts))
    rate = float(count / total) if total else 0.0
    return {"rows": count, "rate": rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check leakage between dataset splits.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--val", required=True, help="Path to val CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--text-column", default="text", help="Text column name.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")

    train_texts = _read_texts(train_path, args.text_column)
    val_texts = _read_texts(val_path, args.text_column)
    test_texts = _read_texts(test_path, args.text_column)

    train_set = set(train_texts)
    val_set = set(val_texts)
    test_set = set(test_texts)

    report = {
        "splits": {
            "train": {"path": str(train_path), **_stats(train_texts)},
            "val": {"path": str(val_path), **_stats(val_texts)},
            "test": {"path": str(test_path), **_stats(test_texts)},
        },
        "overlap_unique_texts": {
            "train_val": int(len(train_set & val_set)),
            "train_test": int(len(train_set & test_set)),
            "val_test": int(len(val_set & test_set)),
        },
        "overlap_rows": {
            "val_in_train": _overlap_rows(train_set, val_texts),
            "test_in_train": _overlap_rows(train_set, test_texts),
            "test_in_val": _overlap_rows(val_set, test_texts),
        },
    }

    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

