#!/usr/bin/env python3
"""
Create a reproducible label-balanced subset from an existing train/val/test split.

This is intended for CPU-constrained Route A development runs where the full
transformer-aligned split is too large for fast iteration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd


VALID_SPLITS = ("train", "val", "test")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sample_per_label(
    df: pd.DataFrame,
    *,
    label_column: str,
    per_label: int,
    seed: int,
) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby(label_column):
        take = min(int(per_label), len(group))
        parts.append(group.sample(n=take, random_state=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _count_labels(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    return {str(label): int(count) for label, count in df[label_column].value_counts().to_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a label-balanced subset from an existing split directory.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing train.csv, val.csv, test.csv, and optionally split_metadata.json.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the sampled subset split files.",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        help="Label column name.",
    )
    parser.add_argument(
        "--per_label",
        type=int,
        default=None,
        help="Default rows per label for every split unless overridden.",
    )
    parser.add_argument("--per_label_train", type=int, default=None)
    parser.add_argument("--per_label_val", type=int, default=None)
    parser.add_argument("--per_label_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_label_defaults = {
        "train": args.per_label_train if args.per_label_train is not None else args.per_label,
        "val": args.per_label_val if args.per_label_val is not None else args.per_label,
        "test": args.per_label_test if args.per_label_test is not None else args.per_label,
    }
    if any(value is None for value in per_label_defaults.values()):
        raise SystemExit("Provide --per_label or explicit --per_label_train/val/test overrides.")

    summary = {
        "created_at": _utcnow(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "label_column": args.label_column,
        "seed": int(args.seed),
        "splits": {},
    }

    source_metadata_path = input_dir / "split_metadata.json"
    if source_metadata_path.exists():
        summary["source_split_metadata_path"] = str(source_metadata_path)

    for split_name in VALID_SPLITS:
        input_path = input_dir / f"{split_name}.csv"
        if not input_path.exists():
            raise SystemExit(f"Missing split file: {input_path}")

        df = pd.read_csv(input_path)
        if args.label_column not in df.columns:
            raise SystemExit(
                f"Label column '{args.label_column}' not found in {input_path}. "
                f"Available columns: {sorted(df.columns)}"
            )

        sampled = _sample_per_label(
            df,
            label_column=args.label_column,
            per_label=per_label_defaults[split_name],
            seed=args.seed,
        )
        output_path = output_dir / f"{split_name}.csv"
        sampled.to_csv(output_path, index=False)

        summary["splits"][split_name] = {
            "source_path": str(input_path),
            "output_path": str(output_path),
            "requested_per_label": int(per_label_defaults[split_name]),
            "rows": int(len(sampled)),
            "label_distribution": _count_labels(sampled, args.label_column),
        }

        print(
            f"Saved {split_name}: {output_path} "
            f"({len(sampled):,} rows, {summary['splits'][split_name]['label_distribution']})"
        )

    metadata_path = output_dir / "subset_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
