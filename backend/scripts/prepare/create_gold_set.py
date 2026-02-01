#!/usr/bin/env python3
"""
Create a small human-labeled gold set template.

This script samples comments from an input CSV and produces a clean
annotation template with an empty `label` column. It can optionally
stratify by an existing label column (e.g., auto labels) to ensure a
balanced or proportional sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ImportError:  # pragma: no cover - fallback for minimal environments
    pd = None


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _dedupe_by_text(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_column]).reset_index(drop=True)


def _sample_balanced(df: pd.DataFrame, label_column: str, n: int, seed: int) -> pd.DataFrame:
    labels = df[label_column].dropna().astype(str).unique().tolist()
    if not labels:
        return df.sample(n=min(n, len(df)), random_state=seed)

    per_class = max(1, n // len(labels))
    chunks: List[pd.DataFrame] = []
    remaining = df.copy()

    for label in labels:
        subset = remaining[remaining[label_column].astype(str) == label]
        take = min(per_class, len(subset))
        if take > 0:
            chunk = subset.sample(n=take, random_state=seed)
            chunks.append(chunk)
            remaining = remaining.drop(chunk.index)

    sampled = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=df.columns)
    remaining_needed = n - len(sampled)
    if remaining_needed > 0 and not remaining.empty:
        extra = remaining.sample(n=min(remaining_needed, len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _sample_proportional(df: pd.DataFrame, label_column: str, n: int, seed: int) -> pd.DataFrame:
    label_counts = df[label_column].value_counts(dropna=True)
    if label_counts.empty:
        return df.sample(n=min(n, len(df)), random_state=seed)

    sampled_chunks: List[pd.DataFrame] = []
    for label, count in label_counts.items():
        proportion = count / len(df)
        take = max(1, int(round(proportion * n)))
        subset = df[df[label_column] == label]
        take = min(take, len(subset))
        if take > 0:
            sampled_chunks.append(subset.sample(n=take, random_state=seed))

    sampled = pd.concat(sampled_chunks, ignore_index=True) if sampled_chunks else df.head(0)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remaining = df.drop(sampled.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a human-labeled gold set template from a CSV."
    )
    parser.add_argument("--input_csv", required=True, help="CSV with at least a text column.")
    parser.add_argument("--output_csv", default=None, help="Output CSV path.")
    parser.add_argument("--text_column", default="text", help="Name of the text column.")
    parser.add_argument("--label_column", default=None, help="Optional label column for stratified sampling.")
    parser.add_argument(
        "--stratify",
        choices=["balanced", "proportional", "none"],
        default="balanced",
        help="Sampling strategy when label_column is provided.",
    )
    parser.add_argument("--sample_size", type=int, default=300, help="Number of rows to sample.")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum text length (chars).")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dedupe", action="store_true", default=True, help="Remove duplicate texts.")
    parser.add_argument("--include_columns", default=None, help="Comma-separated extra columns to keep.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.parent / "gold_set_template.csv"
    )

    row_count = 0
    if pd is not None:
        df = pd.read_csv(input_path)
        if args.text_column not in df.columns:
            raise ValueError(f"Missing text column '{args.text_column}' in {input_path}")

        df = df.copy()
        df[args.text_column] = _normalize_text(df[args.text_column])
        df = df[df[args.text_column].str.len() >= args.min_length]
        df = df[df[args.text_column].astype(bool)]
        if args.dedupe:
            df = _dedupe_by_text(df, args.text_column)

        label_column = args.label_column if args.label_column in df.columns else None
        if args.stratify != "none" and label_column:
            if args.stratify == "balanced":
                sampled = _sample_balanced(df, label_column, args.sample_size, args.random_seed)
            else:
                sampled = _sample_proportional(df, label_column, args.sample_size, args.random_seed)
        else:
            sampled = df.sample(n=min(args.sample_size, len(df)), random_state=args.random_seed)

        extra_columns: List[str] = []
        if args.include_columns:
            extra_columns = [c.strip() for c in args.include_columns.split(",") if c.strip()]

        keep_columns = [args.text_column] + [c for c in extra_columns if c in sampled.columns]
        output = sampled[keep_columns].copy()
        output = output.rename(columns={args.text_column: "text"})
        if args.label_column and args.label_column in output.columns:
            output = output.rename(columns={args.label_column: "source_label"})
        output["label"] = ""
        output.to_csv(output_path, index=False)
        row_count = len(output)
    else:
        import csv
        import hashlib
        import random

        rng = random.Random(args.random_seed)

        def iter_rows():
            with input_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or args.text_column not in reader.fieldnames:
                    raise ValueError(f"Missing text column '{args.text_column}' in {input_path}")
                for row in reader:
                    yield row

        def text_ok(text: str) -> bool:
            if text is None:
                return False
            text = str(text).strip()
            if len(text) < args.min_length:
                return False
            return bool(text)

        def text_key(text: str) -> str:
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        label_column = args.label_column
        use_stratify = args.stratify != "none" and label_column

        # Pass 1: count labels (optional, for proportional targets)
        label_counts: Dict[str, int] = {}
        total_count = 0
        if use_stratify and args.stratify == "proportional":
            for row in iter_rows():
                text = row.get(args.text_column, "")
                if not text_ok(text):
                    continue
                label = str(row.get(label_column, "")).strip()
                if not label:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
                total_count += 1

        # Determine targets
        targets: Dict[str, int] = {}
        if use_stratify:
            if args.stratify == "balanced":
                labels = ["Positive", "Neutral", "Negative"]
                per_label = max(1, args.sample_size // len(labels))
                targets = {label: per_label for label in labels}
                remainder = args.sample_size - per_label * len(labels)
                for label in labels[: max(0, remainder)]:
                    targets[label] += 1
            else:
                if not label_counts:
                    use_stratify = False
                else:
                    for label, count in label_counts.items():
                        take = max(1, int(round((count / total_count) * args.sample_size)))
                        targets[label] = take
                    # Adjust to exact size
                    diff = args.sample_size - sum(targets.values())
                    labels = sorted(targets.keys())
                    idx = 0
                    while diff != 0 and labels:
                        label = labels[idx % len(labels)]
                        if diff > 0:
                            targets[label] += 1
                            diff -= 1
                        else:
                            if targets[label] > 1:
                                targets[label] -= 1
                                diff += 1
                        idx += 1

        # Pass 2: reservoir sampling
        seen_hashes = set()
        reservoirs: Dict[str, List[dict]] = {}
        seen_counts: Dict[str, int] = {}
        sample: List[dict] = []
        seen_total = 0

        for row in iter_rows():
            text = row.get(args.text_column, "")
            if not text_ok(text):
                continue
            norm_text = str(text).strip()
            if args.dedupe:
                key = text_key(norm_text)
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)

            if use_stratify:
                label = str(row.get(label_column, "")).strip()
                if label not in targets:
                    continue
                seen_counts[label] = seen_counts.get(label, 0) + 1
                reservoirs.setdefault(label, [])
                res = reservoirs[label]
                target = targets[label]
                if len(res) < target:
                    res.append(row)
                else:
                    j = rng.randint(0, seen_counts[label] - 1)
                    if j < target:
                        res[j] = row
            else:
                seen_total += 1
                if len(sample) < args.sample_size:
                    sample.append(row)
                else:
                    j = rng.randint(0, seen_total - 1)
                    if j < args.sample_size:
                        sample[j] = row

        if use_stratify:
            sample = []
            for rows in reservoirs.values():
                sample.extend(rows)
            rng.shuffle(sample)
            if len(sample) > args.sample_size:
                sample = sample[: args.sample_size]

        extra_columns: List[str] = []
        if args.include_columns:
            extra_columns = [c.strip() for c in args.include_columns.split(",") if c.strip()]

        output_columns = ["text"]
        for col in extra_columns:
            if col != args.text_column:
                output_columns.append(col)
        if label_column in extra_columns:
            output_columns.append("source_label")

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns + ["label"])
            writer.writeheader()
            for row in sample:
                record = {"text": str(row.get(args.text_column, "")).strip()}
                for col in extra_columns:
                    if col == args.text_column:
                        continue
                    record[col] = row.get(col, "")
                if label_column and label_column in extra_columns:
                    record["source_label"] = row.get(label_column, "")
                record["label"] = ""
                writer.writerow(record)
                row_count += 1

    print("\n✅ Gold set template created")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Rows:   {row_count}")
    print("   Next: fill in the 'label' column with Positive/Neutral/Negative.")


if __name__ == "__main__":
    main()
