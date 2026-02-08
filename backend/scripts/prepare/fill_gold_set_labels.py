#!/usr/bin/env python3
"""
Fill a gold-set template with labels from a labeled source CSV.

Important
---------
This is NOT a replacement for a true human-labeled gold set.
It simply restores/derives labels from an existing labeled dataset
(e.g., `backend/data/test.csv`) so you can:
  - sanity-check evaluation scripts end-to-end
  - compare "dataset labels" vs later human labels (if you add a column)

For a thesis-grade gold set, you should manually annotate the `label` column
in the template file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill gold set template labels from a labeled source CSV.",
    )
    parser.add_argument("--template_csv", required=True, help="Template CSV (expects a text column).")
    parser.add_argument("--source_csv", required=True, help="Source labeled CSV (expects text+label columns).")
    parser.add_argument("--output_csv", required=True, help="Output labeled CSV path.")
    parser.add_argument("--template_text_column", default="text")
    parser.add_argument("--source_text_column", default="text")
    parser.add_argument("--source_label_column", default="label")
    args = parser.parse_args()

    template_path = Path(args.template_csv)
    source_path = Path(args.source_csv)
    output_path = Path(args.output_csv)

    if not template_path.exists():
        raise FileNotFoundError(f"Template CSV not found: {template_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    template = pd.read_csv(
        template_path,
        dtype={args.template_text_column: "string"},
        keep_default_na=False,
    )
    source = pd.read_csv(
        source_path,
        dtype={args.source_text_column: "string", args.source_label_column: "string"},
        keep_default_na=False,
    )

    if args.template_text_column not in template.columns:
        raise ValueError(
            f"Template is missing text column '{args.template_text_column}'. "
            f"Columns: {list(template.columns)}"
        )
    for col in (args.source_text_column, args.source_label_column):
        if col not in source.columns:
            raise ValueError(
                f"Source is missing required column '{col}'. Columns: {list(source.columns)}"
            )

    template = template.copy()
    source = source.copy()

    template["_text_norm"] = _normalize_text(template[args.template_text_column])
    source["_text_norm"] = _normalize_text(source[args.source_text_column])
    source["_label_norm"] = source[args.source_label_column].astype(str).str.strip()

    # Ensure we have a unique mapping from text->label in the source.
    dupe_counts = source["_text_norm"].value_counts()
    dupes = dupe_counts[dupe_counts > 1]
    if not dupes.empty:
        examples = ", ".join([repr(x) for x in dupes.index[:5].tolist()])
        raise ValueError(
            "Source CSV contains duplicate texts after normalization, so mapping is ambiguous. "
            f"Examples: {examples}"
        )

    label_map = source.set_index("_text_norm")["_label_norm"]
    template["source_label"] = template["_text_norm"].map(label_map)

    missing = template["source_label"].isna().sum()
    if missing:
        missing_texts = template.loc[template["source_label"].isna(), args.template_text_column].head(5).tolist()
        raise ValueError(
            f"Could not find labels for {missing} template rows in the source CSV. "
            f"First missing examples: {missing_texts}"
        )

    out = pd.DataFrame(
        {
            "text": template[args.template_text_column].astype(str),
            "source_label": template["source_label"].astype(str),
        }
    )
    # Fill `label` for convenience (so evaluators can run immediately).
    out["label"] = out["source_label"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("✅ Filled gold set labels from source dataset")
    print(f"   Template: {template_path}")
    print(f"   Source:   {source_path}")
    print(f"   Output:   {output_path}")
    print(f"   Rows:     {len(out)}")


if __name__ == "__main__":
    main()

