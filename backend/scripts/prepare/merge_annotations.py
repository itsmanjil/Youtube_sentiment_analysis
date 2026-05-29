#!/usr/bin/env python3
"""
Merge multiple annotator CSV files into a reconciled gold set.

Each annotator file must have at minimum:
  - text   : the comment text (used as the join key)
  - label  : the annotation (Positive / Neutral / Negative)

Usage
-----
# Two annotators
python merge_annotations.py \\
    --annotators data/gold_set_annotator_1.csv data/gold_set_annotator_2.csv \\
    --names "Dataset" "Silver" \\
    --output data/gold_set_human_reconciled.csv \\
    --report results/gold_set/iaa_report.json

# Three annotators
python merge_annotations.py \\
    --annotators ann1.csv ann2.csv ann3.csv \\
    --names "Annotator_A" "Annotator_B" "Annotator_C"

Output files
------------
- <output>.csv    : merged gold set with per-annotator columns + gold_label + is_disputed
- <report>.json   : full IAA metrics
- <report>.md     : human-readable IAA summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from research.iaa import iaa_report, reconcile_annotations

LABEL_ALIASES = {
    "positive": "Positive", "pos": "Positive", "2": "Positive",
    "neutral":  "Neutral",  "neu": "Neutral",  "1": "Neutral",
    "negative": "Negative", "neg": "Negative", "0": "Negative",
}


def _normalize_label(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return LABEL_ALIASES.get(s.lower(), s if s else None)


def _load_annotator_csv(
    path: Path,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """Load ordered texts and {text -> label} from a CSV file."""
    allowed = set(LABEL_ALIASES.values())
    texts: List[str] = []
    mapping: Dict[str, Optional[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found in {path}. Available: {reader.fieldnames}")
        if label_col not in reader.fieldnames:
            raise ValueError(f"Column '{label_col}' not found in {path}. Available: {reader.fieldnames}")
        for row in reader:
            text = str(row[text_col]).strip()
            if not text:
                continue
            label = _normalize_label(row[label_col])
            if label is not None and label not in allowed:
                raise ValueError(
                    f"Invalid label {label!r} in {path}. "
                    f"Expected one of: {', '.join(sorted(allowed))}."
                )
            if text in mapping:
                raise ValueError(f"Duplicate text found in {path}: {text[:80]!r}")
            texts.append(text)
            mapping[text] = label
    return texts, mapping


def _build_matrix(
    texts: List[str],
    annotator_maps: List[Dict[str, Optional[str]]],
) -> List[List[Optional[str]]]:
    """Build (n_items, n_annotators) annotation matrix."""
    return [
        [ann_map.get(t) for ann_map in annotator_maps]
        for t in texts
    ]


def _to_markdown(report: dict, output_csv: Path) -> str:
    lines: List[str] = []
    lines.append("# Inter-Annotator Agreement Report")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at_utc']}")
    lines.append(f"- Gold set: `{output_csv}`")
    lines.append(f"- Annotators: {', '.join(report['annotator_names'])}")
    lines.append(f"- Items: {report['n_items']}  |  Fully annotated: {report['fully_annotated_items']}  |  Disputed: {report['disputed_items']}")
    lines.append("")

    m = report["metrics"]
    lines.append("## Agreement Metrics")
    lines.append("")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Percent agreement | {m['percent_agreement']:.4f} | — |")
    alpha = m['krippendorff_alpha']
    lines.append(f"| Krippendorff alpha | {alpha:.4f} | {report['interpretation']} |")
    lines.append(f"| Fleiss kappa | {m['fleiss_kappa']:.4f} | - |")
    for pair, kappa in m["cohen_kappa_pairwise"].items():
        kappa_str = f"{kappa:.4f}" if kappa == kappa else "n/a"
        lines.append(f"| Cohen kappa ({pair.replace('_vs_', ' vs ')}) | {kappa_str} | - |")
    lines.append("")

    lines.append("## Gold Label Distribution (after reconciliation)")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("| --- | --- |")
    for lbl, cnt in sorted(report["gold_label_distribution"].items()):
        lines.append(f"| {lbl} | {cnt} |")
    if report["disputed_items"] > 0:
        lines.append(f"| *Disputed (no majority)* | {report['disputed_items']} |")
    lines.append("")

    lines.append("## Per-Annotator Label Distribution")
    lines.append("")
    header = "| Label | " + " | ".join(report["annotator_names"]) + " |"
    sep = "| --- |" + " --- |" * len(report["annotator_names"])
    lines.append(header)
    lines.append(sep)
    for lbl in ["Negative", "Neutral", "Positive"]:
        row_vals = [
            str(report["per_annotator_distribution"].get(name, {}).get(lbl, 0))
            for name in report["annotator_names"]
        ]
        lines.append(f"| {lbl} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## Interpreting Krippendorff's Alpha")
    lines.append("")
    lines.append("| Range | Interpretation |")
    lines.append("| --- | --- |")
    lines.append("| alpha >= 0.80 | Strong agreement - gold labels are reliable |")
    lines.append("| 0.67 <= alpha < 0.80 | Tentative agreement - Krippendorff's minimum for tentative conclusions |")
    lines.append("| 0.40 <= alpha < 0.67 | Moderate agreement - treat gold labels with caution |")
    lines.append("| alpha < 0.40 | Poor agreement - adjudication required |")
    lines.append("")

    if report["disputed_items"] > 0:
        lines.append(f"> **Note:** {report['disputed_items']} items had no majority label.")
        lines.append("> These are marked `is_disputed=True` in the gold set and should be")
        lines.append("> reviewed and adjudicated before use as ground truth.")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge annotator CSVs into a reconciled gold set with IAA metrics."
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more annotator CSV files (text + label columns).",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Friendly names for each annotator (same order as --annotators).",
    )
    parser.add_argument("--text_col", default="text", help="Name of the text column (default: text).")
    parser.add_argument("--label_col", default="label", help="Name of the label column (default: label).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/gold_set_human_reconciled.csv).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output IAA report JSON path (default: results/gold_set/iaa_report.json).",
    )
    parser.add_argument(
        "--union",
        action="store_true",
        default=False,
        help="Include ALL texts from any annotator (default: intersection only).",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        default=False,
        help="Allow missing labels in annotator files. By default, missing labels fail the merge.",
    )
    args = parser.parse_args()

    annotator_paths = [Path(p) for p in args.annotators]
    for p in annotator_paths:
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    names: List[str] = args.names or [f"annotator_{i + 1}" for i in range(len(annotator_paths))]
    if len(names) != len(annotator_paths):
        print(
            f"ERROR: --names has {len(names)} entries but --annotators has {len(annotator_paths)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {len(annotator_paths)} annotator file(s)...")
    annotator_maps: List[Dict[str, Optional[str]]] = []
    annotator_text_orders: List[List[str]] = []
    for name, path in zip(names, annotator_paths):
        ordered_texts, m = _load_annotator_csv(path, text_col=args.text_col, label_col=args.label_col)
        print(f"  {name}: {len(m)} texts loaded from {path}")
        annotator_text_orders.append(ordered_texts)
        annotator_maps.append(m)

    # Build text universe
    if args.union:
        seen = set()
        texts = []
        for ordered_texts in annotator_text_orders:
            for text in ordered_texts:
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
    else:
        sets = [set(m.keys()) for m in annotator_maps]
        common = sets[0].intersection(*sets[1:])
        texts = [text for text in annotator_text_orders[0] if text in common]

    print(f"  Overlap: {len(texts)} texts ({'union' if args.union else 'intersection'})")
    if not texts:
        print("ERROR: No overlapping texts found. Check that all annotators used the same text column.", file=sys.stderr)
        sys.exit(1)

    matrix = _build_matrix(texts, annotator_maps)
    missing_labels = sum(1 for row in matrix for label in row if not label)
    if missing_labels and not args.allow_incomplete:
        print(
            f"ERROR: Found {missing_labels} missing label(s) across annotator files. "
            "Finish labeling first, or pass --allow_incomplete for a draft audit.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Reconcile
    reconciled = reconcile_annotations(matrix, annotator_names=names)

    # IAA metrics
    report = iaa_report(matrix, annotator_names=names)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["annotator_files"] = [str(p) for p in annotator_paths]
    report["text_overlap_mode"] = "union" if args.union else "intersection"

    # Output paths
    output_csv = Path(args.output) if args.output else BACKEND / "data" / "gold_set_human_reconciled.csv"
    report_json = Path(args.report) if args.report else BACKEND / "results" / "gold_set" / "iaa_report.json"
    report_md = report_json.with_suffix(".md")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    # Write gold set CSV
    fieldnames = ["text"] + [f"label_{n}" for n in names] + ["gold_label", "is_disputed"]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for text, rec in zip(texts, reconciled):
            row: Dict = {"text": text}
            for name in names:
                row[f"label_{name}"] = rec["annotator_labels"].get(name) or ""
            row["gold_label"] = rec["gold_label"] or ""
            row["is_disputed"] = "yes" if rec["is_disputed"] else "no"
            writer.writerow(row)

    # Write report
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(_to_markdown(report, output_csv), encoding="utf-8")

    # Print summary
    m = report["metrics"]
    print()
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT SUMMARY")
    print("=" * 60)
    print(f"  Items analyzed:      {report['n_items']}")
    print(f"  Fully annotated:     {report['fully_annotated_items']}")
    print(f"  Disputed items:      {report['disputed_items']}")
    print()
    print(f"  Percent agreement:   {m['percent_agreement']:.4f}")
    alpha = m['krippendorff_alpha']
    alpha_str = f"{alpha:.4f}" if alpha == alpha else "n/a"
    print(f"  Krippendorff alpha:  {alpha_str}  ({report['interpretation']})")
    fleiss = m['fleiss_kappa']
    fleiss_str = f"{fleiss:.4f}" if fleiss == fleiss else "n/a"
    print(f"  Fleiss kappa:        {fleiss_str}")
    for pair, kappa in m["cohen_kappa_pairwise"].items():
        kappa_str = f"{kappa:.4f}" if kappa == kappa else "n/a"
        print(f"  Cohen kappa ({pair}): {kappa_str}")
    print()
    print(f"  Gold set CSV:        {output_csv}")
    print(f"  IAA report JSON:     {report_json}")
    print(f"  IAA report Markdown: {report_md}")


if __name__ == "__main__":
    main()
