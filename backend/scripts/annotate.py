#!/usr/bin/env python3
"""
Interactive CLI annotator for the gold set.

Presents each unlabeled comment one at a time and records your label.
Progress is saved after every annotation so you can stop and resume.

Usage
-----
# Start fresh annotation session
python scripts/annotate.py \\
    --input data/gold_set_template.csv \\
    --output data/gold_set_annotator_1.csv

# Resume an interrupted session (skips already-labeled rows)
python scripts/annotate.py \\
    --input data/gold_set_template.csv \\
    --output data/gold_set_annotator_1.csv \\
    --resume

# Show a context window of N chars per comment (default: 500)
python scripts/annotate.py --input data/gold_set_template.csv --max_chars 300

Controls
--------
  p / 1 / + : Positive
  n / 0 / - : Negative
  u / 2     : Neutral (unsure/mixed)
  s         : Skip (leave blank - will show as unannotated)
  q         : Quit and save progress
  ?         : Show last 5 decisions for context
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABEL_MAP = {
    "p": "Positive", "1": "Positive", "+": "Positive",
    "n": "Negative", "0": "Negative", "-": "Negative",
    "u": "Neutral",  "2": "Neutral",
}
LABELS = ["Positive", "Neutral", "Negative"]


def _clear() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


def _load_existing(path: Path, text_col: str, label_col: str) -> Dict[str, str]:
    """Load text->label for already-annotated rows."""
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and text_col in reader.fieldnames and label_col in reader.fieldnames:
            for row in reader:
                text = str(row[text_col]).strip()
                label = str(row.get(label_col, "")).strip()
                if text and label:
                    result[text] = label
    return result


def _load_template(path: Path, text_col: str) -> List[str]:
    """Load ordered list of texts from template CSV."""
    texts: List[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found in {path}")
        for row in reader:
            text = str(row[text_col]).strip()
            if text:
                texts.append(text)
    return texts


def _write_output(
    path: Path,
    texts: List[str],
    labels: Dict[str, str],
    text_col: str,
    label_col: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[text_col, label_col])
        writer.writeheader()
        for text in texts:
            writer.writerow({text_col: text, label_col: labels.get(text, "")})


def _format_comment(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"  [...{len(text) - max_chars} more chars]"


def _print_header(total: int, skipped: int, done: int) -> None:
    pct = done / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"  [{bar}] {done}/{total} ({pct:.0f}%)  skipped={skipped}")
    print()


def _prompt() -> Tuple[Optional[str], bool]:
    """
    Returns (label, quit_flag).
    label is None if the user skips.
    """
    while True:
        raw = input("  Label [p=Pos / n=Neg / u=Neutral / s=skip / q=quit / ?=history]: ").strip().lower()
        if raw in LABEL_MAP:
            return LABEL_MAP[raw], False
        if raw == "s":
            return None, False
        if raw == "q":
            return None, True
        if raw == "?":
            return "?", False
        print("  Invalid input. Use p/n/u/s/q.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CLI annotator for the gold set.")
    parser.add_argument("--input", required=True, help="Template CSV with text column.")
    parser.add_argument("--output", required=True, help="Output CSV to write annotations into.")
    parser.add_argument("--text_col", default="text", help="Name of text column (default: text).")
    parser.add_argument("--label_col", default="label", help="Name of label column (default: label).")
    parser.add_argument("--resume", action="store_true", help="Skip already-labeled texts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument("--max_chars", type=int, default=500, help="Max chars to display per comment.")
    parser.add_argument("--no_clear", action="store_true", help="Disable screen clearing between items.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if output_path.exists() and not args.resume and not args.overwrite:
        print(
            f"ERROR: Output already exists: {output_path}\n"
            "Use --resume to continue it, or --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.max_chars < 1:
        print("ERROR: --max_chars must be at least 1.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading template: {input_path}")
    texts = _load_template(input_path, args.text_col)
    print(f"  {len(texts)} items loaded.")

    labels: Dict[str, str] = {}
    if args.resume and output_path.exists():
        labels = _load_existing(output_path, args.text_col, args.label_col)
        print(f"  Resuming: {len(labels)} items already annotated.")

    # Filter to unannotated items if resuming
    todo = [t for t in texts if t not in labels] if args.resume else list(texts)
    done_before = len(labels)
    total = len(texts)

    history: List[Tuple[str, str]] = []  # (text_snippet, label)
    skipped = 0
    annotated_this_session = 0

    print(f"\n  Remaining: {len(todo)} items to annotate.")
    print("  Press Enter to start...")
    input()

    for i, text in enumerate(todo):
        if not args.no_clear:
            _clear()

        current_done = done_before + annotated_this_session
        print(f"\n{'-' * 70}")
        _print_header(total, skipped, current_done)

        print(f"  Comment #{i + 1} of {len(todo)}:")
        print(f"{'-' * 70}")
        print()
        formatted = _format_comment(text, args.max_chars)
        # Indent for readability
        for line in formatted.splitlines():
            print(f"    {line}")
        print()

        label, quit_flag = _prompt()

        while label == "?":
            print()
            if history:
                print("  Recent decisions:")
                for snip, lbl in history[-5:]:
                    print(f"    [{lbl}] {snip}")
            else:
                print("  No decisions yet.")
            print()
            # Re-prompt for the same item
            label, quit_flag = _prompt()

        if quit_flag:
            print("\n  Saving progress and quitting...")
            break

        if label is None:
            skipped += 1
            print("  Skipped.")
        else:
            labels[text] = label
            annotated_this_session += 1
            history.append((text[:60] + "..." if len(text) > 60 else text, label))
            print(f"  Labeled: {label}")

        # Save after every annotation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_output(output_path, texts, labels, args.text_col, args.label_col)

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(output_path, texts, labels, args.text_col, args.label_col)

    labeled_count = sum(1 for t in texts if labels.get(t))
    print()
    print("=" * 70)
    print("  ANNOTATION SESSION COMPLETE")
    print("=" * 70)
    print(f"  Labeled this session:  {annotated_this_session}")
    print(f"  Skipped this session:  {skipped}")
    print(f"  Total labeled so far:  {labeled_count} / {total}")
    print(f"  Remaining:             {total - labeled_count}")
    print(f"  Output saved to:       {output_path}")
    print()
    if labeled_count < total:
        print(f"  Resume with: python scripts/annotate.py --input {args.input} --output {args.output} --resume")
    print()


if __name__ == "__main__":
    main()
