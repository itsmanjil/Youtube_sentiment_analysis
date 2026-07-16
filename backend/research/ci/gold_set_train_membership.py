"""
Gold Set Train-Membership Audit

Checks whether gold-set items (data/gold_set_silver_labeled.csv) also appear
in the training split, after applying the same preprocessing used to build
train/val/test (emoji conversion + whitespace normalisation). Writes a
per-item membership CSV and a held-out-only subset for use by
gold_set_evaluation_holdout.py.

This exists because the gold set was originally sampled from train.csv
(see README_THESIS.md), not from the held-out test split, so a naive
"independent human evaluation" claim needs this check.
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.preprocessing.youtube import YouTubePreprocessor

SILVER_CSV = BACKEND / "data" / "gold_set_silver_labeled.csv"
OUT_CSV = BACKEND / "data" / "gold_set_split_membership.csv"

pre = YouTubePreprocessor()


def normalize(text: str) -> str:
    out = pre.preprocess_youtube_comment(str(text), emoji_mode="convert")
    if isinstance(out, tuple):
        out = out[0]
    return re.sub(r"\s+", " ", str(out)).strip()


def build_lookup(csv_path: Path) -> set:
    texts = set()
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=200_000):
        texts.update(chunk["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
    return texts


def main() -> None:
    with open(SILVER_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} gold-set rows.")
    normed = [normalize(r["text"]) for r in rows]

    print("Scanning train/val/test splits for membership...")
    train_texts = build_lookup(BACKEND / "data" / "train.csv")
    val_texts = build_lookup(BACKEND / "data" / "val.csv")
    test_texts = build_lookup(BACKEND / "data" / "test.csv")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "in_train", "in_val", "in_test", "split_membership"])
        counts = {"train": 0, "val": 0, "test": 0, "none": 0}
        for r, n in zip(rows, normed):
            in_train = n in train_texts
            in_val = n in val_texts
            in_test = n in test_texts
            if in_train:
                membership = "train"
            elif in_val:
                membership = "val"
            elif in_test:
                membership = "test"
            else:
                membership = "none"
            counts[membership] += 1
            writer.writerow([r["text"], in_train, in_val, in_test, membership])

    print(f"Wrote {OUT_CSV}")
    print(f"  in_train: {counts['train']}")
    print(f"  in_val:   {counts['val']}")
    print(f"  in_test:  {counts['test']}")
    print(f"  no match (post-processing filtered / genuinely new): {counts['none']}")
    print(f"  held-out-only (val+test+none) count: {counts['val'] + counts['test'] + counts['none']}")


if __name__ == "__main__":
    main()
