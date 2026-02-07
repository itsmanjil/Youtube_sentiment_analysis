import os
import json
from datetime import datetime

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def main():
    source_uri = "hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv"

    # Load HF dataset (requires huggingface_hub + fsspec)
    df = pd.read_csv(source_uri)
    raw_rows = len(df)
    raw_columns = list(df.columns)

    # Resolve column names robustly
    text_col = "CommentText" if "CommentText" in df.columns else "comment_text"
    label_col = "Sentiment" if "Sentiment" in df.columns else "label"
    group_col = "VideoID" if "VideoID" in df.columns else None

    # Normalize schema
    df = df.dropna(subset=[text_col, label_col])
    after_dropna = len(df)
    # Normalize text early so duplicate detection is meaningful.
    df["text"] = (
        df[text_col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["label"] = df[label_col].astype(str).str.title()

    # Keep valid labels only
    valid = {"Positive", "Neutral", "Negative"}
    df = df[df["label"].isin(valid)]
    after_valid_labels = len(df)

    # Remove empty texts after normalization
    df = df[df["text"].astype(bool)]
    after_nonempty = len(df)

    # Drop texts with conflicting labels (prevents label noise + leakage across splits).
    label_nunique = df.groupby("text")["label"].nunique()
    conflicting_texts = label_nunique[label_nunique > 1].index
    conflicting_count = int(len(conflicting_texts))
    if len(conflicting_texts) > 0:
        print(f"Dropping {len(conflicting_texts)} texts with conflicting labels")
        df = df[~df["text"].isin(conflicting_texts)]
    after_conflicts = len(df)

    # Deduplicate by text so splits are disjoint by construction.
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    removed_duplicates = int(removed)
    if removed:
        print(f"Removed {removed} duplicate texts before splitting")

    # Optional: filter to English if a language column exists
    if "Language" in df.columns:
        df = df[df["Language"].astype(str).str.lower().isin(["english", "en"])]
    after_language = len(df)

    # Split (grouped by VideoID if available)
    if group_col:
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        gss2 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, val_idx = next(gss2.split(train_df, groups=train_df[group_col]))
        final_train, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )
        final_train, val_df = train_test_split(
            train_df, test_size=0.2, stratify=train_df["label"], random_state=42
        )

    # Export
    os.makedirs("backend/data", exist_ok=True)
    final_train[["text", "label"]].to_csv("backend/data/train.csv", index=False)
    val_df[["text", "label"]].to_csv("backend/data/val.csv", index=False)
    test_df[["text", "label"]].to_csv("backend/data/test.csv", index=False)

    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_uri": source_uri,
        "raw_rows": int(raw_rows),
        "raw_columns": raw_columns,
        "schema": {
            "text_col": text_col,
            "label_col": label_col,
            "group_col": group_col,
        },
        "filters": {
            "dropna_rows": int(raw_rows - after_dropna),
            "invalid_label_rows": int(after_dropna - after_valid_labels),
            "empty_text_rows": int(after_valid_labels - after_nonempty),
            "conflicting_texts_dropped": conflicting_count,
            "duplicate_text_rows_removed": removed_duplicates,
            "language_filter_applied": bool("Language" in raw_columns),
            "rows_after_language_filter": int(after_language),
        },
        "split": {
            "strategy": "group" if group_col else "stratified",
            "random_state": 42,
            "test_size": 0.2,
            "val_size": 0.2,
            "train_rows": int(len(final_train)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "label_distribution": {
                "train": final_train["label"].value_counts().to_dict(),
                "val": val_df["label"].value_counts().to_dict(),
                "test": test_df["label"].value_counts().to_dict(),
            },
        },
    }
    with open("backend/data/split_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved: backend/data/train.csv, val.csv, test.csv")
    print("Saved: backend/data/split_metadata.json")


if __name__ == "__main__":
    main()
