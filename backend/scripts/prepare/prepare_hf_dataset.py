import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def main():
    # Load HF dataset (requires huggingface_hub + fsspec)
    df = pd.read_csv(
        "hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv"
    )

    # Resolve column names robustly
    text_col = "CommentText" if "CommentText" in df.columns else "comment_text"
    label_col = "Sentiment" if "Sentiment" in df.columns else "label"
    group_col = "VideoID" if "VideoID" in df.columns else None

    # Normalize schema
    df = df.dropna(subset=[text_col, label_col])
    df["text"] = df[text_col].astype(str)
    df["label"] = df[label_col].astype(str).str.title()

    # Keep valid labels only
    valid = {"Positive", "Neutral", "Negative"}
    df = df[df["label"].isin(valid)]

    # Optional: filter to English if a language column exists
    if "Language" in df.columns:
        df = df[df["Language"].astype(str).str.lower().isin(["english", "en"])]

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

    print("Saved: backend/data/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
