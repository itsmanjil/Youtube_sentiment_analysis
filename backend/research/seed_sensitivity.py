"""
Seed sensitivity analysis for the TF-IDF + LogReg baseline.

Why
---
A thesis examiner can reasonably ask: "Is the 0.694 macro-F1 a lucky draw, or
does it hold across different random seeds?" This script re-runs the baseline
pipeline with multiple seeds over the same subsample of the training data and
reports mean ± std of accuracy, F1, and Cohen's kappa.

This is much lighter-weight than repeating the full 10-fold CV at three seeds.
The goal is to demonstrate that headline numbers are stable at the 10^-3 level,
not to produce new CV confidence intervals.

Outputs
-------
  - results/seed_sensitivity.json
  - results/seed_sensitivity.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


BACKEND = Path(
    "/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend"
)
TRAIN = BACKEND / "data" / "train.csv"
OUT_JSON = BACKEND / "results" / "seed_sensitivity.json"
OUT_MD = BACKEND / "results" / "seed_sensitivity.md"

LABELS = ("Negative", "Neutral", "Positive")
SEEDS = (0, 7, 13, 42, 123)
N_PER_CLASS = 8000  # 24,000 balanced samples per seed


def load_balanced(seed: int) -> pd.DataFrame:
    df = pd.read_csv(TRAIN, usecols=["text", "label"])
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)]
    parts = []
    for lbl in LABELS:
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(n=min(N_PER_CLASS, len(sub)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_seed(seed: int) -> dict:
    df = load_balanced(seed)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].to_numpy(),
        df["label"].to_numpy(),
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000, min_df=2, max_df=0.95,
        sublinear_tf=True,
    )
    Xt = vec.fit_transform(X_train)
    Xv = vec.transform(X_val)
    model = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=seed
    )
    model.fit(Xt, y_train)
    pred = model.predict(Xv)
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1_macro": float(
            f1_score(y_val, pred, average="macro", labels=list(LABELS))
        ),
        "cohen_kappa": float(cohen_kappa_score(y_val, pred)),
        "vocab_size": int(Xt.shape[1]),
    }


def main() -> None:
    records = []
    for s in SEEDS:
        print(f"seed={s} ...", flush=True)
        r = run_seed(s)
        print(
            f"  acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}  "
            f"kappa={r['cohen_kappa']:.4f}"
        )
        records.append(r)

    def stats(key: str) -> dict:
        vals = np.array([r[key] for r in records])
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
        }

    agg = {
        "accuracy": stats("accuracy"),
        "f1_macro": stats("f1_macro"),
        "cohen_kappa": stats("cohen_kappa"),
    }

    report = {
        "metadata": {
            "data": str(TRAIN),
            "n_per_class": N_PER_CLASS,
            "seeds": list(SEEDS),
            "pipeline": "TF-IDF(1,2, 50k) + LogReg(balanced)",
            "note": (
                "Seed sensitivity analysis. Each seed re-draws the stratified "
                "subsample AND re-splits train/val AND re-seeds the model, so "
                "this captures the full variance stack, not just model init."
            ),
        },
        "per_seed": records,
        "aggregate": agg,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    lines: list[str] = []
    lines.append("# Seed Sensitivity Analysis\n\n")
    lines.append(
        f"Pipeline: TF-IDF(1,2, 50k) + LogReg(balanced)  |  "
        f"{N_PER_CLASS:,} per class, 80/20 train/val  |  seeds: {list(SEEDS)}\n\n"
    )
    lines.append(
        "> Each seed re-samples the training data, re-splits, and re-seeds the "
        "model. This captures the full pipeline variance, not just model "
        "initialisation.\n\n"
    )
    lines.append(
        "| Seed | n_train | n_val | |V| | Accuracy | F1-macro | Cohen κ |\n"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in records:
        lines.append(
            f"| {r['seed']} | {r['n_train']:,} | {r['n_val']:,} | "
            f"{r['vocab_size']:,} | {r['accuracy']:.4f} | "
            f"{r['f1_macro']:.4f} | {r['cohen_kappa']:.4f} |\n"
        )

    lines.append("\n## Aggregate over seeds\n\n")
    lines.append("| Metric | Mean | Std | Range |\n|---|---:|---:|---:|\n")
    for m in ("accuracy", "f1_macro", "cohen_kappa"):
        s = agg[m]
        lines.append(
            f"| {m} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f}–{s['max']:.4f} ({s['range']:.4f}) |\n"
        )

    f1_range = agg["f1_macro"]["range"]
    lines.append("\n## Interpretation\n\n")
    if f1_range < 0.01:
        verdict = (
            "The F1 range across seeds is below 0.01, i.e. the headline macro-F1 "
            "is stable to at least two decimal places. The 0.694 ± ε figure "
            "reported in the main thesis tables is **not a lucky seed** and no "
            "additional caveat is needed."
        )
    elif f1_range < 0.02:
        verdict = (
            "The F1 range across seeds is under 0.02, which is small relative to "
            "the differences between methods reported in the main tables. The "
            "thesis should still note the seed-driven variance when discussing "
            "very close comparisons."
        )
    else:
        verdict = (
            "The F1 range across seeds exceeds 0.02, which is large enough to "
            "affect fine-grained comparisons between CI methods. The thesis "
            "should report results as mean ± std over multiple seeds rather "
            "than single-seed point estimates."
        )
    lines.append(verdict + "\n")

    with open(OUT_MD, "w") as f:
        f.writelines(lines)

    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
