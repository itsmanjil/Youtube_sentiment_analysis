"""
Knob-level ablation for the classical preprocessing module.

Why
---
The existing `thesis_preprocess_ablation.md` compares dataset-level
preprocessing variants (raw / youtube_clean / youtube_filtered). It does NOT
isolate the knobs exposed by `src/preprocessing/classical.py`:

  - expand_negation_contractions  ("dont" -> "do not")
  - negation_tag                  ("not good" -> "not_good")
  - remove_stopwords              (with negator preservation)

This script runs a 2^3 = 8 configuration ablation on a stratified subsample
of the training data with a held-out validation split, using the same
TF-IDF(1,2) + LogReg pipeline used elsewhere in the thesis. It is fast by
design (≈20k samples) so it can be re-run while tuning.

Outputs
-------
  - results/preprocessing_knob_ablation.json
  - results/preprocessing_knob_ablation.md
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import sys

BACKEND = Path("/sessions/funny-wizardly-einstein/mnt/Youtube_sentiment_analysis/backend")
sys.path.insert(0, str(BACKEND))

from src.preprocessing import (  # noqa: E402
    ClassicalPreprocessConfig,
    preprocess_classical_texts,
)


TRAIN = BACKEND / "data" / "train.csv"
OUT_JSON = BACKEND / "results" / "preprocessing_knob_ablation.json"
OUT_MD = BACKEND / "results" / "preprocessing_knob_ablation.md"

LABELS = ("Negative", "Neutral", "Positive")

N_PER_CLASS = 6000  # 18,000 total — fast but large enough for stable F1
SEED = 42


def load_balanced(path: Path, per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["text", "label"])
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)]
    parts = []
    for lbl in LABELS:
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_config(
    X_train: list[str],
    X_val: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    expand_neg: bool,
    neg_tag: bool,
    rm_stop: bool,
) -> dict:
    cfg = ClassicalPreprocessConfig(
        expand_negation_contractions=expand_neg,
        negation_tag=neg_tag,
        remove_stopwords=rm_stop,
    )
    Xt = preprocess_classical_texts(X_train, config=cfg)
    Xv = preprocess_classical_texts(X_val, config=cfg)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    Xt_vec = vec.fit_transform(Xt)
    Xv_vec = vec.transform(Xv)

    model = LogisticRegression(
        max_iter=1500,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(Xt_vec, y_train)
    pred = model.predict(Xv_vec)

    return {
        "expand_negation_contractions": expand_neg,
        "negation_tag": neg_tag,
        "remove_stopwords": rm_stop,
        "vocab_size": int(Xt_vec.shape[1]),
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1_macro": float(
            f1_score(y_val, pred, average="macro", labels=list(LABELS))
        ),
    }


def main() -> None:
    print(f"Loading {TRAIN} (balanced {N_PER_CLASS}/class)...")
    df = load_balanced(TRAIN, N_PER_CLASS, SEED)
    print(f"  n = {len(df):,}")

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    records: list[dict] = []
    configs = list(itertools.product([False, True], repeat=3))
    for expand_neg, neg_tag, rm_stop in configs:
        print(
            f"  expand={expand_neg} neg_tag={neg_tag} rm_stop={rm_stop} ...",
            flush=True,
        )
        r = run_config(X_train, X_val, y_train, y_val, expand_neg, neg_tag, rm_stop)
        print(
            f"    acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}  "
            f"|V|={r['vocab_size']:,}"
        )
        records.append(r)

    # Identify baseline (all False) and best config
    baseline = next(
        r
        for r in records
        if not r["expand_negation_contractions"]
        and not r["negation_tag"]
        and not r["remove_stopwords"]
    )
    best = max(records, key=lambda r: r["f1_macro"])

    report = {
        "metadata": {
            "data": str(TRAIN),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "seed": SEED,
            "pipeline": "TF-IDF(1,2, max_feat=30k) + LogReg(C=1.0, balanced)",
            "note": (
                "Knob-level ablation of ClassicalPreprocessConfig. "
                "Complements thesis_preprocess_ablation.md, which compares "
                "dataset-level cleaning stages rather than preprocessing knobs."
            ),
        },
        "baseline": baseline,
        "best": best,
        "delta_f1_best_vs_baseline": best["f1_macro"] - baseline["f1_macro"],
        "records": records,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    lines: list[str] = []
    lines.append("# Preprocessing Knob Ablation\n\n")
    lines.append(
        f"Data: balanced subsample of `{TRAIN.name}`  "
        f"(train={len(X_train):,}, val={len(X_val):,})  "
        f"|  seed={SEED}  |  pipeline=TF-IDF(1,2) + LogReg(balanced)\n\n"
    )
    lines.append(
        "> Full 2³ ablation of `ClassicalPreprocessConfig` knobs. "
        "Each row trains an independent TF-IDF + LogReg pipeline. "
        "This complements `thesis_preprocess_ablation.md`, which ablates "
        "dataset-level cleaning stages, not preprocessing knobs.\n\n"
    )
    lines.append(
        "| Expand neg | Negation tag | Remove stopw | |V| | Accuracy | F1-macro | ΔF1 vs baseline |\n"
    )
    lines.append(
        "|:---:|:---:|:---:|---:|---:|---:|---:|\n"
    )
    for r in records:
        delta = r["f1_macro"] - baseline["f1_macro"]
        marker = " **←** " if r is best else ""
        lines.append(
            f"| {'✓' if r['expand_negation_contractions'] else '·'} "
            f"| {'✓' if r['negation_tag'] else '·'} "
            f"| {'✓' if r['remove_stopwords'] else '·'} "
            f"| {r['vocab_size']:,} "
            f"| {r['accuracy']:.4f} "
            f"| {r['f1_macro']:.4f}{marker}"
            f"| {delta:+.4f} |\n"
        )

    lines.append("\n## Interpretation\n\n")
    lines.append(
        f"- **Baseline** (all knobs off): F1 = {baseline['f1_macro']:.4f}.\n"
        f"- **Best configuration**: "
        f"expand={'✓' if best['expand_negation_contractions'] else '·'}, "
        f"neg_tag={'✓' if best['negation_tag'] else '·'}, "
        f"rm_stop={'✓' if best['remove_stopwords'] else '·'}, "
        f"F1 = {best['f1_macro']:.4f} "
        f"(**{best['f1_macro'] - baseline['f1_macro']:+.4f}** over baseline).\n"
    )

    # Single-knob deltas (main effects)
    def single_knob(flag: str) -> float:
        on = [r["f1_macro"] for r in records if r[flag]]
        off = [r["f1_macro"] for r in records if not r[flag]]
        return float(np.mean(on) - np.mean(off))

    lines.append("\n### Main effects (mean F1 with knob on − mean F1 with knob off)\n\n")
    lines.append(
        f"- expand_negation_contractions: **{single_knob('expand_negation_contractions'):+.4f}**\n"
        f"- negation_tag: **{single_knob('negation_tag'):+.4f}**\n"
        f"- remove_stopwords: **{single_knob('remove_stopwords'):+.4f}**\n"
    )

    lines.append(
        "\nIf the improvements are ≤ 0.005 F1 the knobs should be described in "
        "the thesis as *behaviour-preserving* rather than accuracy-enhancing — "
        "they stabilise preprocessing across training and inference (no "
        "train/inference skew) at essentially zero cost, which is the real "
        "contribution documented in `preprocessing_consistency_audit.md`.\n"
    )

    with open(OUT_MD, "w") as f:
        f.writelines(lines)

    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
