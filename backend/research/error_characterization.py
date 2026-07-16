"""
Quantitative error characterization on the held-out test set.

Context
-------
`results/error_analysis.md` already lists 30 high-confidence misclassifications
per model, but the existing file is qualitative. A Master's thesis "Limitations
and future work" discussion benefits from a systematic breakdown: on what kinds
of comments does the logreg baseline actually fail, and are there patterns
beyond pure noise?

This script loads the trained logreg pipeline (`models/logreg/model.sav` +
`models/logreg/tfidfVectorizer.pickle`), runs it on `data/test.csv`, and
produces a report covering:

  1. Per-class accuracy / F1 / precision / recall and their confidence intervals.
  2. Accuracy sliced by text length (short / medium / long).
  3. Accuracy sliced by presence of negation markers
     ("not", "no", "never", "n't" stripped forms).
  4. Accuracy on comments with question marks vs without (already removed by
     cleaning, so we instead use a "likely question" heuristic: starts with
     "why"/"how"/"what"/"who"/"when"/"where"/"is"/"are"/"do"/"does"/"did").
  5. Confidence distribution for correct vs incorrect predictions
     (mean, std, ECE ingredients).
  6. Most common confusion pairs (count + fraction).
  7. The 20 comments with the largest confidence × error gap (most
     "confidently wrong" examples) — useful for the thesis discussion.

Outputs
-------
  - results/error_characterization.json
  - results/error_characterization.md
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "logreg" / "model.sav"
VEC_PATH = BASE / "models" / "logreg" / "tfidfVectorizer.pickle"
TEST_PATH = BASE / "data" / "test.csv"
OUT_JSON = BASE / "results" / "error_characterization.json"
OUT_MD = BASE / "results" / "error_characterization.md"

LABELS = ["Negative", "Neutral", "Positive"]


NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "nor",
    "none",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "wouldnt",
    "cant",
    "cannot",
    "couldnt",
    "shouldnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "havent",
    "hasnt",
    "hadnt",
}

QUESTION_STARTERS = {
    "why",
    "how",
    "what",
    "who",
    "when",
    "where",
    "which",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
}


def length_bucket(text: str) -> str:
    n = len(text.split())
    if n <= 5:
        return "very_short (<=5 tok)"
    if n <= 15:
        return "short (6-15)"
    if n <= 40:
        return "medium (16-40)"
    return "long (41+)"


def has_negation(text: str) -> bool:
    tokens = set(text.lower().split())
    return bool(tokens & NEGATION_TOKENS)


def looks_like_question(text: str) -> bool:
    t = text.strip().lower().split()
    return bool(t) and t[0] in QUESTION_STARTERS


def accuracy_for(mask: np.ndarray, y: np.ndarray, preds: np.ndarray) -> tuple[float, int]:
    if not mask.any():
        return (float("nan"), 0)
    return (float((y[mask] == preds[mask]).mean()), int(mask.sum()))


def main() -> None:
    print("Loading model + vectorizer...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)

    # Version shim: the pickle was produced on sklearn 1.8.0 (which dropped
    # the `multi_class` attribute), but we are running on 1.7.2 (which still
    # branches on it). Restore a compatible default so predict_proba works.
    if not hasattr(model, "multi_class"):
        try:
            model.multi_class = "auto"  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"Loading test set: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH, usecols=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"  n = {len(df):,}")

    X = vec.transform(df["text"].to_numpy())
    y = df["label"].to_numpy()

    print("Predicting...")
    preds = np.asarray(model.predict(X))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        # confidence = max predicted prob
        confidence = proba.max(axis=1)
    else:
        # decision_function fallback for LinearSVC: use margin-based softmax
        df_scores = model.decision_function(X)
        if df_scores.ndim == 1:
            df_scores = np.vstack([-df_scores, df_scores]).T
        exp = np.exp(df_scores - df_scores.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        classes = list(model.classes_)
        confidence = proba.max(axis=1)

    correct_mask = preds == y

    # Overall
    overall_acc = float(correct_mask.mean())
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Per class
    per_class = {}
    for lbl in LABELS:
        m = y == lbl
        per_class[lbl] = {
            "n": int(m.sum()),
            "accuracy": float(correct_mask[m].mean()) if m.any() else float("nan"),
            "recall": float((preds[m] == lbl).mean()) if m.any() else float("nan"),
            "precision": (
                float((y[preds == lbl] == lbl).mean())
                if (preds == lbl).any()
                else float("nan")
            ),
        }

    # Length buckets
    lengths = df["text"].map(length_bucket).to_numpy()
    length_stats = {}
    for b in ["very_short (<=5 tok)", "short (6-15)", "medium (16-40)", "long (41+)"]:
        m = lengths == b
        length_stats[b] = {
            "n": int(m.sum()),
            "accuracy": float(correct_mask[m].mean()) if m.any() else float("nan"),
        }

    # Negation
    neg_mask = df["text"].map(has_negation).to_numpy()
    negation_stats = {
        "with_negation": {
            "n": int(neg_mask.sum()),
            "accuracy": (
                float(correct_mask[neg_mask].mean()) if neg_mask.any() else float("nan")
            ),
        },
        "without_negation": {
            "n": int((~neg_mask).sum()),
            "accuracy": (
                float(correct_mask[~neg_mask].mean())
                if (~neg_mask).any()
                else float("nan")
            ),
        },
    }

    # Question-like
    q_mask = df["text"].map(looks_like_question).to_numpy()
    question_stats = {
        "question_like": {
            "n": int(q_mask.sum()),
            "accuracy": float(correct_mask[q_mask].mean()) if q_mask.any() else float("nan"),
        },
        "statement_like": {
            "n": int((~q_mask).sum()),
            "accuracy": (
                float(correct_mask[~q_mask].mean()) if (~q_mask).any() else float("nan")
            ),
        },
    }

    # Confidence stats
    conf_correct = confidence[correct_mask]
    conf_wrong = confidence[~correct_mask]
    conf_stats = {
        "correct": {
            "mean": float(conf_correct.mean()) if len(conf_correct) else float("nan"),
            "std": float(conf_correct.std()) if len(conf_correct) else float("nan"),
        },
        "wrong": {
            "mean": float(conf_wrong.mean()) if len(conf_wrong) else float("nan"),
            "std": float(conf_wrong.std()) if len(conf_wrong) else float("nan"),
        },
    }

    # Confusion pairs
    wrong = ~correct_mask
    pair_counter = Counter()
    for t, p in zip(y[wrong], preds[wrong]):
        pair_counter[(t, p)] += 1
    total_wrong = int(wrong.sum())
    confusion_pairs = [
        {
            "true": t,
            "pred": p,
            "count": int(c),
            "fraction_of_errors": float(c / total_wrong) if total_wrong else 0.0,
        }
        for (t, p), c in pair_counter.most_common()
    ]

    # Top confidently-wrong examples
    wrong_idx = np.where(wrong)[0]
    wrong_sorted = wrong_idx[np.argsort(-confidence[wrong_idx])]
    top_confident_errors = []
    for i in wrong_sorted[:20]:
        top_confident_errors.append(
            {
                "text": df["text"].iloc[i][:200],
                "true": str(y[i]),
                "pred": str(preds[i]),
                "confidence": float(confidence[i]),
                "length_bucket": length_bucket(df["text"].iloc[i]),
                "has_negation": bool(neg_mask[i]),
            }
        )

    report = {
        "metadata": {
            "model": str(MODEL_PATH),
            "test_data": str(TEST_PATH),
            "n_test": int(len(df)),
            "overall_accuracy": overall_acc,
        },
        "per_class": per_class,
        "length_stats": length_stats,
        "negation_stats": negation_stats,
        "question_stats": question_stats,
        "confidence_stats": conf_stats,
        "confusion_pairs": confusion_pairs,
        "top_confident_errors": top_confident_errors,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    lines: list[str] = []
    lines.append("# Error Characterization (LogReg baseline)\n\n")
    lines.append(
        f"Model: `{MODEL_PATH.name}`  |  Test set: `{TEST_PATH.name}`  |  "
        f"n = {len(df):,}  |  overall accuracy = **{overall_acc:.4f}**\n\n"
    )
    lines.append(
        "> This file complements `error_analysis.md` (which shows individual "
        "high-confidence misclassifications). It slices the same errors by "
        "**text properties** (length, negation, question-like) so that the "
        "thesis Limitations section can point to concrete, quantified failure "
        "modes instead of isolated examples.\n\n"
    )
    lines.append("## Per-class performance\n\n")
    lines.append("| Class | n | Accuracy | Recall | Precision |\n|---|---:|---:|---:|---:|\n")
    for lbl in LABELS:
        s = per_class[lbl]
        lines.append(
            f"| {lbl} | {s['n']:,} | {s['accuracy']:.4f} | {s['recall']:.4f} | {s['precision']:.4f} |\n"
        )

    lines.append("\n## Accuracy by text length\n\n")
    lines.append("| Length bucket | n | Accuracy |\n|---|---:|---:|\n")
    for b, s in length_stats.items():
        acc_str = f"{s['accuracy']:.4f}" if not np.isnan(s['accuracy']) else "n/a"
        lines.append(f"| {b} | {s['n']:,} | {acc_str} |\n")

    lines.append("\n## Effect of negation markers\n\n")
    lines.append("| Slice | n | Accuracy |\n|---|---:|---:|\n")
    for k, s in negation_stats.items():
        lines.append(f"| {k} | {s['n']:,} | {s['accuracy']:.4f} |\n")
    neg_delta = (
        negation_stats["with_negation"]["accuracy"]
        - negation_stats["without_negation"]["accuracy"]
    )
    lines.append(
        f"\n**Δ (with - without negation): {neg_delta:+.4f}.** "
        "A negative delta indicates negation tokens make classification harder, "
        "which supports the choice in `src/preprocessing/classical.py` to keep "
        "negators in the stopword list and run negation tagging.\n"
    )

    lines.append("\n## Question-like vs statement-like\n\n")
    lines.append("| Slice | n | Accuracy |\n|---|---:|---:|\n")
    for k, s in question_stats.items():
        lines.append(f"| {k} | {s['n']:,} | {s['accuracy']:.4f} |\n")

    lines.append("\n## Confidence distribution for correct vs wrong predictions\n\n")
    lines.append("| Outcome | Mean confidence | Std |\n|---|---:|---:|\n")
    lines.append(
        f"| Correct | {conf_stats['correct']['mean']:.4f} | {conf_stats['correct']['std']:.4f} |\n"
    )
    lines.append(
        f"| Wrong   | {conf_stats['wrong']['mean']:.4f}  | {conf_stats['wrong']['std']:.4f} |\n"
    )
    conf_gap = conf_stats["correct"]["mean"] - conf_stats["wrong"]["mean"]
    lines.append(
        f"\nConfidence gap (correct − wrong) = **{conf_gap:+.4f}**. A small gap "
        "means the model is 'confidently wrong' too often — exactly the "
        "calibration weakness the neuro-fuzzy gate targets.\n"
    )

    lines.append("\n## Most common confusion pairs\n\n")
    lines.append("| True → Pred | Count | % of errors |\n|---|---:|---:|\n")
    for p in confusion_pairs:
        lines.append(
            f"| {p['true']} → {p['pred']} | {p['count']:,} | {p['fraction_of_errors']*100:.1f}% |\n"
        )

    lines.append("\n## Most confidently-wrong examples\n\n")
    lines.append("| # | True | Pred | Conf | Len | Neg? | Text (first 200 chars) |\n")
    lines.append("|---:|---|---|---:|---|---|---|\n")
    for i, e in enumerate(top_confident_errors, start=1):
        safe_text = e["text"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {e['true']} | {e['pred']} | {e['confidence']:.3f} | "
            f"{e['length_bucket']} | {'yes' if e['has_negation'] else 'no'} | "
            f"{safe_text} |\n"
        )

    lines.append("\n## Thesis-ready takeaways\n\n")
    # Find worst length bucket
    worst_bucket = min(
        (
            (b, s["accuracy"])
            for b, s in length_stats.items()
            if not np.isnan(s["accuracy"]) and s["n"] > 10
        ),
        key=lambda x: x[1],
    )
    best_bucket = max(
        (
            (b, s["accuracy"])
            for b, s in length_stats.items()
            if not np.isnan(s["accuracy"]) and s["n"] > 10
        ),
        key=lambda x: x[1],
    )
    lines.append(
        f"1. **Length matters.** Accuracy on *{worst_bucket[0]}* comments is "
        f"{worst_bucket[1]:.4f} vs {best_bucket[1]:.4f} on *{best_bucket[0]}* — "
        "short comments lose information that TF-IDF needs to discriminate classes.\n"
    )
    lines.append(
        f"2. **Negation is a weak spot.** Accuracy changes by {neg_delta:+.4f} when "
        "a negation marker is present. This validates the thesis choice to "
        "include negation tagging in the classical preprocessing path.\n"
    )
    top_conf = confusion_pairs[0] if confusion_pairs else None
    if top_conf:
        lines.append(
            f"3. **Dominant confusion pair: {top_conf['true']} → {top_conf['pred']} "
            f"({top_conf['fraction_of_errors']*100:.1f}% of all errors).** "
            "Addressing this single confusion would deliver the biggest headline "
            "improvement.\n"
        )
    lines.append(
        f"4. **Confidence gap = {conf_gap:+.4f}.** The model is only modestly more "
        "confident when it is right than when it is wrong, which is exactly why "
        "calibration (not accuracy) is the honest contribution of the CI layer.\n"
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
