"""
Coverage-Accuracy Curve Analysis
==================================

Compares ALL models (logreg, svm, tfidf, ensemble, meta_learner, neuro-fuzzy)
on a unified coverage-accuracy curve — the definitive selective-prediction
benchmark for the thesis CI chapter.

Method
------
For each model:
  1. Score every test sample → class probabilities
  2. Use max class probability as the confidence score
  3. Sort samples by confidence (descending)
  4. At each coverage level k/N (k = 1 … N), compute accuracy and macro-F1
     on the top-k most confident predictions

This traces a monotonically decreasing accuracy curve as coverage grows:
the fastest the curve declines, the better the model knows when it is right.

Summary metric: AUCA (Area Under the Coverage-Accuracy curve, trapezoidal)
  - Equivalent to the expected accuracy of a selective predictor
  - Higher AUCA = model's confidence better predicts its own correctness

Additional metric: AUC-F1 (Area Under the Coverage-F1 curve)

The neuro-fuzzy gate (Fix 5) is included by loading its saved parameters
from results/neuro_fuzzy_gate.json.

Outputs (backend/results/)
--------------------------
    coverage_accuracy_curve.json    per-model sweep data + AUCA / AUC-F1
    coverage_accuracy_curve.md      thesis-ready comparison tables

Usage
-----
    cd backend
    python research/ci/coverage_accuracy_curve.py \\
        --test data/test.csv  \\
        --sample 20000        \\
        --points 50           \\
        --seed 42             \\
        --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from research.ci.neuro_fuzzy_gate import (
    NeuroFuzzyGate,
    model_confidence,
    N_FUZZY_SETS,
)

LABELS: List[str] = list(SENTIMENT_LABELS)
BASE_MODELS = ["logreg", "svm", "tfidf", "ensemble", "meta_learner"]
NF_MODELS   = ["logreg", "svm", "tfidf"]   # neuro-fuzzy gate uses these three
LOG_C = math.log(len(LABELS))


# ---------------------------------------------------------------------------
# Data / scoring
# ---------------------------------------------------------------------------

def load_df(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].apply(normalize_label)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def score_model(name: str, texts: List[str]) -> np.ndarray:
    """Returns (n_samples, n_classes) probability matrix."""
    engine = get_sentiment_engine(name)
    results = engine.batch_analyze(texts)
    mat = np.zeros((len(texts), len(LABELS)))
    for i, r in enumerate(results):
        p = normalize_probs(coerce_sentiment_result(r, name).probs)
        for c, lbl in enumerate(LABELS):
            mat[i, c] = p.get(lbl, 0.0)
    return mat


def load_neuro_fuzzy_gate(json_path: Path) -> NeuroFuzzyGate | None:
    """Reconstruct NeuroFuzzyGate from saved JSON parameters."""
    if not json_path.exists():
        return None
    with open(json_path) as fh:
        data = json.load(fh)
    mfs = data.get("learned_mfs", [])
    if not mfs:
        return None

    gate = NeuroFuzzyGate(n_models=len(NF_MODELS), n_sets=N_FUZZY_SETS)
    set_order = {"Low": 0, "Medium": 1, "High": 2}
    model_order = {m: i for i, m in enumerate(NF_MODELS)}

    for row in mfs:
        m = model_order.get(row["model"])
        k = set_order.get(row["fuzzy_set"])
        if m is None or k is None:
            continue
        base = m * N_FUZZY_SETS * 3 + k * 3
        gate.theta[base + 0] = row["center"]
        gate.theta[base + 1] = math.log(max(row["width"], 1e-6))
        gate.theta[base + 2] = row["alpha"]

    return gate


# ---------------------------------------------------------------------------
# Coverage-Accuracy computation
# ---------------------------------------------------------------------------

def coverage_accuracy_sweep(
    y_true: List[str],
    probs: np.ndarray,
    n_points: int = 50,
) -> List[dict]:
    """
    Sort samples by max-probability confidence, sweep from most → least
    confident, compute accuracy and macro-F1 at each coverage level.

    Returns list of dicts with keys: coverage, accuracy, macro_f1.
    """
    n = len(y_true)
    preds = [LABELS[i] for i in probs.argmax(axis=1)]
    confidence = probs.max(axis=1)

    # Sort by descending confidence
    order = np.argsort(confidence)[::-1]
    y_sorted   = [y_true[i] for i in order]
    p_sorted   = [preds[i]  for i in order]

    # Coverage levels: from 1/n_points up to 1.0
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    records = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        y_sub = y_sorted[:k]
        p_sub = p_sorted[:k]
        acc = accuracy_score(y_sub, p_sub)
        f1  = f1_score(y_sub, p_sub, average="macro", zero_division=0)
        records.append({
            "coverage": round(float(cov), 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        })
    return records


def auc(sweep: List[dict], metric: str = "accuracy") -> float:
    """Area under the coverage-metric curve via trapezoidal rule."""
    cov = [r["coverage"] for r in sweep]
    val = [r[metric]     for r in sweep]
    return float(np.trapz(val, cov))


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    results: dict,
    summary: List[dict],
) -> str:
    # Sort summary by AUCA descending
    summary_sorted = sorted(summary, key=lambda r: r["auca"], reverse=True)

    lines = [
        "# Coverage-Accuracy Curve Analysis\n",
        "## Method\n",
        "Samples are sorted by model confidence (max class probability, descending). "
        "At each coverage level the accuracy and macro-F1 are computed on the "
        "covered subset. "
        "AUCA = Area Under the Coverage-Accuracy curve (trapezoidal); "
        "higher = model's confidence better predicts its own correctness.\n",
        "## Summary: AUCA and AUC-F1 (Test Set)\n",
        "| Rank | Model | AUCA | AUC-F1 | Acc@100% | F1@100% |",
        "|------|-------|------|--------|----------|---------|",
    ]
    for rank, r in enumerate(summary_sorted, start=1):
        lines.append(
            f"| {rank} | {r['model']} "
            f"| {r['auca']:.4f} | {r['aucf1']:.4f} "
            f"| {r['acc_full']:.4f} | {r['f1_full']:.4f} |"
        )

    lines += ["\n## Coverage-Accuracy at Key Coverage Levels\n",
              "| Model | Acc@10% | Acc@25% | Acc@50% | Acc@75% | Acc@100% |",
              "|-------|---------|---------|---------|---------|----------|"]
    for r in summary_sorted:
        sweep = results[r["model"]]["sweep"]
        def acc_at(target):
            closest = min(sweep, key=lambda x: abs(x["coverage"] - target))
            return closest["accuracy"]
        lines.append(
            f"| {r['model']} "
            f"| {acc_at(0.10):.4f} "
            f"| {acc_at(0.25):.4f} "
            f"| {acc_at(0.50):.4f} "
            f"| {acc_at(0.75):.4f} "
            f"| {acc_at(1.00):.4f} |"
        )

    lines += ["\n## Macro-F1 at Key Coverage Levels\n",
              "| Model | F1@10% | F1@25% | F1@50% | F1@75% | F1@100% |",
              "|-------|--------|--------|--------|--------|---------|"]
    for r in summary_sorted:
        sweep = results[r["model"]]["sweep"]
        def f1_at(target):
            closest = min(sweep, key=lambda x: abs(x["coverage"] - target))
            return closest["macro_f1"]
        lines.append(
            f"| {r['model']} "
            f"| {f1_at(0.10):.4f} "
            f"| {f1_at(0.25):.4f} "
            f"| {f1_at(0.50):.4f} "
            f"| {f1_at(0.75):.4f} "
            f"| {f1_at(1.00):.4f} |"
        )

    best = summary_sorted[0]
    worst = summary_sorted[-1]
    lines += [
        "\n## Thesis Interpretation\n",
        f"The best selective predictor by AUCA is **{best['model']}** "
        f"(AUCA={best['auca']:.4f}), meaning its confidence scores are most "
        "informative about its own correctness.\n",
        f"The weakest is **{worst['model']}** (AUCA={worst['auca']:.4f}), "
        "indicating its confidence is less discriminative.\n",
        "The gap between AUCA values across models reveals that **confidence "
        "quality varies significantly by architecture**, independent of raw "
        "accuracy. A model with lower full-coverage F1 can still have higher "
        "AUCA if it abstains intelligently — this is the core argument for "
        "selective prediction in the thesis CI chapter.\n",
        "### Key Findings\n",
    ]

    # Find model with biggest accuracy lift at 10% coverage vs 100%
    lifts = []
    for r in summary:
        sweep = results[r["model"]]["sweep"]
        acc_10  = min(sweep, key=lambda x: abs(x["coverage"] - 0.10))["accuracy"]
        acc_100 = min(sweep, key=lambda x: abs(x["coverage"] - 1.00))["accuracy"]
        lifts.append((r["model"], acc_10, acc_100, acc_10 - acc_100))
    lifts.sort(key=lambda x: -x[3])
    best_lift = lifts[0]
    lines += [
        f"- **{best_lift[0]}** shows the largest accuracy lift at 10% coverage: "
        f"{best_lift[1]:.4f} vs {best_lift[2]:.4f} full coverage "
        f"(+{best_lift[3]:.4f})",
        f"- Neuro-fuzzy gate AUCA vs static ensemble: "
        f"{next(r['auca'] for r in summary if r['model']=='neuro_fuzzy'):.4f} vs "
        f"{next(r['auca'] for r in summary if r['model']=='ensemble'):.4f}"
        if any(r['model']=='neuro_fuzzy' for r in summary) and
           any(r['model']=='ensemble' for r in summary) else "",
    ]
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-Accuracy Curve Analysis")
    parser.add_argument("--test",   default="data/test.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--points", type=int, default=50,
                        help="Number of coverage points in the sweep")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading test ({args.sample:,} max) …")
    df = load_df(args.test, args.sample, args.seed)
    y_test = df["label"].tolist()
    texts  = df["text"].tolist()
    print(f"  {len(texts):,} samples\n")

    # ------------------------------------------------------------------
    # 2. Score all base models
    # ------------------------------------------------------------------
    prob_matrices: dict[str, np.ndarray] = {}
    for name in BASE_MODELS:
        print(f"Scoring {name} …", flush=True)
        prob_matrices[name] = score_model(name, texts)

    # ------------------------------------------------------------------
    # 3. Neuro-fuzzy gate — load saved parameters and predict
    # ------------------------------------------------------------------
    nf_json = out_dir / "neuro_fuzzy_gate.json"
    gate = load_neuro_fuzzy_gate(nf_json)

    if gate is not None:
        print("Loading neuro-fuzzy gate from saved JSON …")
        # Need probability cube for the 3 base models
        nf_cube = np.stack([prob_matrices[m] for m in NF_MODELS], axis=0)
        nf_conf = np.stack([model_confidence(prob_matrices[m]) for m in NF_MODELS], axis=1)
        nf_probs = gate.predict_probs(nf_conf, nf_cube)
        prob_matrices["neuro_fuzzy"] = nf_probs
        print("  Neuro-fuzzy gate loaded OK")
    else:
        print(f"  WARNING: {nf_json} not found — neuro-fuzzy gate skipped")

    # ------------------------------------------------------------------
    # 4. Compute coverage-accuracy curves
    # ------------------------------------------------------------------
    all_names = BASE_MODELS + (["neuro_fuzzy"] if "neuro_fuzzy" in prob_matrices else [])
    results: dict = {}
    summary: list = []

    print("\nComputing coverage-accuracy curves …")
    for name in all_names:
        probs = prob_matrices[name]
        sweep = coverage_accuracy_sweep(y_test, probs, n_points=args.points)

        auca  = auc(sweep, "accuracy")
        aucf1 = auc(sweep, "macro_f1")

        # Full-coverage metrics (last point)
        acc_full = sweep[-1]["accuracy"]
        f1_full  = sweep[-1]["macro_f1"]

        results[name] = {"sweep": sweep, "auca": auca, "aucf1": aucf1}
        summary.append({
            "model": name,
            "auca": round(auca, 4),
            "aucf1": round(aucf1, 4),
            "acc_full": round(acc_full, 4),
            "f1_full": round(f1_full, 4),
        })
        print(f"  {name:<15}  AUCA={auca:.4f}  AUC-F1={aucf1:.4f}  "
              f"Acc@100%={acc_full:.4f}  F1@100%={f1_full:.4f}")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    output_data = {
        "n_samples": len(texts),
        "n_coverage_points": args.points,
        "summary": summary,
        "curves": {
            name: results[name]["sweep"]
            for name in all_names
        },
    }
    json_path = out_dir / "coverage_accuracy_curve.json"
    with open(json_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nSaved JSON → {json_path}")

    report = build_report(results, summary)
    md_path = out_dir / "coverage_accuracy_curve.md"
    with open(md_path, "w") as fh:
        fh.write(report)
    print(f"Saved Markdown → {md_path}")

    # ------------------------------------------------------------------
    # 6. Console summary table
    # ------------------------------------------------------------------
    summary_sorted = sorted(summary, key=lambda r: r["auca"], reverse=True)
    print("\n" + "=" * 70)
    print(f"{'Rank':<5} {'Model':<16} {'AUCA':>7}  {'AUC-F1':>7}  "
          f"{'Acc@10%':>8}  {'Acc@100%':>9}")
    print("-" * 70)
    for rank, r in enumerate(summary_sorted, start=1):
        sweep = results[r["model"]]["sweep"]
        acc_10 = min(sweep, key=lambda x: abs(x["coverage"] - 0.10))["accuracy"]
        print(f"{rank:<5} {r['model']:<16} {r['auca']:>7.4f}  {r['aucf1']:>7.4f}  "
              f"{acc_10:>8.4f}  {r['acc_full']:>9.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
