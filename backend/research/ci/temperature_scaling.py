"""
Post-Hoc Temperature Scaling Calibration
==========================================

Temperature scaling (Guo et al., 2017) is the simplest and most reliable
post-hoc calibration method for multi-class classifiers.

Method
------
Given a model's raw probability vector p, compute pseudo-logits via:

    z_c = log(p_c + ε)          (log-probability as logit proxy)

Then apply a single learned temperature T > 0:

    p_calibrated_c = exp(z_c / T) / sum_c exp(z_c / T)

T is fitted on the validation set by minimising the Negative Log-Likelihood:

    NLL(T) = -1/N * sum_i sum_c y_ic * log(p_calibrated_ic)

T > 1  →  model was overconfident  (probabilities are sharpened too much)
T < 1  →  model was underconfident (probabilities are too flat)
T = 1  →  no change

Evaluation
----------
For each of the 5 models, we report:
    - ECE  (Expected Calibration Error, 15 bins) before and after scaling
    - Brier score before and after scaling
    - Fitted temperature T
    - Macro-F1 (unchanged — temperature scaling preserves the argmax)

Reference
---------
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks.
    ICML 2017.

Usage
-----
    cd backend
    python research/ci/temperature_scaling.py \\
        --val  data/val.csv   \\
        --test data/test.csv  \\
        --sample_val  10000   \\
        --sample_test 20000   \\
        --seed 42             \\
        --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from research.evaluation.calibration import compute_calibration_metrics

LABELS: List[str] = list(SENTIMENT_LABELS)
# Each entry is a *served* model configuration. Temperature is a per-served-model
# output transform, so the two ensemble variants get independent rows instead of
# sharing a single "ensemble" temperature fitted on the PSO blend only (the old
# behavior silently left ensemble_nsga2 uncalibrated at serving time).
ALL_MODELS = ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]
_ENSEMBLE_VARIANTS = {"ensemble_pso": "pso", "ensemble_nsga2": "nsga2"}


# ---------------------------------------------------------------------------
# Data / scoring helpers
# ---------------------------------------------------------------------------

def load_df(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].apply(normalize_label)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def score_model(name: str, texts: List[str]) -> np.ndarray:
    """Returns (n_samples, n_classes) probability matrix (rows sum to 1)."""
    # calibrate=False: fit T on *raw* model output. Without this, re-running this
    # script would score models through the previously-fitted temperature (loaded
    # automatically from the pinned runtime artifact), fitting a new T on top of
    # an already-calibrated distribution and silently compounding the scaling.
    engine_kwargs = {"calibrate": False}
    if name in _ENSEMBLE_VARIANTS:
        engine_name = "ensemble"
        engine_kwargs["weights_optimization"] = _ENSEMBLE_VARIANTS[name]
    else:
        engine_name = name
    engine = get_sentiment_engine(engine_name, **engine_kwargs)
    results = engine.batch_analyze(texts)
    mat = np.zeros((len(texts), len(LABELS)))
    for i, r in enumerate(results):
        p = normalize_probs(coerce_sentiment_result(r, name).probs)
        for c, lbl in enumerate(LABELS):
            mat[i, c] = p.get(lbl, 0.0)
    return mat


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(prob_matrix: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature T to a probability matrix via log-softmax rescaling.

    z_c = log(p_c + ε)  →  softmax(z / T)

    T > 1  softens the distribution (reduces overconfidence)
    T < 1  sharpens the distribution (reduces underconfidence)
    T = 1  identity
    """
    eps = 1e-10
    log_p = np.log(np.clip(prob_matrix, eps, 1.0))
    scaled = log_p / T
    # Numerically stable softmax
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def nll(prob_matrix: np.ndarray, y_true: List[str]) -> float:
    """Negative log-likelihood."""
    label_idx = {lbl: i for i, lbl in enumerate(LABELS)}
    eps = 1e-10
    total = 0.0
    for i, label in enumerate(y_true):
        c = label_idx[label]
        total -= np.log(prob_matrix[i, c] + eps)
    return total / len(y_true)


def fit_temperature(
    prob_matrix: np.ndarray,
    y_true: List[str],
    T_min: float = 0.1,
    T_max: float = 10.0,
) -> float:
    """
    Find T* = argmin_T NLL(apply_temperature(probs, T), y_true).

    Uses Brent's method (golden-section search) — exact and fast for 1-D.
    """
    def objective(T: float) -> float:
        calibrated = apply_temperature(prob_matrix, T)
        return nll(calibrated, y_true)

    result = minimize_scalar(objective, bounds=(T_min, T_max), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Per-model calibration report
# ---------------------------------------------------------------------------

def calibrate_model(
    name: str,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    y_val: List[str],
    y_test: List[str],
) -> dict:
    """Fit T on val, decide whether to keep it using val alone, evaluate on test.

    The fitted T is only *kept* (and therefore applied at serving time) if it
    reduces **validation**-set ECE relative to the uncalibrated model. NLL
    minimisation is not guaranteed to improve ECE on the same split it was
    computed on, so the val ECE check is a real gate, not a tautology — but
    critically, this keep/discard decision never inspects the test set. The
    test-set before/after numbers below are a read-only evaluation of a
    configuration already fixed on validation, matching the no-test-set-tuning
    discipline applied everywhere else in the thesis (see
    docs/THESIS_RISKS_GAPS.md, "Evaluation Methodology").
    """

    # Fit temperature on validation set
    T = fit_temperature(val_probs, y_val)
    val_calibrated = apply_temperature(val_probs, T)

    # Validation ECE before/after — this is what the keep/discard decision uses.
    ece_before_val, _ = compute_calibration_metrics(y_val, val_probs, labels=LABELS)
    ece_after_val_fitted, _ = compute_calibration_metrics(y_val, val_calibrated, labels=LABELS)
    kept = ece_after_val_fitted < ece_before_val
    T_final = T if kept else 1.0

    # Test-set metrics are computed for both T=1 and T_final, purely for
    # reporting; the value of T_final was already fixed above from val alone.
    test_calibrated = apply_temperature(test_probs, T_final)
    ece_before, brier_before = compute_calibration_metrics(y_test, test_probs, labels=LABELS)
    ece_after, brier_after = compute_calibration_metrics(y_test, test_calibrated, labels=LABELS)
    final_probs = test_calibrated if kept else test_probs

    # Macro-F1 is argmax-invariant — verify it doesn't change when T != 1
    preds_before = [LABELS[i] for i in test_probs.argmax(axis=1)]
    preds_after  = [LABELS[i] for i in final_probs.argmax(axis=1)]
    f1_before = f1_score(y_test, preds_before, average="macro", zero_division=0)
    f1_after  = f1_score(y_test, preds_after,  average="macro", zero_division=0)

    return {
        "model": name,
        "temperature": round(T_final, 4),
        "temperature_fitted": round(T, 4),
        "kept": kept,
        "ece_before_val": round(ece_before_val, 6),
        "ece_after_val_fitted": round(ece_after_val_fitted, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "ece_reduction_pct": round(100 * (ece_before - ece_after) / max(ece_before, 1e-10), 2),
        "brier_before": round(brier_before, 6),
        "brier_after": round(brier_after, 6),
        "brier_reduction_pct": round(100 * (brier_before - brier_after) / max(brier_before, 1e-10), 2),
        "macro_f1_before": round(f1_before, 6),
        "macro_f1_after": round(f1_after, 6),
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(results: List[dict]) -> str:
    n_kept = sum(1 for r in results if r.get("kept"))
    n_discarded = len(results) - n_kept
    lines = [
        "# Temperature Scaling Calibration\n",
        "## Method\n",
        "Temperature scaling (Guo et al., 2017) fits a single scalar T per model "
        "on the validation set by minimising Negative Log-Likelihood.\n",
        "    z_c = log(p_c)   →   p_calibrated = softmax(z / T)\n",
        "- T > 1: model was overconfident → scaling softens probabilities  \n"
        "- T < 1: model was underconfident → scaling sharpens probabilities  \n"
        "- T = 1: no change required (already well-calibrated)\n",
        "**Macro-F1 is unaffected** — temperature scaling preserves the argmax.\n",
        "**Gating**: a fitted T is only *kept* (and served) if it reduces "
        "**validation**-set ECE relative to the uncalibrated model; otherwise T "
        "is pinned to 1.0 (identity) and the fitted value is reported for "
        "reference only. This keep/discard decision never inspects the test "
        "set — the test-set numbers below are a read-only evaluation of a "
        f"configuration already fixed on validation. Of {len(results)} models, "
        f"**{n_kept} kept** their fitted temperature and **{n_discarded} were "
        "discarded** as not improving validation ECE on this run.\n",
        "## Results (Test Set)\n",
        "### ECE (Expected Calibration Error, 15 bins)\n",
        "*Lower is better. Reduction % = (before − after) / before × 100. "
        "\"T (fitted)\" is the NLL-optimal value; \"T (served)\" is 1.0 when the "
        "fitted value was discarded for not improving held-out ECE.*\n",
        "| Model | T (fitted) | T (served) | Kept | ECE Before | ECE After | Reduction |",
        "|-------|------------|------------|------|------------|-----------|-----------|",
    ]
    for r in results:
        fitted = r.get("temperature_fitted", r["temperature"])
        lines.append(
            f"| {r['model']} | {fitted:.3f} | {r['temperature']:.3f} "
            f"| {'yes' if r.get('kept') else 'no'} "
            f"| {r['ece_before']:.4f} | {r['ece_after']:.4f} "
            f"| {r['ece_reduction_pct']:+.1f}% |"
        )

    lines += [
        "\n### Brier Score\n",
        "*Lower is better.*\n",
        "| Model | Brier Before | Brier After | Reduction |",
        "|-------|--------------|-------------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['brier_before']:.4f} "
            f"| {r['brier_after']:.4f} "
            f"| {r['brier_reduction_pct']:+.1f}% |"
        )

    lines += [
        "\n### Macro-F1 (unchanged by design)\n",
        "| Model | Macro-F1 Before | Macro-F1 After | Δ |",
        "|-------|-----------------|----------------|---|",
    ]
    for r in results:
        delta = r["macro_f1_after"] - r["macro_f1_before"]
        lines.append(
            f"| {r['model']} | {r['macro_f1_before']:.4f} "
            f"| {r['macro_f1_after']:.4f} "
            f"| {delta:+.4f} |"
        )

    # Summary stats (computed only over *kept*/served temperatures — averaging
    # in discarded rows, whose served ece_after == ece_before by construction,
    # would understate how much the surviving calibrations actually help).
    kept_results = [r for r in results if r.get("kept")]
    avg_ece_red = (
        sum(r["ece_reduction_pct"] for r in kept_results) / len(kept_results)
        if kept_results else 0.0
    )
    avg_brier_red = (
        sum(r["brier_reduction_pct"] for r in kept_results) / len(kept_results)
        if kept_results else 0.0
    )
    most_overconf = max(results, key=lambda r: r.get("temperature_fitted", r["temperature"]))
    best_ece_gain = max(results, key=lambda r: r["ece_reduction_pct"])

    lines += [
        "\n## Summary\n",
        f"- Models kept (served with fitted T): **{len(kept_results)}/{len(results)}**",
        f"- Average ECE reduction among kept models: **{avg_ece_red:.1f}%**"
        + (" (no models kept)" if not kept_results else ""),
        f"- Average Brier reduction among kept models: **{avg_brier_red:.1f}%**"
        + (" (no models kept)" if not kept_results else ""),
        f"- Most overconfident model (by fitted T, irrespective of whether kept): "
        f"**{most_overconf['model']}** "
        f"(T_fitted={most_overconf.get('temperature_fitted', most_overconf['temperature']):.3f})",
        f"- Largest ECE improvement: **{best_ece_gain['model']}** "
        f"({best_ece_gain['ece_reduction_pct']:+.1f}%)\n",
        "## Thesis Interpretation\n",
        "Temperature scaling provides a lightweight, theoretically-grounded "
        "calibration layer that *can* improve probabilistic reliability without "
        "retraining — but NLL-optimal T is not guaranteed to reduce ECE on the "
        "same split it was fitted on. In this run, the fitted temperature only "
        f"improved validation-set ECE for {len(kept_results)}/{len(results)} "
        "model(s); for the rest, the fitted T made validation ECE *worse*, so "
        "those models are served uncalibrated (T=1.0) rather than shipping a "
        "harmful transform. This decision is made entirely on validation data; "
        "the test-set ECE/Brier columns above are a read-only check of the "
        "already-fixed serving configuration, not part of the gating decision. "
        "See the per-model \"Kept\" column above.\n",
        "- Classical ML models (TF-IDF + LogReg/SVM) output decision-function "
        "scores converted to probabilities via Platt scaling, which can be "
        "systematically over- or under-confident depending on the feature space.\n",
        "- The ensemble and meta-learner aggregate multiple models, which can "
        "amplify or dampen individual model biases — their temperatures reveal "
        "whether aggregation helped or hurt calibration.\n",
        "Because gating pins discarded models to T=1.0 (the identity transform), "
        "temperature scaling remains argmax-preserving and safe to apply at "
        "inference time with no accuracy trade-off for every model, kept or not. "
        "The entropy-gated selective predictor (§4.4) should be read against the "
        "*served* ECE values above, not the fitted-but-discarded ones.\n",
        "## Reference\n",
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "
        "On calibration of modern neural networks. *ICML 2017*.\n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature Scaling Calibration")
    parser.add_argument("--val",  default="data/val.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample_val",  type=int, default=10000)
    parser.add_argument("--sample_test", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading val  ({args.sample_val:,} max) …")
    val_df = load_df(args.val,  args.sample_val,  args.seed)
    y_val  = val_df["label"].tolist()
    texts_val = val_df["text"].tolist()

    print(f"Loading test ({args.sample_test:,} max) …")
    test_df = load_df(args.test, args.sample_test, args.seed)
    y_test  = test_df["label"].tolist()
    texts_test = test_df["text"].tolist()

    print(f"Val: {len(texts_val):,}  Test: {len(texts_test):,}\n")

    # ------------------------------------------------------------------
    # 2. Score, fit, evaluate each model
    # ------------------------------------------------------------------
    results = []
    for name in ALL_MODELS:
        print(f"[{name}]")
        print("  Scoring val  …", flush=True)
        val_probs  = score_model(name, texts_val)
        print("  Scoring test …", flush=True)
        test_probs = score_model(name, texts_test)
        print("  Fitting T …", flush=True)
        row = calibrate_model(name, val_probs, test_probs, y_val, y_test)
        results.append(row)
        print(f"  T={row['temperature']:.3f}  "
              f"ECE {row['ece_before']:.4f}->{row['ece_after']:.4f} "
              f"({row['ece_reduction_pct']:+.1f}%)  "
              f"Brier {row['brier_before']:.4f}->{row['brier_after']:.4f}  "
              f"F1={row['macro_f1_before']:.4f}")

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    # encoding="utf-8" explicitly: on Windows, the default locale encoding
    # (cp1252) cannot encode the arrow/epsilon characters used in the report
    # text and raises UnicodeEncodeError when writing the .md file.
    json_path = out_dir / "temperature_scaling.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"models": results}, fh, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    report = build_report(results)
    md_path = out_dir / "temperature_scaling.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Saved Markdown -> {md_path}")

    # ------------------------------------------------------------------
    # 4. Console table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Model':<14} {'T':>6}  {'ECE_before':>10}  {'ECE_after':>9}  "
          f"{'dECE%':>7}  {'F1':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<14} {r['temperature']:>6.3f}  "
              f"{r['ece_before']:>10.4f}  {r['ece_after']:>9.4f}  "
              f"{r['ece_reduction_pct']:>+7.1f}%  {r['macro_f1_before']:>6.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
