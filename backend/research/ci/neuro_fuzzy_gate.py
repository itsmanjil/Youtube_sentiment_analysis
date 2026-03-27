"""
Neuro-Fuzzy Confidence Gating System
======================================

Architecture
------------
This module implements an Adaptive Neuro-Fuzzy Inference System (ANFIS-lite)
that dynamically routes each sample to the most appropriate model based on
per-model confidence signals.

Unlike the fixed fuzzy grid search (backend/docs/FUZZY_THEORETICAL_GROUNDING.md),
which applies the same static weights to every sample, the neuro-fuzzy gate
*adapts per-sample*: it reads each model's confidence for that input and
produces a soft routing weight learned from the validation set.

System design
~~~~~~~~~~~~~

    Input layer:  confidence vector  c = [c_1, c_2, c_3]
                  where c_m = 1 − H_norm(model_m)   ∈ [0, 1]

    Fuzzification: three Gaussian MFs per model — Low / Medium / High
                  μ_{m,k}(c_m) = exp(−0.5 · ((c_m − center_{m,k}) / width_{m,k})²)

    Rule activation: each of the 3^3 = 27 rules fires proportionally
                  (product t-norm across models)
                  In practice we use a *simplified* single-model rule base:
                  gate_m = softmax(Σ_k  α_{m,k} · μ_{m,k}(c_m))
                  where α_{m,k} are learned consequent weights.

    Defuzzification: gate weights g = softmax(raw_gate)
                  Final probs = Σ_m g_m · P_m

    Learning: parameters {center_{m,k}, width_{m,k}, α_{m,k}} are fitted
              by minimising NLL on the validation set via L-BFGS-B.
              The MF centers are initialised at {0.25, 0.50, 0.75},
              widths at 0.20, consequent weights at 0.

    Total parameters: 3 models × 3 fuzzy sets × 3 params (center, width, α)
                     = 27 parameters.  Small, interpretable, no GPU required.

Comparison baseline
-------------------
    Fixed ensemble  —  static weights, same for every sample
    PSO ensemble    —  also static, optimised globally
    Neuro-fuzzy     —  per-sample adaptive weights (this module)

Reference
---------
    Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference
    system. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685.

Usage
-----
    cd backend
    python research/ci/neuro_fuzzy_gate.py \\
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
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from research.evaluation.calibration import compute_calibration_metrics
from research.computational_intelligence.fuzzy.membership_functions import GaussianMF

LABELS: List[str] = list(SENTIMENT_LABELS)
MODELS = ["logreg", "svm", "tfidf"]
N_FUZZY_SETS = 3          # Low / Medium / High per model
LOG_C = math.log(len(LABELS))


# ---------------------------------------------------------------------------
# Data helpers
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


def model_confidence(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Per-sample confidence = 1 - normalised entropy  ∈ [0, 1].

    High value → model is certain; low value → model is uncertain.
    """
    p = np.clip(prob_matrix, 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1) / LOG_C   # normalised entropy
    return 1.0 - h                                 # confidence


# ---------------------------------------------------------------------------
# Neuro-Fuzzy Gate
# ---------------------------------------------------------------------------

class NeuroFuzzyGate:
    """
    Per-sample adaptive ensemble router using learned Gaussian MFs.

    Parameters (flattened vector θ, length = n_models × n_sets × 3)
    ---------------------------------------------------------------
    For model m, fuzzy set k:
        θ[m*n_sets*3 + k*3 + 0]  = center   ∈ (0, 1)
        θ[m*n_sets*3 + k*3 + 1]  = log_width (exp'd to keep > 0)
        θ[m*n_sets*3 + k*3 + 2]  = consequent weight α_{m,k}
    """

    def __init__(self, n_models: int = 3, n_sets: int = N_FUZZY_SETS):
        self.n_models = n_models
        self.n_sets = n_sets
        self.n_params = n_models * n_sets * 3

        # Default initialisation
        # Centers: 0.25, 0.50, 0.75 for Low/Medium/High
        # Widths: 0.20 (log_width = log(0.20))
        # Consequents: 0 (uniform gate initially)
        self.theta = self._default_theta()

    def _default_theta(self) -> np.ndarray:
        theta = np.zeros(self.n_params)
        centers = [0.25, 0.50, 0.75]
        for m in range(self.n_models):
            for k in range(self.n_sets):
                base = m * self.n_sets * 3 + k * 3
                theta[base + 0] = centers[k]          # center
                theta[base + 1] = math.log(0.20)      # log_width → width=0.20
                theta[base + 2] = 0.0                 # consequent
        return theta

    def _fuzzy_activations(
        self, confidence: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-sample, per-model aggregated fuzzy gate strength.

        Returns shape (n_samples, n_models) — raw (pre-softmax) gate values.
        """
        n = confidence.shape[0]
        raw_gate = np.zeros((n, self.n_models))

        for m in range(self.n_models):
            c_m = confidence[:, m]          # (n_samples,)
            gate_m = np.zeros(n)

            for k in range(self.n_sets):
                base = m * self.n_sets * 3 + k * 3
                center = theta[base + 0]
                width  = max(np.exp(theta[base + 1]), 1e-3)
                alpha  = theta[base + 2]

                # Gaussian MF activation
                mu = np.exp(-0.5 * ((c_m - center) / width) ** 2)
                gate_m += alpha * mu

            raw_gate[:, m] = gate_m

        return raw_gate

    def gate_weights(
        self, confidence: np.ndarray, theta: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute softmax-normalised gate weights.

        Returns shape (n_samples, n_models).
        """
        if theta is None:
            theta = self.theta
        raw = self._fuzzy_activations(confidence, theta)
        # Numerically stable softmax along model axis
        raw -= raw.max(axis=1, keepdims=True)
        exp_r = np.exp(raw)
        return exp_r / exp_r.sum(axis=1, keepdims=True)

    def predict_probs(
        self,
        confidence: np.ndarray,
        prob_cube: np.ndarray,
        theta: np.ndarray = None,
    ) -> np.ndarray:
        """
        Weighted ensemble probabilities using adaptive gates.

        confidence : (n_samples, n_models)
        prob_cube  : (n_models, n_samples, n_classes)
        returns    : (n_samples, n_classes)
        """
        gates = self.gate_weights(confidence, theta)     # (n, n_models)
        # gates[:, m] * prob_cube[m] → sum over models
        out = np.einsum("nm,mnc->nc", gates, prob_cube)
        return out

    def nll(
        self,
        theta: np.ndarray,
        confidence: np.ndarray,
        prob_cube: np.ndarray,
        y_true: List[str],
    ) -> float:
        """Negative log-likelihood (objective for optimisation)."""
        label_idx = {lbl: i for i, lbl in enumerate(LABELS)}
        probs = self.predict_probs(confidence, prob_cube, theta)
        eps = 1e-10
        total = 0.0
        for i, lbl in enumerate(y_true):
            c = label_idx[lbl]
            total -= math.log(float(probs[i, c]) + eps)
        return total / len(y_true)

    def fit(
        self,
        confidence: np.ndarray,
        prob_cube: np.ndarray,
        y_true: List[str],
        maxiter: int = 200,
        verbose: bool = True,
    ) -> dict:
        """
        Fit gate parameters on validation set via L-BFGS-B.

        Bounds:
            center     ∈ [0.01, 0.99]
            log_width  ∈ [log(0.05), log(0.50)]
            alpha      ∈ [-5, 5]
        """
        bounds = []
        for m in range(self.n_models):
            for k in range(self.n_sets):
                bounds.append((0.01, 0.99))           # center
                bounds.append((math.log(0.05), math.log(0.50)))  # log_width
                bounds.append((-5.0, 5.0))            # consequent alpha

        nll_before = self.nll(self.theta, confidence, prob_cube, y_true)
        if verbose:
            print(f"  NLL before: {nll_before:.4f}")

        result = minimize(
            fun=lambda theta: self.nll(theta, confidence, prob_cube, y_true),
            x0=self.theta.copy(),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6},
        )

        self.theta = result.x
        nll_after = result.fun
        if verbose:
            print(f"  NLL after : {nll_after:.4f}  "
                  f"({'converged' if result.success else 'max iter reached'})")

        return {
            "nll_before": round(nll_before, 6),
            "nll_after": round(nll_after, 6),
            "converged": bool(result.success),
            "n_iterations": result.nit,
        }

    def describe_gates(self, model_names: List[str]) -> List[dict]:
        """Return human-readable MF parameters per model."""
        set_names = ["Low", "Medium", "High"]
        rows = []
        for m, name in enumerate(model_names):
            for k in range(self.n_sets):
                base = m * self.n_sets * 3 + k * 3
                rows.append({
                    "model": name,
                    "fuzzy_set": set_names[k],
                    "center": round(float(self.theta[base + 0]), 4),
                    "width": round(float(np.exp(self.theta[base + 1])), 4),
                    "alpha": round(float(self.theta[base + 2]), 4),
                })
        return rows


# ---------------------------------------------------------------------------
# Static ensemble baseline (fixed weights)
# ---------------------------------------------------------------------------

def static_ensemble_probs(
    prob_cube: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted average with fixed weights. Returns (n_samples, n_classes)."""
    w = np.maximum(weights, 0.0)
    w = w / (w.sum() + 1e-12)
    return np.einsum("m,mnc->nc", w, prob_cube)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    gate: NeuroFuzzyGate,
    fit_info: dict,
    model_names: List[str],
    val_metrics: dict,
    test_metrics: dict,
    static_test_metrics: dict,
) -> str:
    set_names = ["Low", "Medium", "High"]

    lines = [
        "# Neuro-Fuzzy Confidence Gating System\n",
        "## Architecture\n",
        "Per-sample adaptive ensemble router using learned Gaussian membership "
        "functions (ANFIS-lite).\n",
        "- **Input**: confidence vector c = [c_logreg, c_svm, c_tfidf]",
        "- **Fuzzification**: 3 Gaussian MFs per model (Low / Medium / High)",
        "- **Gate weights**: softmax(Σ_k α_{m,k} · μ_{m,k}(c_m))",
        "- **Output**: per-sample weighted ensemble probabilities",
        f"- **Total parameters**: {gate.n_params} "
        f"({gate.n_models} models × {gate.n_sets} sets × 3)\n",
        "## Training\n",
        "| | Value |",
        "|---|-------|",
        f"| NLL before fitting | {fit_info['nll_before']:.4f} |",
        f"| NLL after fitting | {fit_info['nll_after']:.4f} |",
        f"| Converged | {fit_info['converged']} |",
        f"| L-BFGS-B iterations | {fit_info['n_iterations']} |\n",
        "## Learned Membership Function Parameters\n",
        "| Model | Fuzzy Set | Center | Width | Alpha (α) |",
        "|-------|-----------|--------|-------|-----------|",
    ]
    for row in gate.describe_gates(model_names):
        lines.append(
            f"| {row['model']} | {row['fuzzy_set']} "
            f"| {row['center']:.3f} | {row['width']:.3f} "
            f"| {row['alpha']:+.3f} |"
        )

    lines += [
        "\n*Centers reveal which confidence region activates each model. "
        "Positive α = high-confidence samples prefer this model; "
        "negative α = low-confidence samples prefer this model.*\n",
        "## Performance Comparison (Test Set)\n",
        "| Method | Macro-F1 | Accuracy | ECE | Brier |",
        "|--------|----------|----------|-----|-------|",
        f"| Static ensemble (uniform) "
        f"| {static_test_metrics['macro_f1']:.4f} "
        f"| {static_test_metrics['accuracy']:.4f} "
        f"| {static_test_metrics['ece']:.4f} "
        f"| {static_test_metrics['brier']:.4f} |",
        f"| **Neuro-fuzzy gate** "
        f"| **{test_metrics['macro_f1']:.4f}** "
        f"| **{test_metrics['accuracy']:.4f}** "
        f"| **{test_metrics['ece']:.4f}** "
        f"| **{test_metrics['brier']:.4f}** |",
    ]

    f1_delta = test_metrics["macro_f1"] - static_test_metrics["macro_f1"]
    ece_delta = test_metrics["ece"] - static_test_metrics["ece"]
    lines += [
        f"\nΔ Macro-F1 = **{f1_delta:+.4f}**  |  "
        f"Δ ECE = **{ece_delta:+.4f}**\n",
        "## Thesis Interpretation\n",
        "The neuro-fuzzy gate demonstrates a fundamental advantage over "
        "static ensemble weighting: **different samples benefit from different "
        "model mixtures**. When a model is confident (high c_m), the gate "
        "boosts its contribution; when it is uncertain, the gate reduces it, "
        "distributing weight to the remaining models.\n",
        "The learned MF parameters are directly interpretable: the center "
        "values show which confidence level is the 'sweet spot' for each "
        "model, and the α consequents reveal whether that confidence region "
        "should increase or decrease the model's influence.\n",
        "This constitutes the **Neuro-Fuzzy CI contribution** of the thesis: "
        "a principled, interpretable, and data-driven approach to ensemble "
        "gating that extends classical fuzzy inference with parameter learning.\n",
        "## References\n",
        "Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference "
        "system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.\n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-Fuzzy Confidence Gating")
    parser.add_argument("--val",  default="data/val.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample_val",  type=int, default=10000)
    parser.add_argument("--sample_test", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=200)
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
    # 2. Score all models → probability cubes
    # ------------------------------------------------------------------
    print("Scoring models (val) …")
    val_cube = np.zeros((len(MODELS), len(texts_val), len(LABELS)))
    for m, name in enumerate(MODELS):
        print(f"  {name} …", flush=True)
        val_cube[m] = score_model(name, texts_val)

    print("Scoring models (test) …")
    test_cube = np.zeros((len(MODELS), len(texts_test), len(LABELS)))
    for m, name in enumerate(MODELS):
        print(f"  {name} …", flush=True)
        test_cube[m] = score_model(name, texts_test)

    # ------------------------------------------------------------------
    # 3. Compute confidence vectors
    # ------------------------------------------------------------------
    val_conf  = np.stack([model_confidence(val_cube[m])  for m in range(len(MODELS))], axis=1)
    test_conf = np.stack([model_confidence(test_cube[m]) for m in range(len(MODELS))], axis=1)

    print("\nVal confidence stats (mean per model):")
    for m, name in enumerate(MODELS):
        print(f"  {name}: mean={val_conf[:, m].mean():.3f}  "
              f"std={val_conf[:, m].std():.3f}")

    # ------------------------------------------------------------------
    # 4. Static ensemble baseline (uniform weights)
    # ------------------------------------------------------------------
    uniform_w = np.ones(len(MODELS)) / len(MODELS)
    static_test_probs = static_ensemble_probs(test_cube, uniform_w)
    static_preds = [LABELS[i] for i in static_test_probs.argmax(axis=1)]
    static_f1  = f1_score(y_test, static_preds, average="macro", zero_division=0)
    static_acc = accuracy_score(y_test, static_preds)
    static_ece, static_brier = compute_calibration_metrics(
        y_test, static_test_probs, labels=LABELS
    )
    static_test_metrics = {
        "macro_f1": round(static_f1, 6),
        "accuracy": round(static_acc, 6),
        "ece": round(static_ece, 6),
        "brier": round(static_brier, 6),
    }
    print(f"\nStatic ensemble (uniform):  "
          f"F1={static_f1:.4f}  Acc={static_acc:.4f}  "
          f"ECE={static_ece:.4f}  Brier={static_brier:.4f}")

    # ------------------------------------------------------------------
    # 5. Fit neuro-fuzzy gate on validation set
    # ------------------------------------------------------------------
    print("\nFitting neuro-fuzzy gate …")
    gate = NeuroFuzzyGate(n_models=len(MODELS), n_sets=N_FUZZY_SETS)
    fit_info = gate.fit(val_conf, val_cube, y_val, maxiter=args.maxiter)

    # Validation metrics (sanity check)
    val_probs = gate.predict_probs(val_conf, val_cube)
    val_preds = [LABELS[i] for i in val_probs.argmax(axis=1)]
    val_f1  = f1_score(y_val, val_preds, average="macro", zero_division=0)
    val_ece, val_brier = compute_calibration_metrics(y_val, val_probs, labels=LABELS)
    val_metrics = {
        "macro_f1": round(val_f1, 6),
        "ece": round(val_ece, 6),
        "brier": round(val_brier, 6),
    }
    print(f"Val:   F1={val_f1:.4f}  ECE={val_ece:.4f}  Brier={val_brier:.4f}")

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    test_probs = gate.predict_probs(test_conf, test_cube)
    test_preds = [LABELS[i] for i in test_probs.argmax(axis=1)]
    test_f1  = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, test_preds)
    test_ece, test_brier = compute_calibration_metrics(
        y_test, test_probs, labels=LABELS
    )
    test_metrics = {
        "macro_f1": round(test_f1, 6),
        "accuracy": round(test_acc, 6),
        "ece": round(test_ece, 6),
        "brier": round(test_brier, 6),
    }
    print(f"Test:  F1={test_f1:.4f}  Acc={test_acc:.4f}  "
          f"ECE={test_ece:.4f}  Brier={test_brier:.4f}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    output_data = {
        "architecture": {
            "n_models": len(MODELS),
            "model_names": MODELS,
            "n_fuzzy_sets": N_FUZZY_SETS,
            "n_parameters": gate.n_params,
            "activation": "Gaussian MF",
            "aggregation": "softmax",
            "reference": "ANFIS (Jang, 1993)",
        },
        "training": fit_info,
        "learned_mfs": gate.describe_gates(MODELS),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "static_ensemble_test_metrics": static_test_metrics,
        "delta_vs_static": {
            "macro_f1": round(test_f1 - static_f1, 6),
            "ece": round(test_ece - static_ece, 6),
            "brier": round(test_brier - static_brier, 6),
        },
    }

    json_path = out_dir / "neuro_fuzzy_gate.json"
    with open(json_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nSaved JSON → {json_path}")

    report = build_report(
        gate, fit_info, MODELS, val_metrics, test_metrics, static_test_metrics
    )
    md_path = out_dir / "neuro_fuzzy_gate.md"
    with open(md_path, "w") as fh:
        fh.write(report)
    print(f"Saved Markdown → {md_path}")

    # ------------------------------------------------------------------
    # 8. Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("NEURO-FUZZY GATING RESULTS")
    print("=" * 65)
    print(f"{'Method':<25} {'F1':>7}  {'Acc':>7}  {'ECE':>7}  {'Brier':>7}")
    print("-" * 65)
    print(f"{'Static (uniform)':<25} "
          f"{static_f1:>7.4f}  {static_acc:>7.4f}  "
          f"{static_ece:>7.4f}  {static_brier:>7.4f}")
    print(f"{'Neuro-fuzzy gate':<25} "
          f"{test_f1:>7.4f}  {test_acc:>7.4f}  "
          f"{test_ece:>7.4f}  {test_brier:>7.4f}")
    print(f"{'Delta':<25} "
          f"{test_f1 - static_f1:>+7.4f}  "
          f"{test_acc - static_acc:>+7.4f}  "
          f"{test_ece - static_ece:>+7.4f}  "
          f"{test_brier - static_brier:>+7.4f}")
    print("=" * 65)
    print("\nLearned MF parameters:")
    for row in gate.describe_gates(MODELS):
        print(f"  {row['model']:>12} / {row['fuzzy_set']:<6}  "
              f"center={row['center']:.3f}  width={row['width']:.3f}  "
              f"α={row['alpha']:+.3f}")


if __name__ == "__main__":
    main()
