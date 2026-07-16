"""
CI Method Significance Testing
================================

Runs pairwise McNemar's tests (Holm-adjusted) between every classical model
and every CI method on the same held-out test set.

Methods compared
----------------
Classical baselines:
    logreg        TF-IDF + Logistic Regression
    svm           TF-IDF + SVM
    tfidf         TF-IDF + Naive Bayes
    ensemble      Built-in weighted ensemble
    meta_learner  Stacking meta-learner

CI contributions:
    pso           Single-objective PSO ensemble (logreg/svm/tfidf)
    nsga2         NSGA-II multi-objective knee-point ensemble
    neuro_fuzzy   Adaptive neuro-fuzzy gate
    ts_meta       Temperature-scaled meta_learner (T from Fix 4)

Test
----
McNemar's test on discordant pairs from the same test set (n=20k, seed=42).
Holm-Bonferroni correction applied across all pairwise comparisons.

Outputs
-------
    results/ci_significance_tests.json    raw p-values + Holm-adjusted
    results/ci_significance_tests.md      thesis-ready table

Usage
-----
    cd backend
    python research/ci/ci_significance_tests.py \\
        --test  data/test.csv \\
        --sample 20000        \\
        --seed  42            \\
        --output results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from research.ci.temperature_scaling import apply_temperature
from research.ci.neuro_fuzzy_gate import (
    NeuroFuzzyGate, model_confidence, N_FUZZY_SETS,
)

LABELS: List[str] = list(SENTIMENT_LABELS)
BASE_MODELS     = ["logreg", "svm", "tfidf", "ensemble", "meta_learner"]
ENSEMBLE_MODELS = ["logreg", "svm", "tfidf"]   # used by PSO / NSGA-II / NF


# ---------------------------------------------------------------------------
# Data & scoring
# ---------------------------------------------------------------------------

def load_df(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].apply(normalize_label)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def score_model(name: str, texts: List[str]) -> np.ndarray:
    engine = get_sentiment_engine(name)
    results = engine.batch_analyze(texts)
    mat = np.zeros((len(texts), len(LABELS)))
    for i, r in enumerate(results):
        p = normalize_probs(coerce_sentiment_result(r, name).probs)
        for c, lbl in enumerate(LABELS):
            mat[i, c] = p.get(lbl, 0.0)
    return mat


def weighted_preds(
    prob_matrices: Dict[str, np.ndarray], weights: Dict[str, float]
) -> List[str]:
    total = sum(max(0.0, w) for w in weights.values())
    if total < 1e-12:
        total = 1.0
    combined = np.zeros((next(iter(prob_matrices.values())).shape[0], len(LABELS)))
    for name, mat in prob_matrices.items():
        combined += max(0.0, weights.get(name, 0.0)) / total * mat
    return [LABELS[i] for i in combined.argmax(axis=1)]


# ---------------------------------------------------------------------------
# McNemar's test (exact binomial, two-sided)
# ---------------------------------------------------------------------------

def mcnemar(
    y_true: List[str],
    pred_a: List[str],
    pred_b: List[str],
) -> dict:
    """Exact two-sided McNemar's test on the discordant pairs."""
    ya, yb, yt = np.array(pred_a), np.array(pred_b), np.array(y_true)
    ok_a = ya == yt
    ok_b = yb == yt
    n01 = int(np.sum(~ok_a &  ok_b))   # A wrong, B right
    n10 = int(np.sum( ok_a & ~ok_b))   # A right, B wrong
    n   = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n": 0, "p_value": 1.0, "statistic": None}
    # Exact: under H0, n01 ~ Bin(n, 0.5); two-sided
    k   = min(n01, n10)
    p   = float(2 * binom.cdf(k, n, 0.5))
    p   = min(p, 1.0)
    chi2 = (abs(n01 - n10) - 1) ** 2 / n if n >= 25 else None
    return {"n01": n01, "n10": n10, "n": n, "p_value": p, "statistic": chi2}


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_correct(p_values: List[float]) -> List[float]:
    """Returns Holm-adjusted p-values (same order as input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

ALPHA = 0.05

def significance_star(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def build_report(
    pairs: List[dict],
    method_f1: Dict[str, float],
    n_test: int,
) -> str:
    # Sort by adjusted p-value
    pairs_sorted = sorted(pairs, key=lambda r: r["p_adj"])

    lines = [
        "# CI Method Significance Testing\n",
        "## Setup\n",
        f"- **Test set**: n = {n_test:,} samples (seed=42)",
        "- **Test**: McNemar's exact two-sided test on discordant pairs",
        "- **Correction**: Holm-Bonferroni across all pairwise comparisons",
        f"- **α** = {ALPHA}  (`*` p<0.05  `**` p<0.01  `***` p<0.001  `ns` not significant)\n",
        "## Method Performance Summary\n",
        "| Method | Macro-F1 | Type |",
        "|--------|----------|------|",
    ]
    method_types = {
        "logreg": "Classical baseline",
        "svm": "Classical baseline",
        "tfidf": "Classical baseline",
        "ensemble": "Classical baseline",
        "meta_learner": "Classical baseline",
        "pso": "CI — Single-obj PSO",
        "nsga2": "CI — Multi-obj NSGA-II",
        "neuro_fuzzy": "CI — Neuro-fuzzy gate",
        "ts_meta": "CI — Temp-scaled meta",
    }
    for name, f1 in sorted(method_f1.items(), key=lambda x: -x[1]):
        lines.append(
            f"| {name} | {f1:.4f} | {method_types.get(name, '—')} |"
        )

    lines += [
        "\n## Pairwise McNemar's Tests (Holm-adjusted)\n",
        "| Method A | Method B | F1(A) | F1(B) | n01 | n10 | p_raw | p_adj | Sig |",
        "|----------|----------|-------|-------|-----|-----|-------|-------|-----|",
    ]
    for r in pairs_sorted:
        lines.append(
            f"| {r['method_a']} | {r['method_b']} "
            f"| {method_f1[r['method_a']]:.4f} "
            f"| {method_f1[r['method_b']]:.4f} "
            f"| {r['n01']} | {r['n10']} "
            f"| {r['p_raw']:.4e} | {r['p_adj']:.4e} "
            f"| {significance_star(r['p_adj'])} |"
        )

    # Key findings
    sig_pairs = [r for r in pairs_sorted if r["p_adj"] < ALPHA]
    ns_pairs  = [r for r in pairs_sorted if r["p_adj"] >= ALPHA]

    ci_methods = {"pso", "nsga2", "neuro_fuzzy", "ts_meta"}
    ci_vs_classical = [
        r for r in sig_pairs
        if (r["method_a"] in ci_methods) != (r["method_b"] in ci_methods)
    ]
    ci_internal = [
        r for r in sig_pairs
        if r["method_a"] in ci_methods and r["method_b"] in ci_methods
    ]

    lines += [
        f"\n## Key Findings\n",
        f"- **{len(sig_pairs)}/{len(pairs_sorted)}** pairs are statistically significant (p_adj < {ALPHA})",
        f"- **CI vs Classical** significant pairs: {len(ci_vs_classical)}",
        f"- **CI vs CI** significant pairs: {len(ci_internal)}\n",
    ]

    if ci_vs_classical:
        lines.append("### CI methods significantly outperforming classical baselines:\n")
        for r in ci_vs_classical:
            winner = r["method_a"] if method_f1[r["method_a"]] > method_f1[r["method_b"]] else r["method_b"]
            loser  = r["method_b"] if winner == r["method_a"] else r["method_a"]
            delta  = abs(method_f1[winner] - method_f1[loser])
            lines.append(
                f"- **{winner}** > {loser}  "
                f"(ΔF1={delta:+.4f}, p_adj={r['p_adj']:.4e} {significance_star(r['p_adj'])})"
            )

    if ns_pairs:
        ns_ci = [
            r for r in ns_pairs
            if r["method_a"] in ci_methods or r["method_b"] in ci_methods
        ]
        if ns_ci:
            lines.append(
                f"\n### CI pairs NOT significant (p_adj ≥ {ALPHA}):\n"
            )
            for r in ns_ci[:6]:  # cap at 6 for brevity
                lines.append(
                    f"- {r['method_a']} vs {r['method_b']}  "
                    f"(p_adj={r['p_adj']:.4e} — insufficient evidence of difference)"
                )

    lines += [
        "\n## Thesis Interpretation\n",
        "McNemar's test is appropriate here because all methods are evaluated "
        "on the same test set and the question is whether the *pattern of errors* "
        "differs — not the overall accuracy level. "
        "The Holm-Bonferroni correction controls the family-wise error rate "
        "across all pairwise comparisons.\n",
        "Significant results (p_adj < 0.05) indicate that the two methods make "
        "systematically different errors on specific samples — meaning one "
        "method genuinely recovers samples the other cannot. "
        "Non-significant results indicate that any observed F1 difference is "
        "within the range of chance variation on this test set.\n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CI Method Significance Tests")
    parser.add_argument("--test",    default="data/test.csv")
    parser.add_argument("--sample",  type=int, default=20000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default="results/")
    parser.add_argument(
        "--nf_json",
        default="results/neuro_fuzzy_gate.json",
        help="Neuro-fuzzy gate parameters (from Fix 5)",
    )
    parser.add_argument(
        "--moe_json",
        default="results/multi_objective_ensemble.json",
        help="NSGA-II knee-point weights (from Fix 2)",
    )
    parser.add_argument(
        "--pso_json",
        default="results/pso_convergence.json",
        help="PSO best weights (from pso_convergence_analysis)",
    )
    parser.add_argument(
        "--ts_json",
        default="results/temperature_scaling.json",
        help="Temperature scaling results (from Fix 4)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load parameters
    # ------------------------------------------------------------------
    nf_data  = json.loads(Path(args.nf_json).read_text())
    moe_data = json.loads(Path(args.moe_json).read_text())
    pso_data = json.loads(Path(args.pso_json).read_text())
    ts_data  = json.loads(Path(args.ts_json).read_text())

    nsga2_weights = moe_data["knee_point"]["weights"]           # logreg/svm/tfidf
    pso_weights   = pso_data["results"]["best_weights"]         # logreg/svm/tfidf
    ts_temps      = {r["model"]: r["temperature"] for r in ts_data["models"]}

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    print(f"Loading test ({args.sample:,} max, seed={args.seed}) …")
    df     = load_df(args.test, args.sample, args.seed)
    y_true = df["label"].tolist()
    texts  = df["text"].tolist()
    print(f"  {len(texts):,} samples\n")

    # ------------------------------------------------------------------
    # 3. Score base models
    # ------------------------------------------------------------------
    prob_matrices: Dict[str, np.ndarray] = {}
    print("Scoring models …")
    for name in BASE_MODELS:
        print(f"  {name} …", flush=True)
        prob_matrices[name] = score_model(name, texts)

    ens_probs = {m: prob_matrices[m] for m in ENSEMBLE_MODELS}

    # ------------------------------------------------------------------
    # 4. Build CI method predictions
    # ------------------------------------------------------------------
    print("\nBuilding CI method predictions …")

    # PSO ensemble
    pso_preds   = weighted_preds(ens_probs, pso_weights)

    # NSGA-II ensemble
    nsga2_preds = weighted_preds(ens_probs, nsga2_weights)

    # Temperature-scaled meta_learner
    T_meta = ts_temps.get("meta_learner", 1.0)
    ts_meta_probs = apply_temperature(prob_matrices["meta_learner"], T_meta)
    ts_meta_preds = [LABELS[i] for i in ts_meta_probs.argmax(axis=1)]

    # Neuro-fuzzy gate
    gate = NeuroFuzzyGate(n_models=len(ENSEMBLE_MODELS), n_sets=N_FUZZY_SETS)
    set_order   = {"Low": 0, "Medium": 1, "High": 2}
    model_order = {m: i for i, m in enumerate(ENSEMBLE_MODELS)}
    for row in nf_data.get("learned_mfs", []):
        mi = model_order.get(row["model"])
        k  = set_order.get(row["fuzzy_set"])
        if mi is None or k is None:
            continue
        base = mi * N_FUZZY_SETS * 3 + k * 3
        gate.theta[base + 0] = row["center"]
        gate.theta[base + 1] = math.log(max(row["width"], 1e-6))
        gate.theta[base + 2] = row["alpha"]

    nf_cube = np.stack([prob_matrices[m] for m in ENSEMBLE_MODELS], axis=0)
    nf_conf = np.stack([model_confidence(prob_matrices[m]) for m in ENSEMBLE_MODELS], axis=1)
    nf_probs  = gate.predict_probs(nf_conf, nf_cube)
    nf_preds  = [LABELS[i] for i in nf_probs.argmax(axis=1)]

    # ------------------------------------------------------------------
    # 5. Collect all predictions
    # ------------------------------------------------------------------
    all_preds: Dict[str, List[str]] = {
        "logreg":      [LABELS[i] for i in prob_matrices["logreg"].argmax(axis=1)],
        "svm":         [LABELS[i] for i in prob_matrices["svm"].argmax(axis=1)],
        "tfidf":       [LABELS[i] for i in prob_matrices["tfidf"].argmax(axis=1)],
        "ensemble":    [LABELS[i] for i in prob_matrices["ensemble"].argmax(axis=1)],
        "meta_learner":[LABELS[i] for i in prob_matrices["meta_learner"].argmax(axis=1)],
        "pso":         pso_preds,
        "nsga2":       nsga2_preds,
        "neuro_fuzzy": nf_preds,
        "ts_meta":     ts_meta_preds,
    }

    from sklearn.metrics import f1_score
    method_f1 = {
        name: round(f1_score(y_true, preds, average="macro", zero_division=0), 6)
        for name, preds in all_preds.items()
    }
    print("\nMethod F1 scores:")
    for name, f1 in sorted(method_f1.items(), key=lambda x: -x[1]):
        print(f"  {name:<14}  {f1:.4f}")

    # ------------------------------------------------------------------
    # 6. Pairwise McNemar's tests
    # ------------------------------------------------------------------
    print("\nRunning pairwise McNemar's tests …")
    method_names = list(all_preds.keys())
    pairs_raw = []
    for a, b in combinations(method_names, 2):
        res = mcnemar(y_true, all_preds[a], all_preds[b])
        pairs_raw.append({
            "method_a": a,
            "method_b": b,
            "n01": res["n01"],
            "n10": res["n10"],
            "n_discordant": res["n"],
            "p_raw": round(res["p_value"], 8),
            "statistic": res["statistic"],
        })

    # Holm correction
    p_raws   = [r["p_raw"] for r in pairs_raw]
    p_holms  = holm_correct(p_raws)
    pairs = []
    for r, p_adj in zip(pairs_raw, p_holms):
        pairs.append({**r, "p_adj": round(p_adj, 8),
                      "significant": p_adj < ALPHA})

    sig_count = sum(1 for r in pairs if r["significant"])
    print(f"  {len(pairs)} pairs tested  |  {sig_count} significant (p_adj < {ALPHA})")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    output_data = {
        "n_test": len(y_true),
        "alpha": ALPHA,
        "n_comparisons": len(pairs),
        "correction": "Holm-Bonferroni",
        "method_f1": method_f1,
        "pairs": sorted(pairs, key=lambda r: r["p_adj"]),
    }
    json_path = out_dir / "ci_significance_tests.json"
    with open(json_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nSaved JSON → {json_path}")

    report = build_report(pairs, method_f1, len(y_true))
    md_path = out_dir / "ci_significance_tests.md"
    md_path.write_text(report)
    print(f"Saved Markdown → {md_path}")

    # ------------------------------------------------------------------
    # 8. Console table (significant pairs only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SIGNIFICANT PAIRS (p_adj < {ALPHA})  —  {sig_count}/{len(pairs)}")
    print("=" * 72)
    print(f"{'Method A':<15} {'Method B':<15} {'ΔF1':>7}  {'p_adj':>12}  Sig")
    print("-" * 72)
    for r in sorted(pairs, key=lambda r: r["p_adj"]):
        if not r["significant"]:
            break
        delta = method_f1[r["method_b"]] - method_f1[r["method_a"]]
        print(
            f"{r['method_a']:<15} {r['method_b']:<15} "
            f"{delta:>+7.4f}  {r['p_adj']:>12.4e}  "
            f"{significance_star(r['p_adj'])}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
