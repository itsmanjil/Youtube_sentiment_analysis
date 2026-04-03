"""
Route A CI Analysis — DeBERTa + Classical Ensemble
====================================================

Runs all four CI analyses on a pre-computed probability cube that includes
transformer model probabilities alongside classical models.

Reads .npz cubes produced by research/transformers/export_prob_cube.py
so no live model scoring is needed — the transformer only runs once.

Analyses
--------
1. Temperature scaling       — per-model ECE / Brier before & after
2. Neuro-fuzzy gating        — adaptive per-sample ensemble routing
3. Entropy-gated prediction  — risk-coverage sweep + AURC
4. Coverage-accuracy curve   — all models head-to-head AUCA comparison

Usage
-----
    cd backend
    python research/ci/route_a_ci_analysis.py \\
        --val_cube  results/prob_cubes/route_a_benchmark_cpu_val_deberta_logreg_svm.npz \\
        --test_cube results/prob_cubes/route_a_benchmark_cpu_test_deberta_logreg_svm.npz \\
        --output    results/route_a_benchmark_cpu_ci/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from research.transformers.prob_cube_io import load_probability_cube
from research.evaluation.calibration import compute_calibration_metrics

# Pull reusable functions from existing CI modules
from research.ci.temperature_scaling import apply_temperature, fit_temperature
from research.ci.entropy_gated_prediction import (
    normalised_entropy,
    risk_coverage_sweep,
    compute_aurc,
    cascade_eval,
)
from research.ci.coverage_accuracy_curve import coverage_accuracy_sweep, auc
from research.ci.neuro_fuzzy_gate import (
    NeuroFuzzyGate,
    model_confidence,
    static_ensemble_probs,
    N_FUZZY_SETS,
    build_report as nf_build_report,
)

LABELS_DEFAULT = ["Positive", "Neutral", "Negative"]


# ---------------------------------------------------------------------------
# 1. Temperature scaling
# ---------------------------------------------------------------------------

def run_temperature_scaling(
    val_cube: np.ndarray,
    test_cube: np.ndarray,
    y_val: List[str],
    y_test: List[str],
    model_names: List[str],
    labels: List[str],
    out_dir: Path,
) -> List[dict]:
    print("\n[1/4] Temperature Scaling …")
    results = []
    for m, name in enumerate(model_names):
        T = fit_temperature(val_cube[m], y_val)
        test_cal = apply_temperature(test_cube[m], T)

        ece_b, brier_b = compute_calibration_metrics(y_test, test_cube[m],  labels=labels)
        ece_a, brier_a = compute_calibration_metrics(y_test, test_cal,      labels=labels)

        preds_b = [labels[i] for i in test_cube[m].argmax(axis=1)]
        f1_b    = f1_score(y_test, preds_b, average="macro", zero_division=0)

        row = {
            "model":              name,
            "temperature":        round(T, 4),
            "ece_before":         round(ece_b,   6),
            "ece_after":          round(ece_a,   6),
            "ece_reduction_pct":  round(100 * (ece_b - ece_a) / max(ece_b, 1e-10), 2),
            "brier_before":       round(brier_b, 6),
            "brier_after":        round(brier_a, 6),
            "macro_f1":           round(f1_b,    6),
        }
        results.append(row)
        print(f"  {name:<14} T={T:.3f}  ECE {ece_b:.4f}→{ece_a:.4f} "
              f"({row['ece_reduction_pct']:+.1f}%)  F1={f1_b:.4f}")

    _save(out_dir, "temperature_scaling.json", {"models": results})
    _save_md(out_dir, "temperature_scaling.md", _ts_md(results))
    return results


def _ts_md(results: List[dict]) -> str:
    lines = [
        "# Temperature Scaling — Route A (DeBERTa + Classical)\n",
        "| Model | T | ECE Before | ECE After | Δ ECE% | Macro-F1 |",
        "|-------|---|------------|-----------|--------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['temperature']:.3f} "
            f"| {r['ece_before']:.4f} | {r['ece_after']:.4f} "
            f"| {r['ece_reduction_pct']:+.1f}% | {r['macro_f1']:.4f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Neuro-Fuzzy Gate
# ---------------------------------------------------------------------------

def run_neuro_fuzzy(
    val_cube: np.ndarray,
    test_cube: np.ndarray,
    y_val: List[str],
    y_test: List[str],
    model_names: List[str],
    labels: List[str],
    out_dir: Path,
    maxiter: int = 200,
) -> dict:
    print("\n[2/4] Neuro-Fuzzy Gating …")

    val_conf  = np.stack([model_confidence(val_cube[m])  for m in range(len(model_names))], axis=1)
    test_conf = np.stack([model_confidence(test_cube[m]) for m in range(len(model_names))], axis=1)

    # Static uniform baseline
    uniform_w = np.ones(len(model_names)) / len(model_names)
    static_probs = static_ensemble_probs(test_cube, uniform_w)
    static_preds = [labels[i] for i in static_probs.argmax(axis=1)]
    static_f1    = f1_score(y_test, static_preds, average="macro", zero_division=0)
    static_acc   = accuracy_score(y_test, static_preds)
    static_ece, static_brier = compute_calibration_metrics(y_test, static_probs, labels=labels)

    static_metrics = {
        "macro_f1": round(static_f1, 6), "accuracy": round(static_acc, 6),
        "ece": round(static_ece, 6),     "brier":    round(static_brier, 6),
    }
    print(f"  Static uniform:  F1={static_f1:.4f}  ECE={static_ece:.4f}")

    # Fit gate
    gate = NeuroFuzzyGate(n_models=len(model_names), n_sets=N_FUZZY_SETS)
    fit_info = gate.fit(val_conf, val_cube, y_val, maxiter=maxiter)

    # Test evaluation
    test_probs = gate.predict_probs(test_conf, test_cube)
    test_preds = [labels[i] for i in test_probs.argmax(axis=1)]
    test_f1    = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_acc   = accuracy_score(y_test, test_preds)
    test_ece, test_brier = compute_calibration_metrics(y_test, test_probs, labels=labels)

    test_metrics = {
        "macro_f1": round(test_f1, 6), "accuracy": round(test_acc, 6),
        "ece": round(test_ece, 6),     "brier":    round(test_brier, 6),
    }
    print(f"  Neuro-fuzzy:     F1={test_f1:.4f}  ECE={test_ece:.4f}  "
          f"(ΔF1={test_f1-static_f1:+.4f}  ΔECE={test_ece-static_ece:+.4f})")

    # Val metrics
    val_probs = gate.predict_probs(val_conf, val_cube)
    val_preds = [labels[i] for i in val_probs.argmax(axis=1)]
    val_f1  = f1_score(y_val, val_preds, average="macro", zero_division=0)
    val_ece, val_brier = compute_calibration_metrics(y_val, val_probs, labels=labels)

    output = {
        "architecture": {
            "n_models": len(model_names), "model_names": model_names,
            "n_fuzzy_sets": N_FUZZY_SETS, "n_parameters": gate.n_params,
            "reference": "ANFIS (Jang, 1993)",
        },
        "training": fit_info,
        "learned_mfs": gate.describe_gates(model_names),
        "val_metrics": {"macro_f1": round(val_f1, 6), "ece": round(val_ece, 6), "brier": round(val_brier, 6)},
        "test_metrics": test_metrics,
        "static_ensemble_test_metrics": static_metrics,
        "delta_vs_static": {
            "macro_f1": round(test_f1 - static_f1, 6),
            "ece":      round(test_ece - static_ece, 6),
            "brier":    round(test_brier - static_brier, 6),
        },
    }
    _save(out_dir, "neuro_fuzzy_gate.json", output)
    report = nf_build_report(gate, fit_info, model_names,
                              output["val_metrics"], test_metrics, static_metrics)
    _save_md(out_dir, "neuro_fuzzy_gate.md", report)
    return output


# ---------------------------------------------------------------------------
# 3. Entropy-Gated Prediction
# ---------------------------------------------------------------------------

def run_entropy_gated(
    val_cube: np.ndarray,
    test_cube: np.ndarray,
    y_val: List[str],
    y_test: List[str],
    model_names: List[str],
    labels: List[str],
    out_dir: Path,
    nf_weights_json: Path | None = None,
    n_thresholds: int = 25,
) -> dict:
    print("\n[3/4] Entropy-Gated Selective Prediction …")

    # Use neuro-fuzzy gate probs if available, else uniform ensemble
    if nf_weights_json and nf_weights_json.exists():
        nf_data = json.loads(nf_weights_json.read_text())
        gate = NeuroFuzzyGate(n_models=len(model_names), n_sets=N_FUZZY_SETS)
        set_order = {"Low": 0, "Medium": 1, "High": 2}
        model_order = {m: i for i, m in enumerate(model_names)}
        for row in nf_data.get("learned_mfs", []):
            m = model_order.get(row["model"])
            k = set_order.get(row["fuzzy_set"])
            if m is None or k is None:
                continue
            base = m * N_FUZZY_SETS * 3 + k * 3
            gate.theta[base + 0] = row["center"]
            gate.theta[base + 1] = math.log(max(row["width"], 1e-6))
            gate.theta[base + 2] = row["alpha"]
        test_conf = np.stack([model_confidence(test_cube[m]) for m in range(len(model_names))], axis=1)
        ens_probs = gate.predict_probs(test_conf, test_cube)
        mode = "neuro_fuzzy"
    else:
        uniform_w = np.ones(len(model_names)) / len(model_names)
        ens_probs = static_ensemble_probs(test_cube, uniform_w)
        mode = "uniform"

    entropy   = normalised_entropy(ens_probs)
    ens_preds = [labels[i] for i in ens_probs.argmax(axis=1)]
    stage2    = [labels[i] for i in test_cube[0].argmax(axis=1)]   # model_0 as cascade

    full_acc = accuracy_score(y_test, ens_preds)
    full_f1  = f1_score(y_test, ens_preds, average="macro", zero_division=0)
    print(f"  Ensemble ({mode}):  Acc={full_acc:.4f}  F1={full_f1:.4f}")
    print(f"  Entropy: mean={entropy.mean():.3f}  "
          f"p25={np.percentile(entropy,25):.3f}  p75={np.percentile(entropy,75):.3f}")

    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)[1:]
    sweep = risk_coverage_sweep(y_test, entropy, ens_preds, thresholds)
    aurc  = compute_aurc(sweep)
    print(f"  AURC = {aurc:.4f}")

    cascade_taus = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    cascade_results = [
        cascade_eval(y_test, entropy, ens_preds, stage2, tau)
        for tau in cascade_taus
    ]

    output = {
        "ensemble_mode": mode,
        "model_names": model_names,
        "n_samples": len(y_test),
        "full_ensemble": {"accuracy": round(full_acc, 6), "macro_f1": round(full_f1, 6)},
        "entropy_stats": {
            "mean": round(float(entropy.mean()), 4),
            "p25":  round(float(np.percentile(entropy, 25)), 4),
            "p75":  round(float(np.percentile(entropy, 75)), 4),
            "max":  round(float(entropy.max()), 4),
        },
        "aurc": round(aurc, 6),
        "risk_coverage_sweep": sweep,
        "cascade_results": cascade_results,
    }
    _save(out_dir, "entropy_gated_prediction.json", output)

    # Compact markdown
    lines = [
        "# Entropy-Gated Selective Prediction — Route A\n",
        f"Ensemble mode: **{mode}** | Models: {', '.join(model_names)}\n",
        f"Full ensemble F1 = **{full_f1:.4f}** | AURC = **{aurc:.4f}**\n",
        "## Risk-Coverage Sweep\n",
        "| τ | Coverage | Accuracy | Macro-F1 | Abstain% |",
        "|---|----------|----------|----------|----------|",
    ]
    step = max(1, len(sweep) // 12)
    for r in sweep[::step]:
        if r["accuracy"] is not None:
            lines.append(
                f"| {r['threshold']:.2f} | {r['coverage']:.3f} "
                f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
                f"| {r['abstain_rate']:.3f} |"
            )
    lines += [
        "\n## Cascade (uncertain → first model)\n",
        f"| τ | Ensemble% | Fallback% | Accuracy | Macro-F1 |",
        "|---|-----------|-----------|----------|----------|",
    ]
    for r in cascade_results:
        lines.append(
            f"| {r['tau']:.2f} | {r['ensemble_coverage']:.3f} "
            f"| {r['stage2_coverage']:.3f} "
            f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
        )
    _save_md(out_dir, "entropy_gated_prediction.md", "\n".join(lines))
    return output


# ---------------------------------------------------------------------------
# 4. Coverage-Accuracy Curve
# ---------------------------------------------------------------------------

def run_coverage_accuracy(
    test_cube: np.ndarray,
    y_test: List[str],
    model_names: List[str],
    labels: List[str],
    out_dir: Path,
    nf_weights_json: Path | None = None,
    n_points: int = 50,
) -> dict:
    print("\n[4/4] Coverage-Accuracy Curve …")

    all_probs: dict[str, np.ndarray] = {}
    for m, name in enumerate(model_names):
        all_probs[name] = test_cube[m]

    # Add uniform ensemble
    uniform_w = np.ones(len(model_names)) / len(model_names)
    all_probs["ensemble_uniform"] = static_ensemble_probs(test_cube, uniform_w)

    # Add neuro-fuzzy gate if available
    if nf_weights_json and nf_weights_json.exists():
        nf_data = json.loads(nf_weights_json.read_text())
        gate = NeuroFuzzyGate(n_models=len(model_names), n_sets=N_FUZZY_SETS)
        set_order = {"Low": 0, "Medium": 1, "High": 2}
        model_order = {m: i for i, m in enumerate(model_names)}
        for row in nf_data.get("learned_mfs", []):
            mi = model_order.get(row["model"])
            k  = set_order.get(row["fuzzy_set"])
            if mi is None or k is None:
                continue
            base = mi * N_FUZZY_SETS * 3 + k * 3
            gate.theta[base + 0] = row["center"]
            gate.theta[base + 1] = math.log(max(row["width"], 1e-6))
            gate.theta[base + 2] = row["alpha"]
        test_conf = np.stack([model_confidence(test_cube[m]) for m in range(len(model_names))], axis=1)
        all_probs["neuro_fuzzy"] = gate.predict_probs(test_conf, test_cube)

    summary = []
    curves  = {}
    for name, probs in all_probs.items():
        sweep = coverage_accuracy_sweep(y_test, probs, n_points=n_points)
        auca  = auc(sweep, "accuracy")
        aucf1 = auc(sweep, "macro_f1")
        acc_full = sweep[-1]["accuracy"]
        f1_full  = sweep[-1]["macro_f1"]
        summary.append({
            "model": name, "auca": round(auca, 4), "aucf1": round(aucf1, 4),
            "acc_full": round(acc_full, 4), "f1_full": round(f1_full, 4),
        })
        curves[name] = sweep
        print(f"  {name:<20}  AUCA={auca:.4f}  F1@100%={f1_full:.4f}")

    output = {
        "n_samples": len(y_test),
        "model_names": list(all_probs.keys()),
        "summary": sorted(summary, key=lambda r: -r["auca"]),
        "curves": curves,
    }
    _save(out_dir, "coverage_accuracy_curve.json", output)

    # Markdown table
    ranked = sorted(summary, key=lambda r: -r["auca"])
    lines = [
        "# Coverage-Accuracy Curve — Route A\n",
        "| Rank | Model | AUCA | AUC-F1 | Acc@10% | Acc@100% |",
        "|------|-------|------|--------|---------|----------|",
    ]
    for rank, r in enumerate(ranked, start=1):
        sw = curves[r["model"]]
        acc10 = min(sw, key=lambda x: abs(x["coverage"] - 0.10))["accuracy"]
        lines.append(
            f"| {rank} | {r['model']} | {r['auca']:.4f} | {r['aucf1']:.4f} "
            f"| {acc10:.4f} | {r['acc_full']:.4f} |"
        )
    _save_md(out_dir, "coverage_accuracy_curve.md", "\n".join(lines))
    return output


# ---------------------------------------------------------------------------
# Master summary table
# ---------------------------------------------------------------------------

def build_master_table(
    ts_results: List[dict],
    nf_output: dict,
    eg_output: dict,
    cac_output: dict,
    model_names: List[str],
    out_dir: Path,
) -> None:
    print("\n[+] Building master summary table …")

    lines = [
        "# Route A CI Results — Master Summary\n",
        f"**Models in ensemble**: {', '.join(model_names)}\n",
        "## Model-Level Metrics (Test Set)\n",
        "| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | T |",
        "|-------|----------|-----------|------------------|---|",
    ]
    for r in ts_results:
        lines.append(
            f"| {r['model']} | {r['macro_f1']:.4f} "
            f"| {r['ece_before']:.4f} | {r['ece_after']:.4f} "
            f"| {r['temperature']:.3f} |"
        )

    static = nf_output["static_ensemble_test_metrics"]
    nf_test = nf_output["test_metrics"]
    ens_eg  = eg_output["full_ensemble"]

    lines += [
        "\n## Ensemble-Level Metrics (Test Set)\n",
        "| Method | Macro-F1 | Accuracy | ECE | Brier |",
        "|--------|----------|----------|-----|-------|",
        f"| Uniform ensemble | {static['macro_f1']:.4f} | {static['accuracy']:.4f} "
        f"| {static['ece']:.4f} | {static['brier']:.4f} |",
        f"| Neuro-fuzzy gate | **{nf_test['macro_f1']:.4f}** | {nf_test['accuracy']:.4f} "
        f"| **{nf_test['ece']:.4f}** | {nf_test['brier']:.4f} |",
        f"| Entropy-gated ({eg_output['ensemble_mode']}) | {ens_eg['macro_f1']:.4f} "
        f"| {ens_eg['accuracy']:.4f} | — | — |",
        f"\n## Selective Prediction (AURC / AUCA)\n",
        f"Entropy-gated AURC = **{eg_output['aurc']:.4f}**\n",
        "| Model | AUCA | AUC-F1 | Acc@100% |",
        "|-------|------|--------|----------|",
    ]
    for r in cac_output["summary"]:
        lines.append(
            f"| {r['model']} | {r['auca']:.4f} | {r['aucf1']:.4f} | {r['acc_full']:.4f} |"
        )

    lines += [
        "\n## NF Gate — Learned MF Parameters\n",
        "| Model | Fuzzy Set | Center | Width | Alpha |",
        "|-------|-----------|--------|-------|-------|",
    ]
    for row in nf_output.get("learned_mfs", []):
        lines.append(
            f"| {row['model']} | {row['fuzzy_set']} "
            f"| {row['center']:.3f} | {row['width']:.3f} | {row['alpha']:+.3f} |"
        )

    content = "\n".join(lines)
    _save_md(out_dir, "master_summary.md", content)
    _save(out_dir, "master_summary.json", {
        "temperature_scaling": ts_results,
        "neuro_fuzzy": {
            "static_metrics": static,
            "nf_metrics": nf_test,
            "delta": nf_output["delta_vs_static"],
        },
        "entropy_gated": {
            "full_f1": ens_eg["macro_f1"],
            "aurc": eg_output["aurc"],
        },
        "coverage_accuracy": cac_output["summary"],
    })
    print(f"  Saved master_summary.md + master_summary.json → {out_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(out_dir: Path, name: str, data: dict) -> None:
    path = out_dir / name
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  → {path}")


def _save_md(out_dir: Path, name: str, content: str) -> None:
    path = out_dir / name
    path.write_text(content)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Route A CI Analysis")
    parser.add_argument(
        "--val_cube",
        default="results/prob_cubes/route_a_benchmark_cpu_val_deberta_logreg_svm.npz",
    )
    parser.add_argument(
        "--test_cube",
        default="results/prob_cubes/route_a_benchmark_cpu_test_deberta_logreg_svm.npz",
    )
    parser.add_argument("--output",   default="results/route_a_benchmark_cpu_ci/")
    parser.add_argument("--maxiter",  type=int, default=200)
    parser.add_argument("--points",   type=int, default=50)
    parser.add_argument("--thresholds", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load cubes
    # ------------------------------------------------------------------
    print(f"Loading val  cube: {args.val_cube}")
    val_bundle  = load_probability_cube(args.val_cube)
    print(f"Loading test cube: {args.test_cube}")
    test_bundle = load_probability_cube(args.test_cube)

    model_names = list(val_bundle.model_names)
    labels      = list(val_bundle.labels)
    y_val       = list(val_bundle.y_true)
    y_test      = list(test_bundle.y_true)
    val_cube    = np.asarray(val_bundle.prob_cube,  dtype=np.float64)
    test_cube   = np.asarray(test_bundle.prob_cube, dtype=np.float64)

    print(f"\nModels : {model_names}")
    print(f"Labels : {labels}")
    print(f"Val    : {len(y_val)} samples")
    print(f"Test   : {len(y_test)} samples")
    print(f"Output : {out_dir}")

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    ts = run_temperature_scaling(
        val_cube, test_cube, y_val, y_test, model_names, labels, out_dir
    )

    nf = run_neuro_fuzzy(
        val_cube, test_cube, y_val, y_test, model_names, labels, out_dir,
        maxiter=args.maxiter,
    )
    nf_json = out_dir / "neuro_fuzzy_gate.json"

    eg = run_entropy_gated(
        val_cube, test_cube, y_val, y_test, model_names, labels, out_dir,
        nf_weights_json=nf_json, n_thresholds=args.thresholds,
    )

    cac = run_coverage_accuracy(
        test_cube, y_test, model_names, labels, out_dir,
        nf_weights_json=nf_json, n_points=args.points,
    )

    build_master_table(ts, nf, eg, cac, model_names, out_dir)

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------
    nf_test  = nf["test_metrics"]
    static   = nf["static_ensemble_test_metrics"]
    best_cac = max(cac["summary"], key=lambda r: r["auca"])

    print("\n" + "=" * 65)
    print("ROUTE A CI ANALYSIS — SUMMARY")
    print("=" * 65)
    print(f"Models: {', '.join(model_names)}")
    print(f"Test samples: {len(y_test)}")
    print()
    print(f"Uniform ensemble    F1={static['macro_f1']:.4f}  ECE={static['ece']:.4f}")
    print(f"Neuro-fuzzy gate    F1={nf_test['macro_f1']:.4f}  "
          f"ECE={nf_test['ece']:.4f}  "
          f"(ΔF1={nf['delta_vs_static']['macro_f1']:+.4f}  "
          f"ΔECE={nf['delta_vs_static']['ece']:+.4f})")
    print(f"Entropy AURC        {eg['aurc']:.4f}")
    print(f"Best AUCA           {best_cac['model']} = {best_cac['auca']:.4f}")
    print("=" * 65)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
