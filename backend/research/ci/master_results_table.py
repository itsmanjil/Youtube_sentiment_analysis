"""
Master Results Table
=====================
Consolidates every experiment result into a single thesis-ready comparison
table across all methods, datasets, and metrics.

Reads JSON files from results/ — no re-scoring needed.

Outputs
-------
    results/master_results_table.md    full thesis table (Markdown + LaTeX)
    results/master_results_table.json  machine-readable version

Usage
-----
    cd backend
    python research/ci/master_results_table.py --output results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/")
    parser.add_argument("--output",  default="results/")
    args = parser.parse_args()

    R = Path(args.results)
    O = Path(args.output)
    O.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all result files
    # ------------------------------------------------------------------
    ts_classical   = _load(R / "temperature_scaling.json")
    nf_classical   = _load(R / "neuro_fuzzy_gate.json")
    eg_classical   = _load(R / "entropy_gated_prediction.json")
    cac_classical  = _load(R / "coverage_accuracy_curve.json")
    moe_classical  = _load(R / "multi_objective_ensemble.json")
    pso_conv       = _load(R / "pso_convergence.json")

    route_a_master = _load(R / "route_a_benchmark_cpu_ci" / "master_summary.json")
    route_a_ts     = _load(R / "route_a_benchmark_cpu_ci" / "temperature_scaling.json")
    route_a_nf     = _load(R / "route_a_benchmark_cpu_ci" / "neuro_fuzzy_gate.json")
    route_a_eg     = _load(R / "route_a_benchmark_cpu_ci" / "entropy_gated_prediction.json")
    route_a_cac    = _load(R / "route_a_benchmark_cpu_ci" / "coverage_accuracy_curve.json")

    lines = [
        "# Master Results Table\n",
        "> Generated automatically from results/ JSON files.\n",

        # ----------------------------------------------------------------
        "## Part 1 — Classical Ensemble (Full Dataset: 165k test samples)\n",
        "### 1a. Per-Model Metrics\n",
        "| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | Temperature |",
        "|-------|----------|-----------|------------------|-------------|",
    ]

    if ts_classical:
        for r in ts_classical.get("models", []):
            lines.append(
                f"| {r['model']} | {r['macro_f1_before']:.4f} "
                f"| {r['ece_before']:.4f} | {r['ece_after']:.4f} "
                f"| {r['temperature']:.3f} |"
            )

    lines += [
        "\n### 1b. Ensemble Optimisation (NSGA-II Multi-Objective)\n",
        "| Method | Macro-F1 | ECE | Coverage@70% |",
        "|--------|----------|-----|--------------|",
    ]
    if moe_classical:
        kp = moe_classical.get("knee_point", {})
        t  = kp.get("test", {})
        w  = kp.get("weights", {})
        w_str = "  ".join(f"{m}={v:.3f}" for m, v in w.items())
        lines.append(
            f"| NSGA-II knee ({w_str}) "
            f"| {t.get('macro_f1','—'):.4f} "
            f"| {t.get('ece','—'):.4f} "
            f"| {t.get('coverage','—'):.4f} |"
        )
    if pso_conv:
        lines.append(
            f"| Single-obj PSO (best val) "
            f"| {pso_conv.get('best_val_f1','—')} | — | — |"
        )

    lines += [
        "\n### 1c. Neuro-Fuzzy Gating\n",
        "| Method | Macro-F1 | Accuracy | ECE | Brier |",
        "|--------|----------|----------|-----|-------|",
    ]
    if nf_classical:
        s = nf_classical.get("static_ensemble_test_metrics", {})
        t = nf_classical.get("test_metrics", {})
        d = nf_classical.get("delta_vs_static", {})
        lines += [
            f"| Uniform ensemble | {s.get('macro_f1','—'):.4f} "
            f"| {s.get('accuracy','—'):.4f} | {s.get('ece','—'):.4f} "
            f"| {s.get('brier','—'):.4f} |",
            f"| **Neuro-fuzzy gate** | **{t.get('macro_f1','—'):.4f}** "
            f"| {t.get('accuracy','—'):.4f} | **{t.get('ece','—'):.4f}** "
            f"| {t.get('brier','—'):.4f} |",
            f"| Δ (NF − uniform) | {d.get('macro_f1',0):+.4f} | — "
            f"| {d.get('ece',0):+.4f} | {d.get('brier',0):+.4f} |",
        ]

    lines += [
        "\n### 1d. Selective Prediction (Entropy-Gated)\n",
        f"Full ensemble (neuro-fuzzy mode)  "
        + (f"F1={eg_classical['full_ensemble']['macro_f1']:.4f}  AURC={eg_classical['aurc']:.4f}"
           if eg_classical else "— (not found)"),
        "",
        "| τ | Coverage | Accuracy | Macro-F1 |",
        "|---|----------|----------|----------|",
    ]
    if eg_classical:
        sweep = eg_classical.get("risk_coverage_sweep", [])
        step  = max(1, len(sweep) // 8)
        for r in sweep[::step]:
            if r.get("accuracy") is not None:
                lines.append(
                    f"| {r['threshold']:.2f} | {r['coverage']:.3f} "
                    f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
                )

    lines += [
        "\n### 1e. Coverage-Accuracy (AUCA)\n",
        "| Rank | Model | AUCA | AUC-F1 | Acc@100% |",
        "|------|-------|------|--------|----------|",
    ]
    if cac_classical:
        for rank, r in enumerate(cac_classical.get("summary", []), start=1):
            lines.append(
                f"| {rank} | {r['model']} | {r['auca']:.4f} "
                f"| {r['aucf1']:.4f} | {r['acc_full']:.4f} |"
            )

    # ----------------------------------------------------------------
    lines += [
        "\n---\n",
        "## Part 2 — Route A: DeBERTa-v3 + Classical (450-sample fine-tune, 180 test)\n",
        "> ⚠️  Small test set (n=180). Results are directional; full-scale fine-tuning pending GPU access.\n",
        "### 2a. Per-Model Metrics\n",
        "| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | Temperature |",
        "|-------|----------|-----------|------------------|-------------|",
    ]
    if route_a_ts:
        for r in route_a_ts.get("models", []):
            lines.append(
                f"| {r['model']} | {r['macro_f1']:.4f} "
                f"| {r['ece_before']:.4f} | {r['ece_after']:.4f} "
                f"| {r['temperature']:.3f} |"
            )

    lines += [
        "\n### 2b. Neuro-Fuzzy Gating\n",
        "| Method | Macro-F1 | Accuracy | ECE | Brier |",
        "|--------|----------|----------|-----|-------|",
    ]
    if route_a_nf:
        s = route_a_nf.get("static_ensemble_test_metrics", {})
        t = route_a_nf.get("test_metrics", {})
        d = route_a_nf.get("delta_vs_static", {})
        lines += [
            f"| Uniform ensemble | {s.get('macro_f1','—'):.4f} "
            f"| {s.get('accuracy','—'):.4f} | {s.get('ece','—'):.4f} "
            f"| {s.get('brier','—'):.4f} |",
            f"| **Neuro-fuzzy gate** | **{t.get('macro_f1','—'):.4f}** "
            f"| {t.get('accuracy','—'):.4f} | **{t.get('ece','—'):.4f}** "
            f"| {t.get('brier','—'):.4f} |",
            f"| Δ | {d.get('macro_f1',0):+.4f} | — "
            f"| {d.get('ece',0):+.4f} | {d.get('brier',0):+.4f} |",
        ]

    lines += [
        "\n### 2c. Coverage-Accuracy (AUCA)\n",
        "| Rank | Model | AUCA | AUC-F1 | Acc@100% |",
        "|------|-------|------|--------|----------|",
    ]
    if route_a_cac:
        for rank, r in enumerate(route_a_cac.get("summary", []), start=1):
            lines.append(
                f"| {rank} | {r['model']} | {r['auca']:.4f} "
                f"| {r['aucf1']:.4f} | {r['acc_full']:.4f} |"
            )

    # ----------------------------------------------------------------
    lines += [
        "\n---\n",
        "## Part 3 — Cross-System Comparison\n",
        "| System | Method | Macro-F1 | ECE | AUCA | Notes |",
        "|--------|--------|----------|-----|------|-------|",
    ]

    rows = []
    if ts_classical:
        for r in ts_classical["models"]:
            rows.append((
                "Classical (165k)", r["model"],
                r["macro_f1_before"], r["ece_before"],
                "—", "Full test set"
            ))
    if nf_classical:
        t = nf_classical["test_metrics"]
        rows.append(("Classical (165k)", "Neuro-fuzzy gate",
                     t["macro_f1"], t["ece"], "—", "Adaptive routing"))
    if moe_classical:
        kp = moe_classical["knee_point"]["test"]
        rows.append(("Classical (165k)", "NSGA-II ensemble",
                     kp["macro_f1"], kp["ece"], "—", "Pareto knee-point"))
    if route_a_nf:
        t = route_a_nf["test_metrics"]
        rows.append(("Route A (450 train)", "NF gate (DeBERTa+LogReg+SVM)",
                     t["macro_f1"], t["ece"], "—", "n=180 test, directional"))
    if route_a_cac:
        best = max(route_a_cac["summary"], key=lambda r: r["auca"])
        rows.append(("Route A (450 train)", f"Best AUCA ({best['model']})",
                     best["acc_full"], "—", best["auca"], "n=180 test"))

    for r in rows:
        f1  = f"{r[2]:.4f}" if isinstance(r[2], float) else r[2]
        ece = f"{r[3]:.4f}" if isinstance(r[3], float) else r[3]
        auc_val = f"{r[4]:.4f}" if isinstance(r[4], float) else r[4]
        lines.append(f"| {r[0]} | {r[1]} | {f1} | {ece} | {auc_val} | {r[5]} |")

    # LaTeX version
    latex = _to_latex(rows)

    content = "\n".join(lines)
    (O / "master_results_table.md").write_text(content)
    (O / "master_results_table.tex").write_text(latex)
    print(f"Saved → {O / 'master_results_table.md'}")
    print(f"Saved → {O / 'master_results_table.tex'}")

    # Print headline numbers
    print("\n" + "=" * 65)
    print("HEADLINE NUMBERS FOR THESIS")
    print("=" * 65)
    if ts_classical:
        best_classical = max(ts_classical["models"], key=lambda r: r["macro_f1_before"])
        print(f"Best classical model:  {best_classical['model']}  "
              f"F1={best_classical['macro_f1_before']:.4f}")
    if nf_classical:
        t = nf_classical["test_metrics"]
        s = nf_classical["static_ensemble_test_metrics"]
        print(f"Neuro-fuzzy gate:      F1={t['macro_f1']:.4f}  ECE={t['ece']:.4f}  "
              f"(ΔF1={t['macro_f1']-s['macro_f1']:+.4f}  ΔECE={t['ece']-s['ece']:+.4f})")
    if moe_classical:
        kp = moe_classical["knee_point"]["test"]
        print(f"NSGA-II ensemble:      F1={kp['macro_f1']:.4f}  "
              f"ECE={kp['ece']:.4f}  Coverage@70%={kp['coverage']:.4f}")
    if eg_classical:
        print(f"Entropy-gated AURC:    {eg_classical['aurc']:.4f}")
    if cac_classical:
        best_auca = max(cac_classical["summary"], key=lambda r: r["auca"])
        print(f"Best AUCA:             {best_auca['model']}  {best_auca['auca']:.4f}")
    if route_a_nf:
        t = route_a_nf["test_metrics"]
        print(f"Route A NF gate:       F1={t['macro_f1']:.4f}  ECE={t['ece']:.4f}  "
              f"(DeBERTa+LogReg+SVM, n=180 test)")
    print("=" * 65)


def _to_latex(rows) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-System Results Comparison}",
        r"\label{tab:master_results}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"System & Method & F1 & ECE & AUCA & Notes \\",
        r"\hline",
    ]
    for r in rows:
        f1  = f"{r[2]:.4f}" if isinstance(r[2], float) else str(r[2])
        ece = f"{r[3]:.4f}" if isinstance(r[3], float) else str(r[3])
        au  = f"{r[4]:.4f}" if isinstance(r[4], float) else str(r[4])
        lines.append(
            f"{r[0]} & {r[1]} & {f1} & {ece} & {au} & {r[5]} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
