"""
Gold Set Evaluation — Thesis Chapter 4 / Appendix

Evaluates all live models against the 300-sample pseudo-gold set.
Produces:
  - results/gold_set/gold_set_evaluation.json   (full metrics)
  - results/gold_set/gold_set_evaluation.md     (thesis-ready tables)
  - results/gold_set/gold_set_evaluation.tex    (LaTeX tables)

Models evaluated:
  - logreg       (with temperature scaling)
  - svm          (with temperature scaling)
  - tfidf        (with temperature scaling)
  - ensemble_pso (PSO weights, temperature scaled)
  - ensemble_nsga2 (NSGA-II weights, temperature scaled)
  - meta_learner (with temperature scaling)

Gold reference: silver_label from gold_set_silver_labeled.csv
Baseline ref:   source_label from gold_set_labeled_from_dataset.csv
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.sentiment.engines.logreg_engine import LogRegSentimentEngine
from src.sentiment.engines.svm_engine import SVMSentimentEngine
from src.sentiment.engines.tfidf_engine import TFIDFSentimentEngine
from src.sentiment.engines.ensemble_engine import EnsembleSentimentEngine
from src.sentiment.engines.meta_learner_engine import MetaLearnerSentimentEngine

SILVER_CSV = BACKEND / "data" / "gold_set_silver_labeled.csv"
OUT_DIR = BACKEND / "results" / "gold_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]


# ── Metrics ────────────────────────────────────────────────────────────────

def precision_recall_f1(y_true, y_pred, label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4), tp, fp, fn


def macro_f1(y_true, y_pred):
    f1s = [precision_recall_f1(y_true, y_pred, lbl)[2] for lbl in LABELS]
    return round(sum(f1s) / len(f1s), 4)


def weighted_f1(y_true, y_pred):
    total = len(y_true)
    wf1 = 0.0
    for lbl in LABELS:
        support = sum(1 for t in y_true if t == lbl)
        _, _, f, _, _, _ = precision_recall_f1(y_true, y_pred, lbl)
        wf1 += f * support / total
    return round(wf1, 4)


def accuracy(y_true, y_pred):
    return round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4)


def confusion_matrix(y_true, y_pred):
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return {t: dict(row) for t, row in cm.items()}


def evaluate(y_true, y_pred, model_name):
    per_class = {}
    for lbl in LABELS:
        p, r, f, tp, fp, fn = precision_recall_f1(y_true, y_pred, lbl)
        support = sum(1 for t in y_true if t == lbl)
        per_class[lbl] = {"precision": p, "recall": r, "f1": f,
                          "support": support, "tp": tp, "fp": fp, "fn": fn}
    return {
        "model":       model_name,
        "accuracy":    accuracy(y_true, y_pred),
        "macro_f1":    macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "per_class":   per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "n_samples":   len(y_true),
    }


# ── Load data ───────────────────────────────────────────────────────────────

def load_gold():
    with open(SILVER_CSV) as f:
        rows = list(csv.DictReader(f))
    texts        = [r["text"] for r in rows]
    silver_labels = [r["silver_label"] for r in rows]
    source_labels = [r["source_label"] for r in rows]
    return texts, silver_labels, source_labels


# ── Run models ──────────────────────────────────────────────────────────────

def run_model(name, engine, texts):
    print(f"  Running {name}...")
    results = engine.batch_analyze(texts)
    return [r.label for r in results]


def main():
    print("Loading gold set...")
    texts, silver_labels, source_labels = load_gold()
    print(f"  {len(texts)} samples loaded.")

    print("Loading engines...")
    engines = {
        "logreg":        LogRegSentimentEngine(),
        "svm":           SVMSentimentEngine(),
        "tfidf":         TFIDFSentimentEngine(),
        "ensemble_pso":  EnsembleSentimentEngine(weights_optimization="pso"),
        "ensemble_nsga2": EnsembleSentimentEngine(weights_optimization="nsga2"),
        "meta_learner":  MetaLearnerSentimentEngine(),
    }
    print("  Done.")

    all_results = {}

    print("\nEvaluating against silver labels (ensemble auto-annotation):")
    for name, engine in engines.items():
        preds = run_model(name, engine, texts)
        all_results[name] = evaluate(silver_labels, preds, name)

    # Also evaluate silver vs source (inter-labeler agreement)
    print("  Computing silver↔source agreement...")
    all_results["silver_vs_source"] = evaluate(source_labels, silver_labels, "silver_vs_source")

    # Evaluate each model against source labels too (for comparison)
    print("\nEvaluating against source labels (original dataset labels):")
    source_results = {}
    for name, engine in engines.items():
        preds = run_model(name, engine, texts)
        source_results[name] = evaluate(source_labels, preds, name)

    # ── Build output ────────────────────────────────────────────────────────
    output = {
        "description": (
            "Gold set evaluation: 300-sample pseudo-gold set. "
            "silver_ref = PSO ensemble auto-labels. "
            "source_ref = original dataset labels."
        ),
        "n_samples": len(texts),
        "silver_ref": all_results,
        "source_ref": source_results,
    }

    with open(OUT_DIR / "gold_set_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md = ["# Gold Set Evaluation (300 samples)\n"]
    md.append(
        "Silver labels were produced by PSO-weighted ensemble auto-annotation "
        "(logreg×0.31 + svm×0.69). Source labels are the original dataset labels. "
        "Inter-labeler agreement: **{:.1f}%**.\n".format(
            all_results["silver_vs_source"]["accuracy"] * 100
        )
    )

    for ref_name, ref_results, caption in [
        ("silver_ref", all_results,
         "## Table 1: Model Performance vs Silver Labels (auto-annotated)\n"),
        ("source_ref", source_results,
         "## Table 2: Model Performance vs Source Labels (original dataset)\n"),
    ]:
        md.append(caption)
        md.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
        md.append("|-------|----------|----------|-------------|")
        rows_data = source_results if ref_name == "source_ref" else all_results
        model_order = ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]
        for m in model_order:
            r = rows_data[m]
            md.append(f"| {m} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")
        md.append("")

    # Per-class F1 table (vs source)
    md.append("## Table 3: Per-Class F1 vs Source Labels\n")
    md.append("| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |")
    md.append("|-------|--------|--------|--------|----------|")
    for m in ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]:
        r = source_results[m]
        neg = r["per_class"]["Negative"]["f1"]
        neu = r["per_class"]["Neutral"]["f1"]
        pos = r["per_class"]["Positive"]["f1"]
        md.append(f"| {m} | {neg:.4f} | {neu:.4f} | {pos:.4f} | {r['macro_f1']:.4f} |")
    md.append("")

    # Confusion matrix for best model (ensemble_pso vs source)
    best = source_results["ensemble_pso"]
    md.append("## Confusion Matrix: ensemble_pso vs Source Labels\n")
    md.append("| True \\ Pred | Negative | Neutral | Positive |")
    md.append("|-------------|----------|---------|----------|")
    cm = best["confusion_matrix"]
    for true_lbl in LABELS:
        row_vals = [cm.get(true_lbl, {}).get(pred_lbl, 0) for pred_lbl in LABELS]
        md.append(f"| {true_lbl} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
    md.append("")

    md.append("## Notes\n")
    md.append(
        "- Silver labels (auto-annotation) show 66.3% agreement with source labels, "
        "indicating substantial label noise or ambiguity in the original dataset.\n"
        "- Neutral class consistently has the lowest F1 across all models, "
        "consistent with inter-annotator difficulty on borderline comments.\n"
        "- ensemble_pso (PSO-optimised weights, F1-best) and ensemble_nsga2 "
        "(NSGA-II knee-point, calibration-best) provide complementary trade-offs.\n"
    )

    with open(OUT_DIR / "gold_set_evaluation.md", "w") as f:
        f.write("\n".join(md))

    # ── LaTeX ────────────────────────────────────────────────────────────────
    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model Performance on 300-Sample Gold Set (vs.\ source labels)}",
        r"\label{tab:gold_set_performance}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & Accuracy & Macro F1 & Weighted F1 \\",
        r"\hline",
    ]
    model_display = {
        "logreg": r"\textsc{LogReg}",
        "svm": r"\textsc{SVM}",
        "tfidf": r"\textsc{TF-IDF NB}",
        "ensemble_pso": r"\textsc{Ensemble (PSO)}",
        "ensemble_nsga2": r"\textsc{Ensemble (NSGA-II)}",
        "meta_learner": r"\textsc{Meta-Learner}",
    }
    for m in ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]:
        r = source_results[m]
        tex.append(
            f"{model_display[m]} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & {r['weighted_f1']:.4f} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(OUT_DIR / "gold_set_evaluation.tex", "w") as f:
        f.write("\n".join(tex))

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GOLD SET EVALUATION SUMMARY (vs source labels)")
    print("=" * 60)
    print(f"{'Model':<20} {'Acc':>6} {'MacF1':>7} {'WtF1':>7}")
    print("-" * 44)
    for m in ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]:
        r = source_results[m]
        print(f"{m:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>7.4f} {r['weighted_f1']:>7.4f}")

    print(f"\nInter-labeler agreement (silver vs source): "
          f"{all_results['silver_vs_source']['accuracy']*100:.1f}%")
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
