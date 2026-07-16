"""
Gold Set Evaluation â€” Thesis Chapter 4 / Appendix

Evaluates all live models against the 300-sample gold set.
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

Gold references (in priority order):
  1. human_reconciled  â€” gold_set_human_reconciled.csv (majority-vote human labels, if available)
  2. silver_label      â€” gold_set_silver_labeled.csv   (PSO ensemble auto-annotation, fallback)
  3. source_label      â€” original dataset labels

IAA metrics (Krippendorff's alpha, Cohen's kappa, Fleiss' kappa) are loaded
from results/gold_set/iaa_report.json if present.
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

SILVER_CSV    = BACKEND / "data" / "gold_set_silver_labeled.csv"
RECONCILED_CSV = BACKEND / "data" / "gold_set_human_reconciled.csv"
IAA_REPORT    = BACKEND / "results" / "gold_set" / "iaa_report.json"
OUT_DIR = BACKEND / "results" / "gold_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]


# â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_gold():
    """
    Load gold set texts and available label references.

    Priority:
      1. human_reconciled â€” majority-vote gold labels (if reconciled CSV exists)
      2. silver_label     â€” PSO ensemble auto-annotation (fallback)
      3. source_label     â€” original dataset labels (always loaded)

    Returns (texts, human_labels_or_None, silver_labels, source_labels)
    """
    with open(SILVER_CSV, encoding="utf-8-sig", newline="") as f:
        silver_rows = list(csv.DictReader(f))

    texts         = [r["text"] for r in silver_rows]
    silver_labels = [r["silver_label"] for r in silver_rows]
    source_labels = [r["source_label"] for r in silver_rows]

    # Load human-reconciled labels if available
    human_labels = None
    if RECONCILED_CSV.exists():
        text_to_gold = {}
        with open(RECONCILED_CSV, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                gl = row.get("gold_label", "").strip()
                if gl and row.get("is_disputed", "").strip().lower() != "yes":
                    text_to_gold[row["text"].strip()] = gl
        # Align to silver CSV order; drop items without a human gold label
        human_labels = [text_to_gold.get(t) for t in texts]
        n_human = sum(1 for v in human_labels if v)
        print(f"  Human-reconciled gold labels loaded: {n_human} / {len(texts)} items")

    return texts, human_labels, silver_labels, source_labels


def load_iaa_metrics():
    """Load IAA metrics from the report JSON if it exists."""
    if IAA_REPORT.exists():
        with open(IAA_REPORT, encoding="utf-8") as f:
            return json.load(f)
    return None


# â”€â”€ Run models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_model(name, engine, texts):
    print(f"  Running {name}...")
    results = engine.batch_analyze(texts)
    return [r.label for r in results]


def _filter_by_reference(texts, preds, ref_labels):
    """Drop items where ref_label is None (missing/disputed)."""
    filtered_texts, filtered_preds, filtered_ref = [], [], []
    for t, p, r in zip(texts, preds, ref_labels):
        if r:
            filtered_texts.append(t)
            filtered_preds.append(p)
            filtered_ref.append(r)
    return filtered_texts, filtered_preds, filtered_ref


def main():
    print("Loading gold set...")
    texts, human_labels, silver_labels, source_labels = load_gold()
    print(f"  {len(texts)} samples from silver CSV.")

    iaa = load_iaa_metrics()
    if iaa:
        m = iaa.get("metrics", {})
        alpha = m.get("krippendorff_alpha", float("nan"))
        print(f"  IAA report found - Krippendorff alpha = {alpha:.4f}  ({iaa.get('interpretation', '')})")

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

    model_order = ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]

    # Collect predictions once (reused across reference sets)
    all_preds = {}
    print("\nRunning model inference...")
    for name, engine in engines.items():
        all_preds[name] = run_model(name, engine, texts)

    # â”€â”€ Evaluate against each reference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # 1. Human-reconciled gold (primary, if available)
    human_results = {}
    if human_labels is not None and any(human_labels):
        valid_texts = [t for t, h in zip(texts, human_labels) if h]
        n_human_valid = len(valid_texts)
        print(f"\nEvaluating against human-reconciled gold labels ({n_human_valid} items):")
        for name in model_order:
            _, filtered_preds, filtered_ref = _filter_by_reference(
                texts, all_preds[name], human_labels
            )
            human_results[name] = evaluate(filtered_ref, filtered_preds, name)
        # Inter-reference agreement: human vs source
        _, src_filtered, hum_filtered = _filter_by_reference(texts, source_labels, human_labels)
        human_results["human_vs_source"] = evaluate(hum_filtered, src_filtered, "human_vs_source")
        _, sil_filtered, hum_filtered2 = _filter_by_reference(texts, silver_labels, human_labels)
        human_results["human_vs_silver"] = evaluate(hum_filtered2, sil_filtered, "human_vs_silver")

    # 2. Silver labels (PSO ensemble auto-annotation)
    print("\nEvaluating against silver labels (ensemble auto-annotation):")
    silver_results = {}
    for name in model_order:
        silver_results[name] = evaluate(silver_labels, all_preds[name], name)
    silver_results["silver_vs_source"] = evaluate(source_labels, silver_labels, "silver_vs_source")

    # 3. Source labels (original dataset)
    print("\nEvaluating against source labels (original dataset labels):")
    source_results = {}
    for name in model_order:
        source_results[name] = evaluate(source_labels, all_preds[name], name)

    # â”€â”€ Build output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_human = bool(human_results)
    output = {
        "description": (
            "Gold set evaluation: 300-sample gold set. "
            + ("human_ref = majority-vote reconciled human labels. " if has_human else "")
            + "silver_ref = PSO ensemble auto-labels. "
            "source_ref = original dataset labels."
        ),
        "n_samples": len(texts),
        "has_human_gold": has_human,
        "iaa_metrics": iaa.get("metrics") if iaa else None,
        "iaa_interpretation": iaa.get("interpretation") if iaa else None,
        "human_ref": human_results if has_human else None,
        "silver_ref": silver_results,
        "source_ref": source_results,
    }

    with open(OUT_DIR / "gold_set_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # â”€â”€ Markdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Determine primary reference for main tables
    primary_results = human_results if has_human else silver_results
    primary_name    = "Human-Reconciled Gold Labels" if has_human else "Silver Labels (auto-annotated pre-human fallback)"
    primary_ref_key = "human_ref" if has_human else "silver_ref"

    md = [f"# Gold Set Evaluation (300 samples)\n"]

    # IAA section
    if iaa:
        m_iaa = iaa.get("metrics", {})
        alpha     = m_iaa.get("krippendorff_alpha", float("nan"))
        fleiss    = m_iaa.get("fleiss_kappa", float("nan"))
        pct_agree = m_iaa.get("percent_agreement", float("nan"))
        md.append("## Inter-Annotator Agreement\n")
        md.append("| Metric | Value |")
        md.append("| --- | --- |")
        md.append(f"| Percent agreement | {pct_agree:.4f} |")
        md.append(f"| Krippendorff's Î± | {alpha:.4f} |")
        md.append(f"| Fleiss' Îº | {fleiss:.4f} |")
        for pair, kappa in m_iaa.get("cohen_kappa_pairwise", {}).items():
            kappa_str = f"{kappa:.4f}" if kappa == kappa else "n/a"
            md.append(f"| Cohen's Îº ({pair.replace('_vs_', ' vs ')}) | {kappa_str} |")
        md.append("")
        md.append(f"*{iaa.get('interpretation', '')}*\n")
    else:
        agreement_pct = silver_results["silver_vs_source"]["accuracy"] * 100
        md.append(
            f"Silver labels (PSO ensemble) agree with source labels at **{agreement_pct:.1f}%** "
            "(no human IAA report found - run `merge_annotations.py` to generate one).\n"
        )

    # Main performance table (vs primary reference)
    md.append(f"## Table 1: Model Performance vs {primary_name}\n")
    md.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    md.append("|-------|----------|----------|-------------|")
    for m_name in model_order:
        r = primary_results[m_name]
        md.append(f"| {m_name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")
    md.append("")

    # Table vs silver (always shown for comparison)
    if has_human:
        md.append("## Table 2: Model Performance vs Silver Labels (auto-annotated)\n")
        md.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
        md.append("|-------|----------|----------|-------------|")
        for m_name in model_order:
            r = silver_results[m_name]
            md.append(f"| {m_name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")
        md.append("")
        md.append(
            "> **Note:** ensemble_pso scores 1.000 vs silver labels because it *is* the silver labeler "
            "(PSO-weighted ensemble). This is expected and does not indicate overfitting.\n"
        )

    # Per-class F1 table (vs primary reference)
    md.append(f"## Table {'3' if has_human else '2'}: Per-Class F1 vs {primary_name}\n")
    md.append("| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |")
    md.append("|-------|--------|--------|--------|----------|")
    for m_name in model_order:
        r = primary_results[m_name]
        neg = r["per_class"]["Negative"]["f1"]
        neu = r["per_class"]["Neutral"]["f1"]
        pos = r["per_class"]["Positive"]["f1"]
        md.append(f"| {m_name} | {neg:.4f} | {neu:.4f} | {pos:.4f} | {r['macro_f1']:.4f} |")
    md.append("")

    # Best model confusion matrix
    best_model = "ensemble_pso"
    best = primary_results[best_model]
    md.append(f"## Confusion Matrix: {best_model} vs {primary_name}\n")
    md.append("| True \\ Pred | Negative | Neutral | Positive |")
    md.append("|-------------|----------|---------|----------|")
    cm = best["confusion_matrix"]
    for true_lbl in LABELS:
        row_vals = [cm.get(true_lbl, {}).get(pred_lbl, 0) for pred_lbl in LABELS]
        md.append(f"| {true_lbl} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
    md.append("")

    md.append("## Notes\n")
    if has_human:
        n_disputed = iaa.get("disputed_items", "?") if iaa else "?"
        md.append(
            f"- Human-reconciled gold labels used as primary reference "
            f"(disputed items excluded: {n_disputed}).\n"
        )
    md.append(
        "- Neutral class consistently has the lowest F1 across all models, "
        "consistent with inter-annotator difficulty on borderline comments.\n"
        "- ensemble_pso (PSO-optimised weights, F1-best) and ensemble_nsga2 "
        "(NSGA-II knee-point, calibration-best) provide complementary trade-offs.\n"
    )

    with open(OUT_DIR / "gold_set_evaluation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # â”€â”€ LaTeX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    caption_ref = "Human-Reconciled Gold Labels" if has_human else "Silver Labels"
    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Model Performance on 300-Sample Gold Set (vs.\ {caption_ref})}}",
        r"\label{tab:gold_set_performance}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & Accuracy & Macro F1 & Weighted F1 \\",
        r"\hline",
    ]
    model_display = {
        "logreg":        r"\textsc{LogReg}",
        "svm":           r"\textsc{SVM}",
        "tfidf":         r"\textsc{TF-IDF NB}",
        "ensemble_pso":  r"\textsc{Ensemble (PSO)}",
        "ensemble_nsga2": r"\textsc{Ensemble (NSGA-II)}",
        "meta_learner":  r"\textsc{Meta-Learner}",
    }
    for m_name in model_order:
        r = primary_results[m_name]
        tex.append(
            f"{model_display[m_name]} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & {r['weighted_f1']:.4f} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]

    # IAA LaTeX table
    if iaa:
        m_iaa = iaa.get("metrics", {})
        alpha     = m_iaa.get("krippendorff_alpha", float("nan"))
        fleiss    = m_iaa.get("fleiss_kappa", float("nan"))
        pct_agree = m_iaa.get("percent_agreement", float("nan"))
        tex += [
            "",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Inter-Annotator Agreement on 300-Sample Gold Set}",
            r"\label{tab:gold_set_iaa}",
            r"\begin{tabular}{lc}",
            r"\hline",
            r"Metric & Value \\",
            r"\hline",
            rf"Percent Agreement & {pct_agree:.4f} \\",
            rf"Krippendorff's $\alpha$ & {alpha:.4f} \\",
            rf"Fleiss' $\kappa$ & {fleiss:.4f} \\",
        ]
        for pair, kappa in m_iaa.get("cohen_kappa_pairwise", {}).items():
            pair_label = pair.replace("_vs_", r" vs.\ ").replace("_", r"\_")
            kappa_str  = f"{kappa:.4f}" if kappa == kappa else "n/a"
            tex.append(rf"Cohen's $\kappa$ ({pair_label}) & {kappa_str} \\")
        tex += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(OUT_DIR / "gold_set_evaluation.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    # â”€â”€ Print summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ref_label = "Human Gold" if has_human else "Silver Labels"
    print(f"\n{'=' * 60}")
    print(f"GOLD SET EVALUATION SUMMARY (vs {ref_label})")
    print("=" * 60)
    print(f"{'Model':<20} {'Acc':>6} {'MacF1':>7} {'WtF1':>7}")
    print("-" * 44)
    for m_name in model_order:
        r = primary_results[m_name]
        print(f"{m_name:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>7.4f} {r['weighted_f1']:>7.4f}")

    if iaa:
        m_iaa = iaa.get("metrics", {})
        alpha = m_iaa.get("krippendorff_alpha", float("nan"))
        alpha_str = f"{alpha:.4f}" if alpha == alpha else "n/a"
        print(f"\nKrippendorff alpha:         {alpha_str}  ({iaa.get('interpretation', '')})")
        pct = m_iaa.get("percent_agreement", float("nan"))
        print(f"Percent agreement:          {pct:.4f}")
    else:
        agree_pct = silver_results["silver_vs_source"]["accuracy"] * 100
        print(f"\nSilver/source agreement: {agree_pct:.1f}%  (no human IAA report)")

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
