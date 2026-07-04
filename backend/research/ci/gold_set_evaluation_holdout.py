"""
Gold Set Evaluation, Held-Out-Only Subset

Companion to gold_set_evaluation.py. That script evaluates all 300 gold-set
items, but ~95 of them (see gold_set_train_membership.py /
data/gold_set_split_membership.csv) are exact-text members of the training
split, because the gold set was originally sampled from train.csv rather
than from the held-out test split. Models may have memorised these items,
so this script re-runs the same evaluation restricted to the ~205 items
NOT found in train (val + test + no-match), producing a train-leakage-free
estimate of model-vs-human performance for comparison with
results/gold_set/gold_set_evaluation.md.
"""

import csv
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.sentiment.engines.logreg_engine import LogRegSentimentEngine
from src.sentiment.engines.svm_engine import SVMSentimentEngine
from src.sentiment.engines.tfidf_engine import TFIDFSentimentEngine
from src.sentiment.engines.ensemble_engine import EnsembleSentimentEngine
from src.sentiment.engines.meta_learner_engine import MetaLearnerSentimentEngine

from research.ci.gold_set_evaluation import (
    evaluate,
    _filter_by_reference,
    load_iaa_metrics,
)

SILVER_CSV = BACKEND / "data" / "gold_set_silver_labeled.csv"
RECONCILED_CSV = BACKEND / "data" / "gold_set_human_reconciled.csv"
MEMBERSHIP_CSV = BACKEND / "data" / "gold_set_split_membership.csv"
OUT_DIR = BACKEND / "results" / "gold_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]
MODEL_ORDER = ["logreg", "svm", "tfidf", "ensemble_pso", "ensemble_nsga2", "meta_learner"]


def main() -> None:
    with open(SILVER_CSV, encoding="utf-8-sig", newline="") as f:
        silver_rows = list(csv.DictReader(f))
    texts = [r["text"] for r in silver_rows]
    source_labels = [r["source_label"] for r in silver_rows]

    with open(MEMBERSHIP_CSV, encoding="utf-8-sig", newline="") as f:
        membership = {r["text"]: r["split_membership"] for r in csv.DictReader(f)}

    text_to_gold = {}
    with open(RECONCILED_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gl = row.get("gold_label", "").strip()
            if gl and row.get("is_disputed", "").strip().lower() != "yes":
                text_to_gold[row["text"].strip()] = gl
    human_labels = [text_to_gold.get(t) for t in texts]

    held_out_mask = [membership.get(t, "none") != "train" for t in texts]
    n_train = sum(1 for m in held_out_mask if not m)
    n_holdout = sum(held_out_mask)
    print(f"Gold set: {len(texts)} total, {n_train} in-train (excluded), {n_holdout} held-out.")

    print("Loading engines...")
    engines = {
        "logreg": LogRegSentimentEngine(),
        "svm": SVMSentimentEngine(),
        "tfidf": TFIDFSentimentEngine(),
        "ensemble_pso": EnsembleSentimentEngine(weights_optimization="pso"),
        "ensemble_nsga2": EnsembleSentimentEngine(weights_optimization="nsga2"),
        "meta_learner": MetaLearnerSentimentEngine(),
    }

    print("Running model inference (held-out subset only)...")
    holdout_texts = [t for t, keep in zip(texts, held_out_mask) if keep]
    holdout_human = [h for h, keep in zip(human_labels, held_out_mask) if keep]

    all_preds = {}
    for name, engine in engines.items():
        print(f"  {name}...")
        results = engine.batch_analyze(holdout_texts)
        all_preds[name] = [r.label for r in results]

    holdout_results = {}
    for name in MODEL_ORDER:
        _, filtered_preds, filtered_ref = _filter_by_reference(
            holdout_texts, all_preds[name], holdout_human
        )
        holdout_results[name] = evaluate(filtered_ref, filtered_preds, name)

    n_human_valid = sum(1 for h in holdout_human if h)

    output = {
        "description": (
            "Gold set evaluation restricted to items NOT present in the training "
            "split (excludes 95/300 items that are exact-text training-split "
            "members). human_ref = majority-vote reconciled human labels."
        ),
        "n_samples_total": len(texts),
        "n_train_excluded": n_train,
        "n_holdout": n_holdout,
        "n_human_valid": n_human_valid,
        "holdout_ref": holdout_results,
    }
    with open(OUT_DIR / "gold_set_evaluation_holdout.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    md = [
        "# Gold Set Evaluation, Held-Out-Only Subset\n",
        f"- Full gold set: {len(texts)} items ({n_train} are exact-text members of "
        f"the training split and excluded here; see `gold_set_train_membership.py`).",
        f"- Held-out subset (val + test + no-match): **{n_holdout} items** "
        f"({n_human_valid} with a non-disputed human label).\n",
        "## Model Performance vs Human-Reconciled Gold Labels (held-out subset)\n",
        "| Model | Accuracy | Macro F1 | Weighted F1 | N |",
        "|-------|----------|----------|-------------|---|",
    ]
    for name in MODEL_ORDER:
        r = holdout_results[name]
        md.append(
            f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['weighted_f1']:.4f} | {r['n_samples']} |"
        )
    md.append("")
    md.append(
        "> Compare against `results/gold_set/gold_set_evaluation.md` Table 1 "
        "(all 300 items, including 95 that overlap the training split). If "
        "accuracy/macro-F1 drop materially here, the full-gold-set figures were "
        "inflated by training-set memorisation.\n"
    )
    with open(OUT_DIR / "gold_set_evaluation_holdout.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nWrote {OUT_DIR / 'gold_set_evaluation_holdout.md'}")
    print(f"{'Model':<20} {'Acc':>6} {'MacF1':>7} {'WtF1':>7} {'N':>5}")
    for name in MODEL_ORDER:
        r = holdout_results[name]
        print(f"{name:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>7.4f} {r['weighted_f1']:>7.4f} {r['n_samples']:>5}")


if __name__ == "__main__":
    main()
