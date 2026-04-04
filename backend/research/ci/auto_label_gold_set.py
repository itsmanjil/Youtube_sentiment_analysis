"""
Auto-label gold_set_template.csv using the live PSO ensemble.

Uses the project's own logreg + svm engines with PSO weights
(logreg:0.3087, svm:0.6913) to produce silver labels for the gold set.
Outputs gold_set_silver_labeled.csv with:
  - text
  - silver_label     : ensemble prediction
  - silver_confidence: max prob
  - silver_probs     : full prob dict (JSON string)
  - source_label     : original dataset label (from gold_set_labeled_from_dataset.csv)
  - agreement        : silver_label == source_label
"""

import csv
import json
import sys
from pathlib import Path

# Add backend/src to path
BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.sentiment.engines.logreg_engine import LogRegSentimentEngine
from src.sentiment.engines.svm_engine import SVMSentimentEngine

# PSO weights
PSO_WEIGHTS = {"logreg": 0.3087, "svm": 0.6913}

TEMPLATE = BACKEND / "data" / "gold_set_template.csv"
LABELED  = BACKEND / "data" / "gold_set_labeled_from_dataset.csv"
OUT      = BACKEND / "data" / "gold_set_silver_labeled.csv"


def normalize(probs: dict) -> dict:
    total = sum(probs.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def ensemble_predict(text: str, logreg, svm) -> dict:
    lr_res = logreg.analyze(text)
    sv_res = svm.analyze(text)

    labels = ["Negative", "Neutral", "Positive"]
    combined = {}
    for lbl in labels:
        combined[lbl] = (
            PSO_WEIGHTS["logreg"] * lr_res.probs.get(lbl, 0.0)
            + PSO_WEIGHTS["svm"] * sv_res.probs.get(lbl, 0.0)
        )
    return normalize(combined)


def main():
    print("Loading engines...")
    logreg = LogRegSentimentEngine()
    svm = SVMSentimentEngine()
    print("Engines loaded.")

    # Load source labels for agreement check
    source_labels = {}
    with open(LABELED) as f:
        for row in csv.DictReader(f):
            source_labels[row["text"].strip()] = row["label"].strip()

    # Load template
    with open(TEMPLATE) as f:
        rows = list(csv.DictReader(f))

    print(f"Labeling {len(rows)} rows...")
    results = []
    for i, row in enumerate(rows):
        text = row["text"].strip()
        probs = ensemble_predict(text, logreg, svm)
        silver_label = max(probs, key=probs.get)
        silver_conf  = probs[silver_label]
        src_label    = source_labels.get(text, "")
        agreement    = "yes" if silver_label == src_label else "no"

        results.append({
            "text":              text,
            "silver_label":      silver_label,
            "silver_confidence": f"{silver_conf:.4f}",
            "silver_probs":      json.dumps({k: round(v, 4) for k, v in probs.items()}),
            "source_label":      src_label,
            "agreement":         agreement,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    # Write output
    fieldnames = ["text", "silver_label", "silver_confidence", "silver_probs",
                  "source_label", "agreement"]
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Stats
    agree_count = sum(1 for r in results if r["agreement"] == "yes")
    from collections import Counter
    label_dist = Counter(r["silver_label"] for r in results)
    disagree_rows = [r for r in results if r["agreement"] == "no"]

    print(f"\nDone. Wrote {OUT}")
    print(f"Agreement with source labels: {agree_count}/{len(results)} "
          f"({agree_count/len(results)*100:.1f}%)")
    print(f"Silver label distribution: {dict(label_dist)}")
    print(f"\nDisagreements ({len(disagree_rows)}):")
    for r in disagree_rows[:10]:
        print(f"  source={r['source_label']:8s} silver={r['silver_label']:8s} "
              f"conf={r['silver_confidence']} | {r['text'][:70]}")
    if len(disagree_rows) > 10:
        print(f"  ... and {len(disagree_rows) - 10} more")


if __name__ == "__main__":
    main()
