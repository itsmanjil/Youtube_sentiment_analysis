"""
Reliability diagrams for the pinned live-runtime models (RQ3 / calibration).

Reuses the deployed application engines (src.sentiment.get_sentiment_engine) for
base/meta probabilities and reconstructs the NSGA-II ensemble from the pinned
knee-point weights + "ensemble" temperature, so the plotted curves correspond to
the pinned benchmark. Produces a 15-bin reliability diagram comparing logistic
regression, the NSGA-II ensemble, and the meta-learner.

Usage:
    cd backend
    python research/evaluation/reliability_diagrams.py \
        --data data/test.csv --out figures/reliability_diagrams.png
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression  # noqa: E402
if "multi_class" not in vars(LogisticRegression):
    LogisticRegression.multi_class = "multinomial"
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from research.evaluation.calibration import compute_ece, probs_to_matrix  # noqa: E402
from src.sentiment import get_sentiment_engine  # noqa: E402

LABELS = ("Positive", "Neutral", "Negative"); LIDX = {l: i for i, l in enumerate(LABELS)}
NORM = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}; NB = 15


def predict(model, texts, chunk=20000):
    e = get_sentiment_engine(model); probs = []
    for i in range(0, len(texts), chunk):
        for r in e.batch_analyze(texts[i:i + chunk]):
            probs.append(r.probs or {})
    return probs_to_matrix(probs, labels=LABELS)


def tscale(P, t):
    if t == 1.0:
        return P
    s = np.clip(P, 1e-10, None) ** (1.0 / t); return s / s.sum(1, keepdims=True)


def bin_curve(P, y_i):
    conf = P.max(1); corr = (P.argmax(1) == y_i).astype(float)
    edges = np.linspace(0, 1, NB + 1); b = np.clip(np.digitize(conf, edges[1:-1]), 0, NB - 1)
    xs, ys = [], []
    for k in range(NB):
        m = b == k
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(corr[m].mean())
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.csv")
    ap.add_argument("--out", default="figures/reliability_diagrams.png")
    a = ap.parse_args()
    ART = ROOT / "results" / "runtime" / "route_a_live_v1"
    df = pd.read_csv(a.data); y = df["label"].astype(str).values
    y_i = np.array([LIDX[v] for v in y]); texts = df["text"].astype(str).tolist()

    base = {m: predict(m, texts) for m in ("logreg", "svm", "tfidf")}
    meta = predict("meta_learner", texts)
    ts = {d["model"]: d["temperature"] for d in json.load(open(ART / "temperature_scaling.json"))["models"]}
    T = float(ts.get("ensemble", 1.0))
    nw = json.load(open(ART / "multi_objective_ensemble.json"))["knee_point"]["weights"]
    Pn = sum(nw.get(m, 0.0) * base[m] for m in ("logreg", "svm", "tfidf"))
    Pn = tscale(Pn / Pn.sum(1, keepdims=True), T)

    series = [("Logistic Regression", base["logreg"], "#1f77b4"),
              ("NSGA-II ensemble", Pn, "#2ca02c"),
              ("Meta-learner", meta, "#d62728")]
    plt.figure(figsize=(6.2, 6.0)); plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, P, c in series:
        ece = compute_ece(y, P, labels=LABELS)
        xs, ys = bin_curve(P, y_i)
        plt.plot(xs, ys, marker="o", ms=4, lw=1.6, color=c, label=f"{name} (ECE {ece:.4f})")
    plt.xlabel("Mean predicted confidence (bin)"); plt.ylabel("Empirical accuracy (bin)")
    plt.title(f"Reliability diagrams ({NB} bins, n={len(y):,} test)")
    plt.legend(loc="upper left", fontsize=8.5); plt.grid(alpha=0.3)
    plt.xlim(0.3, 1.0); plt.ylim(0.3, 1.0); plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); plt.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
