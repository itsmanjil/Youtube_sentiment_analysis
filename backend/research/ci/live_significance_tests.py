"""
Live-runtime paired significance tests and bootstrap confidence intervals.
=========================================================================

Addresses RQ1/RQ3 inferential gap: the pinned live benchmark (route_a_live_v1)
reports point estimates for macro-F1 and ECE but no evidence that model
differences are statistically reliable rather than sampling noise.

Method
------
Base/meta probabilities are produced by the SAME application engines used by
the deployed runtime (src.sentiment.get_sentiment_engine), so reproduced
metrics match the pinned benchmark by construction. The PSO and NSGA-II
ensembles are reconstructed from the *pinned* weight artifacts (knee point /
pso_ensemble_weights) and the pinned "ensemble" temperature -- identical to the
EnsembleEngine's weighted soft-voting + p^(1/T) calibration -- which avoids
re-running the (~7 min) weight optimisation while remaining faithful. Every
model is validated against the pinned benchmark before any test is reported.

It then computes:
  * paired McNemar tests (Holm-adjusted) on label correctness, and
  * paired bootstrap 95% CIs on macro-F1 and ECE differences.

Usage
-----
    cd backend
    python research/ci/live_significance_tests.py \
        --data data/test.csv --bootstrap 2000 --seed 42 \
        --output results/runtime/route_a_live_v1/
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression  # noqa: E402
# Restore attribute removed in newer scikit-learn so unpickled multinomial
# models can predict_proba; argmax labels (and pinned accuracy) are unchanged.
if "multi_class" not in vars(LogisticRegression):
    LogisticRegression.multi_class = "multinomial"
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.evaluation.calibration import compute_ece, probs_to_matrix  # noqa: E402
from src.sentiment import get_sentiment_engine  # noqa: E402

LABELS = ("Positive", "Neutral", "Negative")
LIDX = {l: i for i, l in enumerate(LABELS)}
NORM = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
NB = 15

PINNED = {
    "logreg": (0.6946, 0.6928, 0.003900), "svm": (0.6801, 0.6780, 0.016953),
    "tfidf": (0.6622, 0.6567, 0.017889), "meta_learner": (0.6953, 0.6945, 0.015711),
    "ensemble_pso": (0.6872, 0.6852, 0.011272), "ensemble_nsga2": (0.6959, 0.6940, 0.004601),
}
PAIRS = [("meta_learner", "logreg", "macro_f1"), ("meta_learner", "ensemble_nsga2", "macro_f1"),
         ("ensemble_nsga2", "meta_learner", "ece"), ("ensemble_nsga2", "logreg", "ece"),
         ("ensemble_nsga2", "ensemble_pso", "ece"), ("ensemble_nsga2", "ensemble_pso", "macro_f1")]


def engine_predict(model, texts, chunk=20000):
    eng = get_sentiment_engine(model)
    labs, probs = [], []
    for i in range(0, len(texts), chunk):
        for r in eng.batch_analyze(texts[i:i + chunk]):
            labs.append(NORM.get(str(r.label).lower(), r.label)); probs.append(r.probs or {})
    return np.asarray(labs), probs_to_matrix(probs, labels=LABELS)


def temp_scale(P, T):
    if T == 1.0:
        return P
    s = np.clip(P, 1e-10, None) ** (1.0 / T); return s / s.sum(1, keepdims=True)


def ensemble(weights, base, T):
    P = sum(weights.get(m, 0.0) * base[m][1] for m in ("logreg", "svm", "tfidf"))
    P = temp_scale(P / P.sum(1, keepdims=True), T)
    return np.array(LABELS)[P.argmax(1)], P


def fast_f1(yt, yp):
    cm = np.bincount(yt * 3 + yp, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(denom > 0, 2 * tp / denom, 0.0).mean()


def fast_ece_prep(prob, y_i):
    conf = prob.max(1); corr = (prob.argmax(1) == y_i).astype(float)
    b = np.clip(np.digitize(conf, np.linspace(0, 1, NB + 1)[1:-1]), 0, NB - 1)
    return conf, corr, b


def fast_ece(prep, ix):
    conf, corr, b = prep; bb = b[ix]
    cnt = np.bincount(bb, minlength=NB); sc = np.bincount(bb, weights=conf[ix], minlength=NB)
    sa = np.bincount(bb, weights=corr[ix], minlength=NB); nz = cnt > 0
    return float(np.sum(cnt[nz] / len(ix) * np.abs(sa[nz] / cnt[nz] - sc[nz] / cnt[nz])))


def mcnemar(ca, cb):
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    return b, c, (1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1.0) ** 2 / (b + c), 1)))


def holm(ps):
    m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/runtime/route_a_live_v1/")
    a = ap.parse_args()

    df = pd.read_csv(a.data); y = df["label"].astype(str).values
    y_i = np.array([LIDX[v] for v in y]); texts = df["text"].astype(str).tolist()
    ART = ROOT / "results" / "runtime" / "route_a_live_v1"

    base = {m: engine_predict(m, texts) for m in ("logreg", "svm", "tfidf")}
    meta = engine_predict("meta_learner", texts)
    ts = {d["model"]: d["temperature"] for d in json.load(open(ART / "temperature_scaling.json"))["models"]}
    T = float(ts.get("ensemble", 1.0))
    nw = json.load(open(ART / "multi_objective_ensemble.json"))["knee_point"]["weights"]
    pw = json.load(open(ART / "pso_ensemble_weights.json"))["weights"]
    npred, nprob = ensemble(nw, base, T); ppred, pprob = ensemble(pw, base, T)

    M = {"logreg": base["logreg"], "svm": base["svm"], "tfidf": base["tfidf"],
         "meta_learner": meta, "ensemble_pso": (ppred, pprob), "ensemble_nsga2": (npred, nprob)}
    D = {k: dict(pred=v[0], prob=v[1], pred_i=np.array([LIDX.get(x, -1) for x in v[0]]),
                 correct=(v[0] == y)) for k, v in M.items()}

    validation = []
    for k, d in D.items():
        acc = accuracy_score(y, d["pred"]); f1 = f1_score(y, d["pred"], average="macro", labels=list(LABELS))
        ece = compute_ece(y, d["prob"], labels=LABELS); pa, pf, pe = PINNED[k]
        validation.append(dict(model=k, accuracy=round(float(acc), 6), macro_f1=round(float(f1), 6),
                               ece=round(float(ece), 6), pinned_accuracy=pa, pinned_macro_f1=pf,
                               pinned_ece=pe, validated=bool(abs(acc - pa) <= 0.0015 and abs(f1 - pf) <= 0.0015)))

    mcp = list({(x, z) for x, z, _ in PAIRS}); raw = []; rows = []
    for x, z in mcp:
        bb, cc, p = mcnemar(D[x]["correct"], D[z]["correct"]); raw.append(p)
        rows.append(dict(model_a=x, model_b=z, b_a_correct_b_wrong=bb, c_a_wrong_b_correct=cc, p_raw=p))
    for r, pa in zip(rows, holm(raw)):
        r["p_holm"] = float(pa); r["significant_0.05"] = bool(pa < 0.05)

    eprep = {k: fast_ece_prep(D[k]["prob"], y_i) for k in D}
    rng = np.random.default_rng(a.seed); N = len(y_i); bt = []
    for x, z, metric in PAIRS:
        d = np.empty(a.bootstrap)
        for i in range(a.bootstrap):
            ix = rng.integers(0, N, N)
            if metric == "macro_f1":
                yb = y_i[ix]; d[i] = fast_f1(yb, D[x]["pred_i"][ix]) - fast_f1(yb, D[z]["pred_i"][ix])
            else:
                d[i] = fast_ece(eprep[x], ix) - fast_ece(eprep[z], ix)
        lo, hi = np.percentile(d, [2.5, 97.5])
        if metric == "macro_f1":
            pt = fast_f1(y_i, D[x]["pred_i"]) - fast_f1(y_i, D[z]["pred_i"])
        else:
            pt = fast_ece(eprep[x], np.arange(N)) - fast_ece(eprep[z], np.arange(N))
        bt.append(dict(model_a=x, model_b=z, metric=metric, point_diff=float(pt),
                       ci_low=float(lo), ci_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0)))

    payload = dict(dataset=a.data, n_samples=int(N), bootstrap_resamples=a.bootstrap, seed=a.seed,
                   runtime_artifact_version="route_a_live_v1", ensemble_temperature=T,
                   nsga2_weights=nw, pso_weights=pw, validation=validation, mcnemar=rows, bootstrap_ci=bt)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "live_significance_tests.json").write_text(json.dumps(payload, indent=2))
    (out / "live_significance_tests.md").write_text(build_md(payload))
    print("wrote", out / "live_significance_tests.md")


def build_md(p):
    out = []
    out.append("# Live-Runtime Significance Tests & Bootstrap Confidence Intervals")
    out.append("")
    out.append("- Runtime artifact: `route_a_live_v1`  |  Dataset: `%s` (n = %s)" % (p["dataset"], format(p["n_samples"], ",")))
    out.append("- Bootstrap: %d resamples (seed %d); ensemble temperature T = %s" % (p["bootstrap_resamples"], p["seed"], p["ensemble_temperature"]))
    out.append("- NSGA-II knee weights: %s" % json.dumps(p["nsga2_weights"]))
    out.append("- PSO weights: %s" % json.dumps(p["pso_weights"]))
    out.append("")
    out.append("## Reproduction validation (reconstructed vs pinned benchmark)")
    out.append("")
    out.append("| Model | Acc (repro/pinned) | Macro-F1 (repro/pinned) | ECE (repro/pinned) | Validated |")
    out.append("| --- | --- | --- | --- | --- |")
    for v in p["validation"]:
        out.append("| %s | %.4f / %.4f | %.4f / %.4f | %.6f / %.6f | %s |" % (
            v["model"], v["accuracy"], v["pinned_accuracy"], v["macro_f1"], v["pinned_macro_f1"],
            v["ece"], v["pinned_ece"], "yes" if v["validated"] else "NO"))
    out.append("")
    out.append("## Paired McNemar tests (label correctness, Holm-adjusted)")
    out.append("")
    out.append("| Model A | Model B | A correct / B wrong | A wrong / B correct | p (raw) | p (Holm) | Significant (a=0.05) |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in p["mcnemar"]:
        out.append("| %s | %s | %d | %d | %.3e | %.3e | %s |" % (
            r["model_a"], r["model_b"], r["b_a_correct_b_wrong"], r["c_a_wrong_b_correct"],
            r["p_raw"], r["p_holm"], "yes" if r["significant_0.05"] else "no"))
    out.append("")
    out.append("## Paired bootstrap 95% CIs on metric differences (A - B)")
    out.append("")
    out.append("| Model A | Model B | Metric | Point diff | 95% CI | Excludes 0 |")
    out.append("| --- | --- | --- | ---: | --- | --- |")
    for r in p["bootstrap_ci"]:
        out.append("| %s | %s | %s | %+.5f | [%+.5f, %+.5f] | %s |" % (
            r["model_a"], r["model_b"], r["metric"], r["point_diff"], r["ci_low"], r["ci_high"],
            "yes" if r["excludes_zero"] else "no"))
    out.append("")
    out.append("_For an ECE difference, a negative value favours Model A (lower calibration error). "
               "A CI excluding zero indicates a statistically reliable difference at the 5% level._")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
