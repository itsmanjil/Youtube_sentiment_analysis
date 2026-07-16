#!/usr/bin/env python3
"""
Thesis-grade significance testing on a single held-out test set.

Outputs
-------
- McNemar's exact test (paired) for model-vs-model comparisons
- Paired bootstrap confidence intervals for accuracy and macro-F1
  using a multinomial resampling trick (equivalent to bootstrap resampling
  examples with replacement, but much faster for deterministic predictions).

This script is designed for the repo's 3-class sentiment setting:
  Negative / Neutral / Positive
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label


SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
LABEL_TO_INT = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].apply(normalize_label)
    return df


def _coerce_model_list(value: str) -> List[str]:
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def _load_ensemble_weights(value: str | None) -> Dict[str, float] | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / raw
        if not path.exists():
            alt = BASE_DIR / raw
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Ensemble weights file not found: {raw}")
        payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
        payload = payload["weights"]
    if not isinstance(payload, dict):
        raise ValueError("ensemble_weights must be a JSON dict or a JSON file path.")

    weights = {}
    for k, v in payload.items():
        try:
            weights[str(k).strip().lower()] = float(v)
        except Exception:
            continue
    return weights or None


def _predict_labels(
    *,
    model: str,
    texts: List[str],
    preprocess: bool,
    ensemble_models: List[str],
    ensemble_weights: Dict[str, float] | None,
) -> List[str]:
    model = str(model).strip().lower()
    kwargs: Dict[str, Any] = {}

    if preprocess and model in {"tfidf", "logreg", "svm", "ensemble", "meta_learner", "fuzzy_ensemble"}:
        kwargs["preprocess"] = True

    if model == "ensemble":
        kwargs["base_models"] = ensemble_models
        kwargs["weights"] = ensemble_weights

    engine = get_sentiment_engine(model, **kwargs)
    results = engine.batch_analyze(texts)
    preds = [coerce_sentiment_result(r, model).label for r in results]
    return [normalize_label(p) for p in preds]


def _to_int_labels(labels: List[str]) -> np.ndarray:
    out = np.empty(len(labels), dtype=np.int8)
    for i, lab in enumerate(labels):
        out[i] = LABEL_TO_INT.get(normalize_label(lab), 1)  # default Neutral
    return out


def _confusion_from_counts(counts: np.ndarray) -> np.ndarray:
    """counts is a 3x3 confusion matrix."""
    if counts.shape != (3, 3):
        raise ValueError(f"Expected (3,3), got {counts.shape}")
    return counts.astype(np.int64, copy=False)


def _macro_f1_from_confusion(conf: np.ndarray) -> float:
    conf = _confusion_from_counts(conf)
    f1s: List[float] = []
    for c in range(3):
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - conf[c, c])
        fn = float(conf[c, :].sum() - conf[c, c])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _accuracy_from_confusion(conf: np.ndarray) -> float:
    conf = _confusion_from_counts(conf)
    total = float(conf.sum())
    if total <= 0:
        return 0.0
    return float(np.trace(conf) / total)


def _holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjustment."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = float(p_values[idx])
        adj = (m - rank) * p
        if adj > 1.0:
            adj = 1.0
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [float(x) for x in adjusted]


def _mcnemar_exact(y_true: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    correct_a = a == y_true
    correct_b = b == y_true
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))

    n = n01 + n10
    if n == 0:
        p_value = 1.0
    else:
        k = min(n01, n10)
        # Exact two-sided binomial test
        p_value = float(stats.binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)

    odds_ratio = None
    if n01 > 0 and n10 > 0:
        odds_ratio = float(n01 / n10)
    elif n01 > 0 and n10 == 0:
        odds_ratio = float("inf")
    elif n01 == 0 and n10 > 0:
        odds_ratio = 0.0
    else:
        odds_ratio = 1.0

    return {
        "n11": n11,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "p_value": p_value,
        "odds_ratio_n01_over_n10": odds_ratio,
    }


def _multinomial_bootstrap_joint(
    *,
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Paired bootstrap for multiple models using a multinomial resampling trick.

    Each example is categorized by (y_true, pred_model_1, ..., pred_model_M),
    which yields K^(M+1) categories (K=3). Bootstrap resampling over examples
    corresponds to sampling category counts ~ Multinomial(N, p_hat).
    """
    model_names = list(preds.keys())
    m = len(model_names)
    k = 3
    n = int(len(y_true))

    # Encode each example into a base-K index representing (true, pred1..predM).
    idx = y_true.astype(np.int64, copy=False)
    for name in model_names:
        idx = idx * k + preds[name].astype(np.int64, copy=False)

    n_categories = k ** (m + 1)
    counts = np.bincount(idx, minlength=n_categories).astype(np.int64, copy=False)
    probs = counts / float(n)

    rng = np.random.default_rng(int(seed))

    acc_samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in model_names}
    f1_samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in model_names}

    # Reshape target for marginalization: axis0=true, axis i+1 = preds[model_i]
    shape = (k,) * (m + 1)
    keep_axes_cache: Dict[str, Tuple[int, int]] = {name: (0, i + 1) for i, name in enumerate(model_names)}

    for b in range(n_bootstrap):
        boot_counts = rng.multinomial(n, probs).reshape(shape)
        for name in model_names:
            keep = keep_axes_cache[name]
            sum_axes = tuple(ax for ax in range(m + 1) if ax not in keep)
            conf = boot_counts.sum(axis=sum_axes)  # (3,3)
            acc_samples[name][b] = _accuracy_from_confusion(conf)
            f1_samples[name][b] = _macro_f1_from_confusion(conf)

    def ci(values: np.ndarray) -> Dict[str, float]:
        lo = float(np.quantile(values, alpha / 2.0))
        hi = float(np.quantile(values, 1.0 - alpha / 2.0))
        return {"low": lo, "high": hi}

    out: Dict[str, Any] = {"per_model": {}}
    for name in model_names:
        out["per_model"][name] = {
            "accuracy": ci(acc_samples[name]),
            "macro_f1": ci(f1_samples[name]),
        }

    # Pairwise deltas (B - A)
    out["pairwise_deltas"] = {}
    for a, b in combinations(model_names, 2):
        out["pairwise_deltas"][f"{b}_minus_{a}"] = {
            "accuracy": ci(acc_samples[b] - acc_samples[a]),
            "macro_f1": ci(f1_samples[b] - f1_samples[a]),
        }

    return out


def _write_table(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance tests on a held-out test set.")
    parser.add_argument("--data", required=True, help="Path to test CSV (expects text,label).")
    parser.add_argument(
        "--models",
        default="tfidf,logreg,svm,ensemble,meta_learner",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--ensemble-models",
        default="logreg,svm,tfidf",
        help="Comma-separated base models for the ensemble.",
    )
    parser.add_argument("--ensemble-weights", default=None, help="JSON dict or path to JSON weights.")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable shared classical preprocessing for classical models and derived ensembles.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples (default: 1000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_adjust", choices=["holm", "bonferroni", "none"], default="holm")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: backend/results/testset_significance.json).",
    )
    parser.add_argument(
        "--write_tables",
        action="store_true",
        help="Also write thesis-ready markdown/LaTeX tables into backend/results/.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    df = _load_dataset(data_path)

    texts = df["text"].astype(str).tolist()
    y_true_str = df["label"].astype(str).tolist()
    y_true = _to_int_labels(y_true_str)

    model_list = _coerce_model_list(args.models)
    ensemble_models = _coerce_model_list(args.ensemble_models)
    ensemble_weights = _load_ensemble_weights(args.ensemble_weights)

    preds_int: Dict[str, np.ndarray] = {}
    preds_str: Dict[str, List[str]] = {}

    for model in model_list:
        pred = _predict_labels(
            model=model,
            texts=texts,
            preprocess=bool(args.preprocess),
            ensemble_models=ensemble_models,
            ensemble_weights=ensemble_weights,
        )
        preds_str[model] = pred
        preds_int[model] = _to_int_labels(pred)

    # Point metrics (confusion + accuracy + macro-F1)
    point_metrics: Dict[str, Any] = {}
    for model in model_list:
        y_pred = preds_int[model]
        conf = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(y_true, y_pred, strict=False):
            conf[int(t), int(p)] += 1
        point_metrics[model] = {
            "accuracy": _accuracy_from_confusion(conf),
            "macro_f1": _macro_f1_from_confusion(conf),
            "confusion_matrix": conf.tolist(),
        }

    # McNemar tests (all pairs)
    pairs = list(combinations(model_list, 2))
    mcnemar_rows: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    for a, b in pairs:
        res = _mcnemar_exact(y_true, preds_int[a], preds_int[b])
        row = {"model_a": a, "model_b": b, **res}
        raw_p.append(float(res["p_value"]))
        mcnemar_rows.append(row)

    if args.p_adjust == "none":
        p_adj = raw_p[:]
    elif args.p_adjust == "bonferroni":
        p_adj = [min(1.0, float(p) * len(raw_p)) for p in raw_p]
    else:
        p_adj = _holm_adjust(raw_p)

    for row, padj in zip(mcnemar_rows, p_adj, strict=False):
        row["p_value_adjusted"] = float(padj)
        row["significant"] = bool(padj < float(args.alpha))

    # Bootstrap CIs (paired, across all models)
    bootstrap = _multinomial_bootstrap_joint(
        y_true=y_true,
        preds=preds_int,
        n_bootstrap=int(args.bootstrap),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )

    output = {
        "data": str(data_path),
        "n_samples": int(len(df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "models": model_list,
        "ensemble_models": ensemble_models,
        "ensemble_weights": ensemble_weights,
        "preprocess": bool(args.preprocess),
        "alpha": float(args.alpha),
        "bootstrap_samples": int(args.bootstrap),
        "seed": int(args.seed),
        "p_adjust": str(args.p_adjust),
        "point_metrics": point_metrics,
        "mcnemar": mcnemar_rows,
        "bootstrap_ci": bootstrap,
    }

    out_path = Path(args.output) if args.output else (BASE_DIR / "results" / "testset_significance.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.write_tables:
        results_dir = out_path.parent

        # Bootstrap CI table
        ci_lines = []
        ci_lines.append("| Model | Accuracy | Acc CI (low, high) | Macro-F1 | F1 CI (low, high) |")
        ci_lines.append("| --- | --- | --- | --- | --- |")
        for model in model_list:
            pm = point_metrics[model]
            ci_acc = bootstrap["per_model"][model]["accuracy"]
            ci_f1 = bootstrap["per_model"][model]["macro_f1"]
            ci_lines.append(
                "| {m} | {acc:.4f} | ({alo:.4f}, {ahi:.4f}) | {f1:.4f} | ({flo:.4f}, {fhi:.4f}) |".format(
                    m=model,
                    acc=float(pm["accuracy"]),
                    alo=float(ci_acc["low"]),
                    ahi=float(ci_acc["high"]),
                    f1=float(pm["macro_f1"]),
                    flo=float(ci_f1["low"]),
                    fhi=float(ci_f1["high"]),
                )
            )
        _write_table(results_dir / "thesis_bootstrap_ci.md", "\n".join(ci_lines) + "\n")

        # McNemar table
        m_lines = []
        m_lines.append("| Model A | Model B | n01 (A wrong, B correct) | n10 (A correct, B wrong) | p | p_adj | sig |")
        m_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in mcnemar_rows:
            m_lines.append(
                "| {a} | {b} | {n01} | {n10} | {p:.3g} | {pa:.3g} | {sig} |".format(
                    a=row["model_a"],
                    b=row["model_b"],
                    n01=int(row["n01"]),
                    n10=int(row["n10"]),
                    p=float(row["p_value"]),
                    pa=float(row["p_value_adjusted"]),
                    sig="yes" if row["significant"] else "no",
                )
            )
        _write_table(results_dir / "thesis_mcnemar.md", "\n".join(m_lines) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

