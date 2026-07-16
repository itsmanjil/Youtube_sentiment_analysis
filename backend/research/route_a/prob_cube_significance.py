#!/usr/bin/env python3
"""
Significance testing for Route A probability-cube experiments.

This compares base models and adaptive CI outputs on the same held-out test
cube, which keeps mixed-profile transformer/classical experiments consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from research.ci.multi_objective_ensemble import ensemble_probs
from research.ci.neuro_fuzzy_gate import NeuroFuzzyGate, model_confidence
from research.testset_significance import (
    _holm_adjust,
    _mcnemar_exact,
    _multinomial_bootstrap_joint,
)
from research.transformers.prob_cube_io import load_probability_cube


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _preds_from_probs(prob_matrix: np.ndarray, labels: List[str]) -> List[str]:
    return [labels[index] for index in np.asarray(prob_matrix).argmax(axis=1)]


def _metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _confidence_matrix(prob_cube: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [model_confidence(prob_cube[model_index]) for model_index in range(prob_cube.shape[0])]
    )


def build_prediction_sets(
    *,
    val_cube_path: Path,
    test_cube_path: Path,
    ci_results_path: Path | None,
) -> Dict[str, object]:
    val_bundle = load_probability_cube(val_cube_path)
    test_bundle = load_probability_cube(test_cube_path)
    labels = list(test_bundle.labels)

    predictions: Dict[str, List[str]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    compare_models: List[str] = []

    for model_index, model_name in enumerate(test_bundle.model_names):
        model_preds = _preds_from_probs(test_bundle.prob_cube[model_index], labels)
        predictions[model_name] = model_preds
        metrics[model_name] = _metrics(test_bundle.y_true, model_preds)
        compare_models.append(model_name)

    static_probs = test_bundle.prob_cube.mean(axis=0)
    static_name = "static_uniform"
    static_preds = _preds_from_probs(static_probs, labels)
    predictions[static_name] = static_preds
    metrics[static_name] = _metrics(test_bundle.y_true, static_preds)
    compare_models.append(static_name)

    ci_payload = None
    if ci_results_path is not None:
        ci_payload = json.loads(ci_results_path.read_text(encoding="utf-8"))
        weights = np.array(
            [ci_payload["knee_point"]["weights"][name] for name in test_bundle.model_names],
            dtype=float,
        )
        nsga_probs = ensemble_probs(weights, test_bundle.prob_cube)
        nsga_name = "nsga_knee"
        nsga_preds = _preds_from_probs(nsga_probs, labels)
        predictions[nsga_name] = nsga_preds
        metrics[nsga_name] = _metrics(test_bundle.y_true, nsga_preds)
        compare_models.append(nsga_name)

    gate = NeuroFuzzyGate(n_models=len(test_bundle.model_names))
    fit_info = gate.fit(
        _confidence_matrix(val_bundle.prob_cube),
        val_bundle.prob_cube,
        val_bundle.y_true,
        verbose=False,
    )
    fuzzy_probs = gate.predict_probs(
        _confidence_matrix(test_bundle.prob_cube),
        test_bundle.prob_cube,
    )
    fuzzy_name = "neuro_fuzzy"
    fuzzy_preds = _preds_from_probs(fuzzy_probs, labels)
    predictions[fuzzy_name] = fuzzy_preds
    metrics[fuzzy_name] = _metrics(test_bundle.y_true, fuzzy_preds)
    compare_models.append(fuzzy_name)

    return {
        "labels": labels,
        "compare_models": compare_models,
        "metrics": metrics,
        "predictions": predictions,
        "fit_info": fit_info,
        "y_true": list(test_bundle.y_true),
        "val_cube_path": str(val_cube_path),
        "test_cube_path": str(test_cube_path),
        "ci_payload": ci_payload,
    }


def _write_markdown(
    *,
    output_path: Path,
    compare_models: Iterable[str],
    metrics: Dict[str, Dict[str, float]],
    bootstrap: Dict[str, object],
    mcnemar_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "| Model | Accuracy | Macro-F1 | Acc CI | F1 CI |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for model_name in compare_models:
        acc_ci = bootstrap["per_model"][model_name]["accuracy"]
        f1_ci = bootstrap["per_model"][model_name]["macro_f1"]
        lines.append(
            f"| {model_name} | {metrics[model_name]['accuracy']:.4f} | "
            f"{metrics[model_name]['macro_f1']:.4f} | "
            f"({acc_ci['low']:.4f}, {acc_ci['high']:.4f}) | "
            f"({f1_ci['low']:.4f}, {f1_ci['high']:.4f}) |"
        )

    lines.extend(
        [
            "",
            "| Model A | Model B | n01 | n10 | p | p_adj | sig |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in mcnemar_rows:
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['n01']} | {row['n10']} | "
            f"{row['p_value']:.4g} | {row['p_value_adjusted']:.4g} | "
            f"{'yes' if row['significant'] else 'no'} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance testing for Route A probability cubes.")
    parser.add_argument("--val_cube", required=True, help="Validation probability cube (.npz).")
    parser.add_argument("--test_cube", required=True, help="Test probability cube (.npz).")
    parser.add_argument(
        "--ci_results",
        default=None,
        help="Optional NSGA-II result JSON. When provided, includes the knee-point ensemble.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults next to the test cube with a _significance suffix.",
    )
    args = parser.parse_args()

    val_cube_path = Path(args.val_cube)
    test_cube_path = Path(args.test_cube)
    ci_results_path = Path(args.ci_results) if args.ci_results else None
    output_path = (
        Path(args.output)
        if args.output
        else test_cube_path.with_name(f"{test_cube_path.stem}_significance.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = output_path.with_suffix(".md")

    bundle = build_prediction_sets(
        val_cube_path=val_cube_path,
        test_cube_path=test_cube_path,
        ci_results_path=ci_results_path,
    )

    labels = bundle["labels"]
    label_to_int = {label: index for index, label in enumerate(labels)}
    y_true = bundle["y_true"]
    y_true_int = np.array([label_to_int[label] for label in y_true], dtype=np.int8)
    preds_int = {
        name: np.array([label_to_int[label] for label in preds], dtype=np.int8)
        for name, preds in bundle["predictions"].items()
    }

    mcnemar_rows: List[Dict[str, object]] = []
    raw_p_values: List[float] = []
    for model_a, model_b in combinations(bundle["compare_models"], 2):
        row = {
            "model_a": model_a,
            "model_b": model_b,
            **_mcnemar_exact(y_true_int, preds_int[model_a], preds_int[model_b]),
        }
        mcnemar_rows.append(row)
        raw_p_values.append(float(row["p_value"]))

    adjusted = _holm_adjust(raw_p_values)
    for row, p_adjusted in zip(mcnemar_rows, adjusted, strict=False):
        row["p_value_adjusted"] = float(p_adjusted)
        row["significant"] = bool(p_adjusted < float(args.alpha))

    bootstrap = _multinomial_bootstrap_joint(
        y_true=y_true_int,
        preds={name: preds_int[name] for name in bundle["compare_models"]},
        n_bootstrap=int(args.bootstrap),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )

    output_payload = {
        "created_at": _utcnow(),
        "alpha": float(args.alpha),
        "bootstrap_samples": int(args.bootstrap),
        "seed": int(args.seed),
        "val_cube": bundle["val_cube_path"],
        "test_cube": bundle["test_cube_path"],
        "ci_results": str(ci_results_path) if ci_results_path else None,
        "n_samples": len(y_true),
        "labels": labels,
        "models": bundle["compare_models"],
        "metrics": bundle["metrics"],
        "fit_info": bundle["fit_info"],
        "mcnemar": mcnemar_rows,
        "bootstrap_ci": bootstrap,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    _write_markdown(
        output_path=markdown_path,
        compare_models=bundle["compare_models"],
        metrics=bundle["metrics"],
        bootstrap=bootstrap,
        mcnemar_rows=mcnemar_rows,
    )

    print(f"Saved JSON → {output_path}")
    print(f"Saved Markdown → {markdown_path}")


if __name__ == "__main__":
    main()
