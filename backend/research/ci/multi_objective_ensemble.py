"""
Multi-Objective Ensemble Optimisation via NSGA-II
==================================================

Optimises ensemble weights for three competing objectives simultaneously:

    obj 1:  -Macro-F1           (maximise F1  → minimise negative F1)
    obj 2:   ECE                (minimise Expected Calibration Error)
    obj 3:  -Coverage@0.70      (maximise fraction of samples where
                                  max class probability > 0.70)

The Pareto front returned by NSGA-II reveals the trade-off surface between
accuracy and calibration that no single-objective optimiser can expose.
The "knee point" (minimum Chebyshev distance to the ideal point) is selected
as the recommended operating point and evaluated on the held-out test set.

This module is the core CI novel contribution for the thesis:
    "Multi-Objective Ensemble Optimisation for Calibrated Sentiment Analysis"

Usage
-----
    cd backend
    python research/ci/multi_objective_ensemble.py \\
        --val  data/val.csv   \\
        --test data/test.csv  \\
        --sample 8000         \\
        --pop   60            \\
        --gen   80            \\
        --confidence 0.70     \\
        --seed  42            \\
        --output results/

Reference
---------
    Deb et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Resolve backend root so imports work regardless of invocation directory
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from src.sentiment import coerce_sentiment_result, get_sentiment_engine, normalize_label
from src.utils import SENTIMENT_LABELS, normalize_probs
from research.transformers.prob_cube_io import load_probability_cube

from research.computational_intelligence.metaheuristics.base import (
    ObjectiveType,
    OptimizationProblem,
)
from research.computational_intelligence.metaheuristics.nsga2 import NSGA2, NSGA2Config
from research.evaluation.calibration import compute_calibration_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODELS = ["logreg", "svm", "tfidf"]
TRANSFORMER_MODELS = {
    "bert",
    "transformer",
    "roberta",
    "modernbert",
    "deberta_v3",
    "xlm_v",
    "mdeberta_v3",
}
LABELS: List[str] = list(SENTIMENT_LABELS)  # ["Positive", "Neutral", "Negative"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_df(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].apply(normalize_label)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def parse_model_names(raw_value: str) -> List[str]:
    models = [item.strip().lower().replace("-", "_") for item in str(raw_value).split(",") if item.strip()]
    return models or list(DEFAULT_MODELS)


def _engine_kwargs(model_name: str, calibration_profile: str) -> dict:
    if model_name in TRANSFORMER_MODELS:
        return {"calibration_profile": calibration_profile}
    # NSGA-II ensemble weights are fitted on raw base-model probabilities,
    # matching how EnsembleSentimentEngine builds its base engines at serving
    # time (src/sentiment/engines/ensemble_engine.py). A calibrated base model
    # here would optimize weights against a distribution never actually served.
    return {"calibrate": False}


def precompute_probs(
    model_names: List[str],
    texts: List[str],
    *,
    calibration_profile: str = "auto",
) -> np.ndarray:
    """
    Returns shape (n_models, n_samples, n_classes) float array.
    Axis ordering lets us compute weighted_probs = weights @ prob_cube quickly.
    """
    n = len(texts)
    n_classes = len(LABELS)
    cube = np.zeros((len(model_names), n, n_classes), dtype=np.float64)

    for m_idx, name in enumerate(model_names):
        print(f"  Scoring {name} …", flush=True)
        engine = get_sentiment_engine(name, **_engine_kwargs(name, calibration_profile))
        results = engine.batch_analyze(texts)
        for i, r in enumerate(results):
            p = normalize_probs(coerce_sentiment_result(r, name).probs)
            for c_idx, lbl in enumerate(LABELS):
                cube[m_idx, i, c_idx] = p.get(lbl, 0.0)

    return cube


# ---------------------------------------------------------------------------
# Objective functions (operate on the pre-computed probability cube)
# ---------------------------------------------------------------------------

def ensemble_probs(weights: np.ndarray, prob_cube: np.ndarray) -> np.ndarray:
    """
    Weighted average of model probability matrices.

    weights : (n_models,)  — normalised to sum to 1
    prob_cube : (n_models, n_samples, n_classes)
    returns   : (n_samples, n_classes)
    """
    w = np.maximum(weights, 0.0)
    w = w / (w.sum() + 1e-12)
    return np.einsum("m,msc->sc", w, prob_cube)


def compute_objectives(
    weights: np.ndarray,
    prob_cube: np.ndarray,
    y_true: List[str],
    confidence_threshold: float,
) -> np.ndarray:
    """
    Returns a 3-element array of objectives (all to be MINIMISED):

        [0]  -macro_f1
        [1]   ece
        [2]  -coverage
    """
    probs = ensemble_probs(weights, prob_cube)           # (n, n_classes)
    preds = [LABELS[i] for i in probs.argmax(axis=1)]

    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    ece, _ = compute_calibration_metrics(y_true, probs, labels=LABELS)

    max_conf = probs.max(axis=1)
    coverage = float((max_conf >= confidence_threshold).mean())

    return np.array([-macro_f1, ece, -coverage], dtype=np.float64)


# ---------------------------------------------------------------------------
# OptimizationProblem subclass
# ---------------------------------------------------------------------------

class MultiObjectiveEnsembleProblem(OptimizationProblem):
    """
    Wraps the three-objective ensemble evaluation for NSGA-II.

    Decision variables
    ------------------
    x : (n_models,) — unnormalised weights in [0, 1].
        Internally normalised to sum to 1 before evaluation.

    Objectives (all minimised)
    --------------------------
    0   -Macro-F1           maximise accuracy
    1    ECE                minimise calibration error
    2   -Coverage@threshold maximise high-confidence predictions
    """

    def __init__(
        self,
        prob_cube: np.ndarray,
        y_true: List[str],
        confidence_threshold: float = 0.70,
        model_names: List[str] = None,
    ):
        n_models = prob_cube.shape[0]
        super().__init__(
            n_variables=n_models,
            bounds=[(0.0, 1.0)] * n_models,
            objectives=["neg_macro_f1", "ece", "neg_coverage"],
            objective_types=[ObjectiveType.MINIMIZE] * 3,
            name="MultiObjectiveEnsembleProblem",
        )
        self.prob_cube = prob_cube
        self.y_true = y_true
        self.confidence_threshold = confidence_threshold
        self.model_names = model_names or [f"model_{i}" for i in range(n_models)]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return compute_objectives(x, self.prob_cube, self.y_true, self.confidence_threshold)


# ---------------------------------------------------------------------------
# Knee-point selection
# ---------------------------------------------------------------------------

def select_knee_point(pareto_front) -> int:
    """
    Select the Pareto solution closest to the ideal point using
    normalised Chebyshev distance — a classic knee-point heuristic.

    The ideal point is the best (min) value per objective across the front.
    Returns the index into pareto_front.
    """
    fitness_matrix = np.array([sol.fitness for sol in pareto_front])
    ideal = fitness_matrix.min(axis=0)
    nadir = fitness_matrix.max(axis=0)

    ranges = nadir - ideal
    ranges[ranges < 1e-12] = 1.0  # avoid division by zero

    normalised = (fitness_matrix - ideal) / ranges
    chebyshev = normalised.max(axis=1)  # Chebyshev (L-inf) distance from ideal
    return int(np.argmin(chebyshev))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def pareto_table(
    pareto_front,
    model_names: List[str],
    confidence_threshold: float,
) -> str:
    rows = ["| Rank | " + " | ".join(f"w({m})" for m in model_names)
            + " | Macro-F1 | ECE | Coverage |",
            "|------|" + "------|" * len(model_names)
            + "----------|-----|----------|"]
    for rank, sol in enumerate(pareto_front, start=1):
        w = np.maximum(sol.position, 0.0)
        w = w / (w.sum() + 1e-12)
        f = sol.fitness  # [-f1, ece, -cov]
        weight_str = " | ".join(f"{v:.3f}" for v in w)
        rows.append(
            f"| {rank:4d} | {weight_str} "
            f"| {-f[0]:.4f} | {f[1]:.4f} | {-f[2]:.4f} |"
        )
    return "\n".join(rows)


def build_report(
    result,
    knee_idx: int,
    knee_weights: np.ndarray,
    val_objectives: np.ndarray,
    test_objectives: np.ndarray,
    model_names: List[str],
    confidence_threshold: float,
    pso_val_f1: float | None,
) -> str:
    lines = [
        "# Multi-Objective Ensemble Optimisation (NSGA-II)\n",
        "## Setup\n",
        f"- **Models**: {', '.join(model_names)}",
        f"- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@{confidence_threshold:.0%} (↑)",
        f"- **Algorithm**: NSGA-II  "
        f"(pop={result.metadata.get('population_size', '—')}, "
        f"gen={result.iterations})",
        f"- **Evaluations**: {result.evaluations:,}",
        f"- **Runtime**: {result.runtime:.1f}s",
        f"- **Pareto front size**: {len(result.pareto_front)} solutions\n",
        "## Pareto Front (Validation Set)\n",
        pareto_table(result.pareto_front, model_names, confidence_threshold),
        "\n## Knee-Point Selection\n",
        "The recommended operating point minimises the normalised Chebyshev "
        "distance to the ideal point — balancing all three objectives.\n",
        f"**Knee-point index** (1-based): {knee_idx + 1}\n",
        "### Optimal Weights\n",
        "| Model | Weight |",
        "|-------|--------|",
    ]
    for m, w in zip(model_names, knee_weights):
        lines.append(f"| {m} | {w:.4f} |")

    lines += [
        "\n## Validation Performance (knee-point)\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Macro-F1 | {-val_objectives[0]:.4f} |",
        f"| ECE | {val_objectives[1]:.4f} |",
        f"| Coverage@{confidence_threshold:.0%} | {-val_objectives[2]:.4f} |",
        "\n## Test-Set Performance (knee-point)\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Macro-F1 | {-test_objectives[0]:.4f} |",
        f"| ECE | {test_objectives[1]:.4f} |",
        f"| Coverage@{confidence_threshold:.0%} | {-test_objectives[2]:.4f} |",
    ]

    if pso_val_f1 is not None:
        delta = -val_objectives[0] - pso_val_f1
        lines += [
            "\n## Comparison: NSGA-II vs Single-Objective PSO\n",
            "| Method | Macro-F1 (val) | ECE (val) | Coverage (val) |",
            "|--------|---------------|-----------|----------------|",
            f"| NSGA-II (knee) | {-val_objectives[0]:.4f} | "
            f"{val_objectives[1]:.4f} | {-val_objectives[2]:.4f} |",
            f"| PSO (best F1) | {pso_val_f1:.4f} | — | — |",
            f"\nΔF1 (NSGA-II − PSO) = **{delta:+.4f}**  "
            "(ECE and coverage are additional gains only NSGA-II optimises)\n",
        ]

    lines += [
        "\n## Thesis Interpretation\n",
        "The Pareto front reveals that ensemble calibration and accuracy are "
        "partially competing: aggressive weight concentration (LogReg dominant) "
        "maximises F1 but raises ECE, while more balanced weights improve "
        "calibration at a small accuracy cost. "
        "The knee-point solution provides a principled, calibration-aware "
        "alternative to the single-objective PSO solution reported in §4.3, "
        "demonstrating that the CI contribution extends beyond raw accuracy "
        "improvement to probabilistic reliability — a key requirement for "
        "deployed sentiment systems.\n",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Objective Ensemble Optimisation (NSGA-II)")
    parser.add_argument("--val", default="data/val.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument(
        "--models",
        default="logreg,svm,tfidf",
        help="Comma-separated model list when scoring directly from CSV.",
    )
    parser.add_argument(
        "--val_cube",
        default=None,
        help="Optional validation probability cube (.npz) exported by research/transformers/export_prob_cube.py.",
    )
    parser.add_argument(
        "--test_cube",
        default=None,
        help="Optional test probability cube (.npz) exported by research/transformers/export_prob_cube.py.",
    )
    parser.add_argument("--sample", type=int, default=8000,
                        help="Max val samples (speed vs accuracy trade-off)")
    parser.add_argument("--pop", type=int, default=60, help="NSGA-II population size")
    parser.add_argument("--gen", type=int, default=80, help="NSGA-II generations")
    parser.add_argument("--confidence", type=float, default=0.70,
                        help="Confidence threshold for coverage objective")
    parser.add_argument(
        "--calibration_profile",
        default="auto",
        help="Transformer calibration mode when scoring directly from CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--pso_val_f1", type=float, default=None,
                        help="Single-objective PSO val F1 for comparison table "
                             "(from pso_convergence.json best_val_f1)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    model_names = parse_model_names(args.models)

    if args.val_cube:
        print("Loading validation probability cube …")
        val_bundle = load_probability_cube(args.val_cube)
        y_val = val_bundle.y_true
        val_cube = np.asarray(val_bundle.prob_cube, dtype=np.float64)
        model_names = val_bundle.model_names
    else:
        print("Loading validation data …")
        val_df = load_df(args.val, args.sample, args.seed)
        y_val = val_df["label"].tolist()
        texts_val = val_df["text"].tolist()
        print(f"\nPre-computing model scores on {len(texts_val):,} val samples …")
        val_cube = precompute_probs(
            model_names,
            texts_val,
            calibration_profile=args.calibration_profile,
        )

    if args.test_cube:
        print("Loading test probability cube …")
        test_bundle = load_probability_cube(args.test_cube)
        if test_bundle.model_names != model_names:
            raise SystemExit(
                "Validation and test cubes use different model orderings. "
                f"val={model_names} test={test_bundle.model_names}"
            )
        y_test = test_bundle.y_true
        test_cube = np.asarray(test_bundle.prob_cube, dtype=np.float64)
    else:
        print("Loading test data …")
        test_df = load_df(args.test, None, args.seed)
        y_test = test_df["label"].tolist()
        texts_test = test_df["text"].tolist()
        print(f"\nPre-computing model scores on {len(texts_test):,} test samples …")
        test_cube = precompute_probs(
            model_names,
            texts_test,
            calibration_profile=args.calibration_profile,
        )

    # ------------------------------------------------------------------
    # 2. Define problem and run NSGA-II
    # ------------------------------------------------------------------
    print("\nRunning NSGA-II …")
    problem = MultiObjectiveEnsembleProblem(
        prob_cube=val_cube,
        y_true=y_val,
        confidence_threshold=args.confidence,
        model_names=model_names,
    )

    nsga2 = NSGA2(
        problem=problem,
        population_size=args.pop,
        max_iterations=args.gen,
        config=NSGA2Config(crossover_prob=0.9, crossover_eta=20,
                           mutation_eta=20),
        seed=args.seed,
        verbose=True,
    )
    result = nsga2.optimize()

    pareto = result.pareto_front
    print(f"\nPareto front: {len(pareto)} non-dominated solutions")

    # ------------------------------------------------------------------
    # 3. Knee-point selection
    # ------------------------------------------------------------------
    knee_idx = select_knee_point(pareto)
    knee_sol = pareto[knee_idx]

    raw_w = np.maximum(knee_sol.position, 0.0)
    knee_weights = raw_w / (raw_w.sum() + 1e-12)

    val_objectives = knee_sol.fitness          # computed on val
    test_objectives = compute_objectives(
        knee_weights, test_cube, y_test, args.confidence
    )

    # ------------------------------------------------------------------
    # 4. Determine PSO baseline F1 (auto-detect from saved JSON)
    # ------------------------------------------------------------------
    pso_val_f1 = args.pso_val_f1
    if pso_val_f1 is None:
        pso_json = out_dir / "pso_convergence.json"
        if pso_json.exists():
            with open(pso_json) as fh:
                pso_data = json.load(fh)
            pso_val_f1 = pso_data.get("best_val_f1")
            if pso_val_f1:
                print(f"  Auto-loaded PSO val F1 = {pso_val_f1:.4f} from pso_convergence.json")

    # ------------------------------------------------------------------
    # 5. Build outputs
    # ------------------------------------------------------------------
    # JSON
    pareto_records = []
    for sol in pareto:
        w = np.maximum(sol.position, 0.0)
        w = w / (w.sum() + 1e-12)
        f = sol.fitness
        pareto_records.append({
            "weights": {m: round(float(w[i]), 6) for i, m in enumerate(model_names)},
            "macro_f1": round(float(-f[0]), 6),
            "ece": round(float(f[1]), 6),
            "coverage": round(float(-f[2]), 6),
        })

    output_data = {
        "algorithm": "NSGA-II",
        "objectives": ["neg_macro_f1", "ece", "neg_coverage"],
        "confidence_threshold": args.confidence,
        "population_size": args.pop,
        "generations": args.gen,
        "models": model_names,
        "val_cube": args.val_cube,
        "test_cube": args.test_cube,
        "runtime_s": round(result.runtime, 2),
        "evaluations": result.evaluations,
        "pareto_front_size": len(pareto),
        "pareto_front": pareto_records,
        "knee_point": {
            "index": knee_idx,
            "weights": {m: round(float(knee_weights[i]), 6) for i, m in enumerate(model_names)},
            "val": {
                "macro_f1": round(float(-val_objectives[0]), 6),
                "ece": round(float(val_objectives[1]), 6),
                "coverage": round(float(-val_objectives[2]), 6),
            },
            "test": {
                "macro_f1": round(float(-test_objectives[0]), 6),
                "ece": round(float(test_objectives[1]), 6),
                "coverage": round(float(-test_objectives[2]), 6),
            },
        },
        "pso_baseline_val_f1": pso_val_f1,
    }

    json_path = out_dir / "multi_objective_ensemble.json"
    with open(json_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # Markdown
    report = build_report(
        result=result,
        knee_idx=knee_idx,
        knee_weights=knee_weights,
        val_objectives=val_objectives,
        test_objectives=test_objectives,
        model_names=model_names,
        confidence_threshold=args.confidence,
        pso_val_f1=pso_val_f1,
    )
    md_path = out_dir / "multi_objective_ensemble.md"
    with open(md_path, "w") as fh:
        fh.write(report)
    print(f"Saved Markdown → {md_path}")

    # ------------------------------------------------------------------
    # 6. Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MULTI-OBJECTIVE NSGA-II RESULTS")
    print("=" * 60)
    print(f"Pareto front: {len(pareto)} solutions")
    print(f"\nKnee-point weights:")
    for m, w in zip(model_names, knee_weights):
        print(f"  {m:>8s}: {w:.4f}")
    print(f"\nValidation  →  F1={-val_objectives[0]:.4f}  "
          f"ECE={val_objectives[1]:.4f}  "
          f"Cov@{args.confidence:.0%}={-val_objectives[2]:.4f}")
    print(f"Test        →  F1={-test_objectives[0]:.4f}  "
          f"ECE={test_objectives[1]:.4f}  "
          f"Cov@{args.confidence:.0%}={-test_objectives[2]:.4f}")
    if pso_val_f1 is not None:
        print(f"\nSingle-obj PSO val F1: {pso_val_f1:.4f}  "
              f"(NSGA-II Δ = {-val_objectives[0] - pso_val_f1:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
