"""
Ablation Study Framework.

This module provides a systematic framework for conducting ablation studies,
which are essential for thesis work to demonstrate the contribution of
individual components.

Ablation Study Purpose
----------------------
An ablation study removes or modifies components of a model one at a time
to measure their individual contributions. This helps:

1. Validate that each component adds value
2. Quantify the contribution of each component
3. Understand component interactions
4. Justify architectural decisions in the thesis

Types of Ablation
-----------------
1. Component Ablation: Remove entire model components
   - Example: Remove attention mechanism from CNN-BiLSTM-Attention

2. Hyperparameter Ablation: Vary hyperparameters
   - Example: Test different embedding dimensions

3. Feature Ablation: Remove preprocessing steps
   - Example: Skip stopword removal

References
----------
Meyes, R., Lu, M., de Puiseau, C. W., & Meisen, T. (2019).
Ablation Studies in Artificial Neural Networks. arXiv:1901.08644.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class AblationStudyFramework:
    """
    Framework for systematic ablation studies.

    This class helps organize, run, and report ablation experiments
    in a reproducible manner.

    Parameters
    ----------
    base_model_fn : Callable
        Function that creates the base (full) model.
        Signature: () -> model
    evaluation_fn : Callable
        Function that evaluates a model and returns metrics.
        Signature: (model) -> Dict[str, float]
    metric_name : str, optional
        Primary metric to track (e.g., 'f1_macro').
        Default: 'f1_macro'
    n_runs : int, optional
        Number of runs per configuration for statistical reliability.
        Default: 3
    random_seed : int, optional
        Base random seed for reproducibility.
        Default: 42

    Attributes
    ----------
    ablations : List[Dict]
        Registered ablation configurations.
    results : Dict
        Results from completed ablation runs.

    Examples
    --------
    >>> # Define model creation and evaluation functions
    >>> def create_model():
    ...     return HybridCNNBiLSTM()
    >>>
    >>> def evaluate_model(model):
    ...     return {'accuracy': 0.85, 'f1_macro': 0.84}
    >>>
    >>> # Create ablation framework
    >>> ablation = AblationStudyFramework(
    ...     base_model_fn=create_model,
    ...     evaluation_fn=evaluate_model
    ... )
    >>>
    >>> # Register ablations
    >>> ablation.add_component_ablation(
    ...     'attention',
    ...     lambda: create_model(use_attention=False),
    ...     description="Remove multi-head attention mechanism"
    ... )
    >>>
    >>> # Run all ablations
    >>> results = ablation.run()
    >>>
    >>> # Generate thesis-ready report
    >>> ablation.generate_report('ablation_results/')
    """

    def __init__(
        self,
        base_model_fn: Callable,
        evaluation_fn: Callable,
        metric_name: str = "f1_macro",
        n_runs: int = 3,
        random_seed: int = 42,
    ):
        self.base_model_fn = base_model_fn
        self.evaluation_fn = evaluation_fn
        self.metric_name = metric_name
        self.n_runs = n_runs
        self.random_seed = random_seed

        self.ablations: List[Dict] = []
        self.results: Dict[str, Any] = {}
        self.baseline_results: Optional[Dict] = None

    def add_component_ablation(
        self,
        name: str,
        modified_model_fn: Callable,
        description: str = "",
    ) -> None:
        """
        Register a component ablation experiment.

        Parameters
        ----------
        name : str
            Name of the ablation (e.g., 'remove_attention').
        modified_model_fn : Callable
            Function that creates the modified model.
        description : str, optional
            Description of what is being ablated.

        Examples
        --------
        >>> ablation.add_component_ablation(
        ...     'no_cnn',
        ...     lambda: create_model(use_cnn=False),
        ...     description="Remove CNN branch, keep BiLSTM only"
        ... )
        """
        self.ablations.append({
            "name": name,
            "type": "component",
            "model_fn": modified_model_fn,
            "description": description,
        })

    def add_hyperparameter_ablation(
        self,
        param_name: str,
        param_values: List[Any],
        model_fn_factory: Callable[[Any], Callable],
        description: str = "",
    ) -> None:
        """
        Register a hyperparameter ablation experiment.

        Parameters
        ----------
        param_name : str
            Name of the hyperparameter (e.g., 'embed_dim').
        param_values : List[Any]
            Values to test for this hyperparameter.
        model_fn_factory : Callable
            Function that takes a param value and returns a model_fn.
            Signature: (param_value) -> Callable[[], model]
        description : str, optional
            Description of the hyperparameter.

        Examples
        --------
        >>> ablation.add_hyperparameter_ablation(
        ...     'embed_dim',
        ...     [100, 200, 300, 400],
        ...     lambda dim: lambda: create_model(embed_dim=dim),
        ...     description="Test different embedding dimensions"
        ... )
        """
        for value in param_values:
            self.ablations.append({
                "name": f"{param_name}_{value}",
                "type": "hyperparameter",
                "param_name": param_name,
                "param_value": value,
                "model_fn": model_fn_factory(value),
                "description": f"{description}: {param_name}={value}",
            })

    def add_feature_ablation(
        self,
        name: str,
        preprocessing_fn: Callable,
        description: str = "",
    ) -> None:
        """
        Register a feature/preprocessing ablation.

        This modifies the data preprocessing rather than the model.

        Parameters
        ----------
        name : str
            Name of the ablation (e.g., 'no_stopwords').
        preprocessing_fn : Callable
            Modified preprocessing function.
        description : str, optional
            Description of the feature being ablated.
        """
        self.ablations.append({
            "name": name,
            "type": "feature",
            "preprocessing_fn": preprocessing_fn,
            "model_fn": self.base_model_fn,  # Use base model
            "description": description,
        })

    def run(
        self,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all registered ablation experiments.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress updates.
            Default: True

        Returns
        -------
        Dict[str, Any]
            Complete results including:
            - baseline: Baseline (full model) results
            - ablations: Results for each ablation
            - summary: Summary statistics and comparisons
        """
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "metric_name": self.metric_name,
                "n_runs": self.n_runs,
                "random_seed": self.random_seed,
            },
            "baseline": None,
            "ablations": {},
            "summary": {},
        }

        # Run baseline first
        if verbose:
            print("Running baseline (full model)...")
        self.baseline_results = self._run_experiment(
            "baseline", self.base_model_fn, verbose
        )
        self.results["baseline"] = self.baseline_results

        # Run each ablation
        for ablation in self.ablations:
            name = ablation["name"]
            if verbose:
                print(f"\nRunning ablation: {name}")
                if ablation.get("description"):
                    print(f"  Description: {ablation['description']}")

            ablation_results = self._run_experiment(
                name, ablation["model_fn"], verbose
            )
            ablation_results["description"] = ablation.get("description", "")
            ablation_results["type"] = ablation["type"]

            self.results["ablations"][name] = ablation_results

        # Compute summary
        self._compute_summary()

        return self.results

    def _run_experiment(
        self,
        name: str,
        model_fn: Callable,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run a single experiment with multiple runs."""
        metrics_per_run = []
        times_per_run = []

        for run_idx in range(self.n_runs):
            # Set seed for reproducibility
            seed = self.random_seed + run_idx
            np.random.seed(seed)

            start_time = time.time()

            try:
                model = model_fn()
                metrics = self.evaluation_fn(model)
                elapsed = time.time() - start_time

                metrics_per_run.append(metrics)
                times_per_run.append(elapsed)

                if verbose:
                    primary_metric = metrics.get(self.metric_name, 0)
                    print(f"  Run {run_idx + 1}/{self.n_runs}: "
                          f"{self.metric_name}={primary_metric:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Run {run_idx + 1}/{self.n_runs}: FAILED - {str(e)}")
                metrics_per_run.append({"error": str(e)})
                times_per_run.append(0)

        # Aggregate results
        valid_metrics = [m for m in metrics_per_run if "error" not in m]

        if not valid_metrics:
            return {
                "status": "failed",
                "error": "All runs failed",
                "runs": metrics_per_run,
            }

        # Compute statistics for each metric
        aggregated = {}
        for metric_key in valid_metrics[0].keys():
            values = [m[metric_key] for m in valid_metrics]
            aggregated[metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

        return {
            "status": "success",
            "n_successful_runs": len(valid_metrics),
            "metrics": aggregated,
            "primary_metric": aggregated.get(self.metric_name, {}),
            "avg_time_seconds": float(np.mean(times_per_run)),
        }

    def _compute_summary(self) -> None:
        """Compute summary statistics comparing ablations to baseline."""
        if not self.baseline_results or self.baseline_results["status"] != "success":
            return

        baseline_metric = self.baseline_results["primary_metric"]["mean"]
        summary = {
            "baseline_metric": baseline_metric,
            "ablation_impacts": {},
            "ranking": [],
        }

        impacts = []
        for name, results in self.results["ablations"].items():
            if results["status"] != "success":
                continue

            ablation_metric = results["primary_metric"]["mean"]
            absolute_diff = ablation_metric - baseline_metric
            relative_diff = (absolute_diff / baseline_metric * 100) if baseline_metric != 0 else 0

            impact = {
                "ablation_metric": ablation_metric,
                "absolute_diff": absolute_diff,
                "relative_diff_percent": relative_diff,
                "description": results.get("description", ""),
            }

            summary["ablation_impacts"][name] = impact
            impacts.append((name, ablation_metric, absolute_diff))

        # Rank by performance (descending)
        impacts.sort(key=lambda x: x[1], reverse=True)
        summary["ranking"] = [
            {"name": name, "metric": metric, "diff": diff}
            for name, metric, diff in impacts
        ]

        self.results["summary"] = summary

    def generate_report(
        self,
        output_dir: Union[str, Path],
        format: str = "all",
    ) -> None:
        """
        Generate thesis-ready ablation study report.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save report files.
        format : str, optional
            Output format: 'json', 'markdown', 'latex', or 'all'.
            Default: 'all'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format in ("json", "all"):
            json_path = output_dir / "ablation_results.json"
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

        if format in ("markdown", "all"):
            md_path = output_dir / "ablation_report.md"
            with open(md_path, "w") as f:
                f.write(self._generate_markdown_report())

        if format in ("latex", "all"):
            latex_path = output_dir / "ablation_table.tex"
            with open(latex_path, "w") as f:
                f.write(self._generate_latex_table())

    def _generate_markdown_report(self) -> str:
        """Generate markdown-formatted report."""
        lines = [
            "# Ablation Study Results",
            "",
            f"**Timestamp:** {self.results['timestamp']}",
            f"**Primary Metric:** {self.metric_name}",
            f"**Number of Runs:** {self.n_runs}",
            "",
            "## Baseline (Full Model)",
            "",
        ]

        if self.results["baseline"]["status"] == "success":
            baseline = self.results["baseline"]["primary_metric"]
            lines.append(f"- **{self.metric_name}:** {baseline['mean']:.4f} "
                        f"(+/- {baseline['std']:.4f})")
        else:
            lines.append("- **Status:** FAILED")

        lines.extend([
            "",
            "## Ablation Results",
            "",
            "| Ablation | Description | Metric | Change | % Change |",
            "|----------|-------------|--------|--------|----------|",
        ])

        summary = self.results.get("summary", {})
        impacts = summary.get("ablation_impacts", {})

        for name, impact in impacts.items():
            desc = impact.get("description", "")[:30]
            metric = impact["ablation_metric"]
            diff = impact["absolute_diff"]
            rel_diff = impact["relative_diff_percent"]

            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            rel_str = f"+{rel_diff:.1f}%" if rel_diff > 0 else f"{rel_diff:.1f}%"

            lines.append(f"| {name} | {desc} | {metric:.4f} | {diff_str} | {rel_str} |")

        lines.extend([
            "",
            "## Key Findings",
            "",
        ])

        if impacts:
            # Find most impactful ablations
            sorted_impacts = sorted(
                impacts.items(),
                key=lambda x: abs(x[1]["absolute_diff"]),
                reverse=True
            )

            lines.append("**Most impactful components (by absolute change):**")
            for name, impact in sorted_impacts[:3]:
                lines.append(f"- {name}: {impact['relative_diff_percent']:.1f}% change")

        lines.extend([
            "",
            "## Conclusion",
            "",
            "The ablation study demonstrates the contribution of each component. ",
            "Components with larger negative changes when removed are more important.",
        ])

        return "\n".join(lines)

    def _generate_latex_table(self) -> str:
        """Generate LaTeX table for thesis."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & " + self.metric_name.replace("_", r"\_") +
            r" & $\Delta$ & \% Change \\",
            r"\midrule",
        ]

        # Baseline
        if self.results["baseline"]["status"] == "success":
            baseline = self.results["baseline"]["primary_metric"]
            lines.append(
                f"Full Model (Baseline) & {baseline['mean']:.4f} & -- & -- \\\\"
            )

        # Ablations
        summary = self.results.get("summary", {})
        impacts = summary.get("ablation_impacts", {})

        for name, impact in impacts.items():
            metric = impact["ablation_metric"]
            diff = impact["absolute_diff"]
            rel_diff = impact["relative_diff_percent"]

            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            rel_str = f"+{rel_diff:.1f}\\%" if rel_diff > 0 else f"{rel_diff:.1f}\\%"
            name_escaped = name.replace("_", r"\_")

            lines.append(f"{name_escaped} & {metric:.4f} & {diff_str} & {rel_str} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)
