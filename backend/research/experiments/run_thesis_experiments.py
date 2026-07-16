#!/usr/bin/env python3
"""
Unified Thesis Experiment Runner

Run complete thesis experiments with a single command.
Integrates fuzzy logic, PSO optimization, and benchmark evaluation.

Usage:
    # Quick test run
    python -m research.experiments.run_thesis_experiments --mode quick

    # Full benchmark
    python -m research.experiments.run_thesis_experiments --mode full

    # Cross-domain study
    python -m research.experiments.run_thesis_experiments --mode cross_domain

    # Custom config
    python -m research.experiments.run_thesis_experiments --config my_config.yaml

Author: Thesis Project — Computational Intelligence & Sentiment Analysis
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from research.experiments.config import (
    ExperimentConfig,
    ExperimentType,
    get_quick_test_config,
    get_full_benchmark_config,
    get_cross_domain_config,
    get_ablation_config,
)
from research.experiments.runner import ThesisExperiment
from research.experiments.results import ThesisReportGenerator


def print_banner():
    """Print experiment banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   COMPUTATIONAL INTELLIGENCE THESIS EXPERIMENT RUNNER            ║
║   ─────────────────────────────────────────────────────          ║
║                                                                  ║
║   Integrates:                                                    ║
║     • Fuzzy Sentiment Classification                             ║
║     • PSO Ensemble Optimization                                  ║
║     • Benchmark Dataset Evaluation                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def print_smoke_test_warning():
    """Print warning for non-thesis smoke-test modes."""
    print(
        "\n[NOTICE] quick/demo modes may use synthetic data and mock models. "
        "Do NOT report these numbers as thesis results."
    )


def run_quick_test(output_dir: str = None):
    """Run quick test experiment."""
    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)
    print_smoke_test_warning()

    config = get_quick_test_config()
    config.name = f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir:
        config.output_dir = output_dir

    experiment = ThesisExperiment(config)
    result = experiment.run()

    print("\n" + experiment.get_summary())

    return result


def run_full_benchmark(output_dir: str = None):
    """Run full benchmark experiment."""
    print("\n" + "=" * 60)
    print("FULL BENCHMARK MODE")
    print("=" * 60)

    config = get_full_benchmark_config()
    config.name = f"full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir:
        config.output_dir = output_dir

    experiment = ThesisExperiment(config)
    result = experiment.run()

    print("\n" + experiment.get_summary())

    # Generate LaTeX tables
    generator = ThesisReportGenerator(result)
    generator.save_all(config.output_dir)
    print(f"\nLaTeX tables saved to: {config.output_dir}")

    return result


def run_cross_domain(output_dir: str = None):
    """Run cross-domain evaluation."""
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN EVALUATION MODE")
    print("=" * 60)

    config = get_cross_domain_config()
    config.name = f"cross_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir:
        config.output_dir = output_dir

    experiment = ThesisExperiment(config)
    result = experiment.run()

    print("\n" + experiment.get_summary())

    # Generate cross-domain table
    generator = ThesisReportGenerator(result)
    latex = generator.generate_cross_domain_table()
    if latex:
        print("\nCross-Domain LaTeX Table:")
        print("-" * 40)
        print(latex)

    return result


def run_ablation(output_dir: str = None):
    """Run ablation study."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY MODE")
    print("=" * 60)

    config = get_ablation_config()
    config.name = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir:
        config.output_dir = output_dir

    experiment = ThesisExperiment(config)
    result = experiment.run()

    print("\n" + experiment.get_summary())

    return result


def run_custom(config_path: str, output_dir: str = None):
    """Run experiment with custom config."""
    print("\n" + "=" * 60)
    print(f"CUSTOM CONFIG: {config_path}")
    print("=" * 60)

    config = ExperimentConfig.load(config_path)
    if output_dir:
        config.output_dir = output_dir

    experiment = ThesisExperiment(config)
    result = experiment.run()

    print("\n" + experiment.get_summary())

    # Generate reports
    generator = ThesisReportGenerator(result)
    generator.save_all(config.output_dir)
    print(f"\nLaTeX tables saved to: {config.output_dir}")

    return result


def demo_pipeline(output_dir: str = None):
    """Runs the pipeline on synthetic data only; not part of reported results."""
    print("\n" + "=" * 60)
    print("THESIS PIPELINE DEMONSTRATION (SYNTHETIC DATA -- NOT REPORTED RESULTS)")
    print("=" * 60)
    print_smoke_test_warning()

    # Step 1: Configuration
    print("\n[Step 1] Creating experiment configuration...")
    config = ExperimentConfig(
        name="thesis_demo",
        description="Complete thesis experiment demonstration",
        experiment_type=ExperimentType.FULL_BENCHMARK,
    )
    config.evaluation.datasets = ['synthetic']
    config.evaluation.allow_synthetic_data = True
    config.model.allow_mock_fallbacks = True
    if output_dir is None:
        output_dir = "./results/_demo_synthetic"
    if output_dir:
        config.output_dir = output_dir

    # Use smaller dataset for demo
    config.evaluation.max_samples_per_dataset = 200
    config.optimization.max_iterations = 20
    config.optimization.population_size = 15
    config.model.use_fuzzy = True
    config.model.optimize_weights = True

    print(f"  Experiment: {config.name}")
    print(f"  Type: {config.experiment_type.value}")
    print(f"  Models: {config.model.get_active_models()}")
    print(f"  Fuzzy: {config.model.use_fuzzy}")
    print(f"  PSO Optimization: {config.model.optimize_weights}")

    # Step 2: Run experiment
    print("\n[Step 2] Running thesis experiment...")
    experiment = ThesisExperiment(config)
    result = experiment.run()

    # Step 3: Generate summary
    print("\n[Step 3] Experiment Summary:")
    print(experiment.get_summary())

    # Step 4: Generate LaTeX
    print("\n[Step 4] Generating LaTeX tables for thesis...")
    generator = ThesisReportGenerator(result)
    tables = generator.generate_all_tables()

    for table_name in tables:
        if tables[table_name]:
            print(f"  ✓ Generated: {table_name}.tex")

    # Show main results table
    print("\n[Step 5] Main Results Table (LaTeX):")
    print("-" * 50)
    print(tables.get('main_results', 'N/A'))

    # Show ensemble weights if optimized
    if result.ensemble_weights is not None:
        print("\n[Step 6] PSO-Optimized Ensemble Weights:")
        print("-" * 50)
        print(tables.get('ensemble_weights', 'N/A'))

    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    result_path = output_dir / f"{config.name}_results.json"
    result.save(str(result_path))
    print(f"  ✓ Results: {result_path}")

    # Save LaTeX tables
    generator.save_all(str(output_dir))
    print(f"  ✓ LaTeX tables: {output_dir}/")

    # Save config
    config_path = output_dir / f"{config.name}_config.json"
    config.save(str(config_path))
    print(f"  ✓ Config: {config_path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"""
    Your smoke-test experiment has been completed.
    Do NOT include these metrics in your thesis results chapter.

    Generated files in {output_dir}/:
    ─────────────────────────────────────
    • {config.name}_results.json    - Full experiment results
    • {config.name}_config.json     - Experiment configuration
    • main_results.tex              - Main results table
    • model_comparison.tex          - Model comparison
    • cross_domain.tex              - Cross-domain matrix
    • ensemble_weights.tex          - PSO-optimized weights
    • per_class_f1.tex              - Per-class metrics
    • all_tables.tex                - All tables combined

    To include in your thesis:
    ─────────────────────────────────────
    \\input{{tables/main_results.tex}}
    \\input{{tables/ensemble_weights.tex}}

    Next steps:
    ─────────────────────────────────────
    1. Run thesis-grade datasets (--mode full)
    2. Add more optimization runs for statistical significance
    3. Include in your Results chapter
    """)

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Thesis Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m research.experiments.run_thesis_experiments --mode demo
  python -m research.experiments.run_thesis_experiments --mode quick
  python -m research.experiments.run_thesis_experiments --mode full
  python -m research.experiments.run_thesis_experiments --config my_config.yaml
        """
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'quick', 'full', 'cross_domain', 'ablation'],
        default='full',
        help='Experiment mode (default: full)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file (YAML or JSON)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print_banner()

    if args.config:
        result = run_custom(args.config, output_dir=args.output)
    elif args.mode == 'demo':
        result = demo_pipeline(output_dir=args.output)
    elif args.mode == 'quick':
        result = run_quick_test(output_dir=args.output)
    elif args.mode == 'full':
        result = run_full_benchmark(output_dir=args.output)
    elif args.mode == 'cross_domain':
        result = run_cross_domain(output_dir=args.output)
    elif args.mode == 'ablation':
        result = run_ablation(output_dir=args.output)

    return result


if __name__ == '__main__':
    main()
