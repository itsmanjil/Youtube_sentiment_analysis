"""
Unified Experiment Runner for Thesis

Integrates fuzzy logic, metaheuristics optimization, and benchmark
evaluation into a single pipeline for reproducible thesis experiments.

Features:
- Complete experiment orchestration
- Automatic hyperparameter optimization via PSO
- Cross-domain evaluation on standard benchmarks
- LaTeX report generation for thesis
- JSON export for reproducibility

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    EvaluationConfig,
)
from .runner import (
    ThesisExperiment,
    ExperimentResult,
)
from .results import (
    ResultsAggregator,
    ThesisReportGenerator,
)

__all__ = [
    # Configuration
    'ExperimentConfig',
    'ModelConfig',
    'OptimizationConfig',
    'EvaluationConfig',
    # Runner
    'ThesisExperiment',
    'ExperimentResult',
    # Results
    'ResultsAggregator',
    'ThesisReportGenerator',
]
