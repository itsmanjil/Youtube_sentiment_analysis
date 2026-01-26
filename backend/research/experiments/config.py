"""
Experiment Configuration Classes

Provides structured configuration for thesis experiments with
sensible defaults and validation.

Author: [Your Name]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
import json

# YAML is optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ExperimentType(Enum):
    """Type of experiment to run."""
    SINGLE_DATASET = "single_dataset"
    CROSS_DOMAIN = "cross_domain"
    ABLATION_STUDY = "ablation_study"
    HYPERPARAMETER_SEARCH = "hyperparameter_search"
    FULL_BENCHMARK = "full_benchmark"


class OptimizerType(Enum):
    """Optimization algorithm."""
    PSO = "pso"
    ADAPTIVE_PSO = "adaptive_pso"
    MOPSO = "mopso"
    NSGA2 = "nsga2"
    GRID_SEARCH = "grid_search"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for sentiment models."""

    # Base models to use
    use_logreg: bool = True
    use_svm: bool = True
    use_tfidf: bool = True
    use_bert: bool = False  # Requires GPU
    use_hybrid_nn: bool = False  # Requires training

    # Fuzzy configuration
    use_fuzzy: bool = True
    fuzzy_defuzz_method: str = "centroid"
    fuzzy_mf_type: str = "gaussian"

    # Ensemble configuration
    ensemble_method: str = "weighted_average"  # or "voting", "stacking"
    optimize_weights: bool = True

    def get_active_models(self) -> List[str]:
        """Get list of active model names."""
        models = []
        if self.use_logreg:
            models.append("logreg")
        if self.use_svm:
            models.append("svm")
        if self.use_tfidf:
            models.append("tfidf")
        if self.use_bert:
            models.append("bert")
        if self.use_hybrid_nn:
            models.append("hybrid_nn")
        return models


@dataclass
class OptimizationConfig:
    """Configuration for metaheuristic optimization."""

    # Optimizer selection
    optimizer_type: OptimizerType = OptimizerType.PSO

    # PSO parameters
    population_size: int = 30
    max_iterations: int = 50
    c1: float = 2.0  # Cognitive coefficient
    c2: float = 2.0  # Social coefficient
    w_start: float = 0.9  # Initial inertia
    w_end: float = 0.4  # Final inertia

    # Adaptive PSO
    use_adaptive: bool = True
    diversity_threshold: float = 0.1

    # Multi-objective
    n_objectives: int = 1  # 1 for accuracy, 2 for accuracy+speed
    archive_size: int = 50

    # General
    seed: int = 42
    n_runs: int = 5  # For statistical significance
    verbose: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""

    # Datasets to use
    datasets: List[str] = field(default_factory=lambda: [
        "synthetic"  # Start with synthetic for testing
    ])

    # Evaluation settings
    max_samples_per_dataset: int = 1000  # Limit for faster testing
    test_split_ratio: float = 0.2
    validation_split_ratio: float = 0.1

    # Cross-domain
    run_cross_domain: bool = True

    # Metrics
    primary_metric: str = "f1_macro"  # or "accuracy", "f1_weighted"
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95

    # Statistical tests
    run_statistical_tests: bool = True
    significance_level: float = 0.05

    def get_active_datasets(self) -> List[str]:
        """Get list of active dataset names."""
        return list(self.datasets)


@dataclass
class ExperimentConfig:
    """
    Master configuration for thesis experiments.

    Example
    -------
    >>> config = ExperimentConfig(
    ...     name="main_experiment",
    ...     experiment_type=ExperimentType.FULL_BENCHMARK
    ... )
    >>> config.model.use_fuzzy = True
    >>> config.optimization.optimizer_type = OptimizerType.PSO
    >>> config.save("experiment_config.yaml")
    """

    # Experiment identification
    name: str = "thesis_experiment"
    description: str = "Computational Intelligence Sentiment Analysis Experiment"
    experiment_type: ExperimentType = ExperimentType.FULL_BENCHMARK

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Output settings
    output_dir: str = "./results"
    save_models: bool = True
    save_predictions: bool = False
    generate_latex: bool = True
    generate_plots: bool = True

    # Reproducibility
    random_seed: int = 42

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'experiment_type': self.experiment_type.value,
            'model': {
                'use_logreg': self.model.use_logreg,
                'use_svm': self.model.use_svm,
                'use_tfidf': self.model.use_tfidf,
                'use_bert': self.model.use_bert,
                'use_hybrid_nn': self.model.use_hybrid_nn,
                'use_fuzzy': self.model.use_fuzzy,
                'fuzzy_defuzz_method': self.model.fuzzy_defuzz_method,
                'fuzzy_mf_type': self.model.fuzzy_mf_type,
                'ensemble_method': self.model.ensemble_method,
                'optimize_weights': self.model.optimize_weights,
            },
            'optimization': {
                'optimizer_type': self.optimization.optimizer_type.value,
                'population_size': self.optimization.population_size,
                'max_iterations': self.optimization.max_iterations,
                'c1': self.optimization.c1,
                'c2': self.optimization.c2,
                'w_start': self.optimization.w_start,
                'w_end': self.optimization.w_end,
                'use_adaptive': self.optimization.use_adaptive,
                'seed': self.optimization.seed,
                'n_runs': self.optimization.n_runs,
            },
            'evaluation': {
                'datasets': self.evaluation.datasets,
                'max_samples_per_dataset': self.evaluation.max_samples_per_dataset,
                'run_cross_domain': self.evaluation.run_cross_domain,
                'primary_metric': self.evaluation.primary_metric,
            },
            'output_dir': self.output_dir,
            'random_seed': self.random_seed,
        }

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        data = self.to_dict()

        if (filepath.suffix == '.yaml' or filepath.suffix == '.yml') and YAML_AVAILABLE:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        filepath = Path(filepath)

        if (filepath.suffix == '.yaml' or filepath.suffix == '.yml') and YAML_AVAILABLE:
            with open(filepath) as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath) as f:
                data = json.load(f)

        # Create config from dict
        config = cls(
            name=data.get('name', 'experiment'),
            description=data.get('description', ''),
            experiment_type=ExperimentType(data.get('experiment_type', 'full_benchmark')),
            output_dir=data.get('output_dir', './results'),
            random_seed=data.get('random_seed', 42),
        )

        # Load model config
        if 'model' in data:
            m = data['model']
            config.model = ModelConfig(
                use_logreg=m.get('use_logreg', True),
                use_svm=m.get('use_svm', True),
                use_tfidf=m.get('use_tfidf', True),
                use_bert=m.get('use_bert', False),
                use_fuzzy=m.get('use_fuzzy', True),
                fuzzy_defuzz_method=m.get('fuzzy_defuzz_method', 'centroid'),
                ensemble_method=m.get('ensemble_method', 'weighted_average'),
                optimize_weights=m.get('optimize_weights', True),
            )

        # Load optimization config
        if 'optimization' in data:
            o = data['optimization']
            config.optimization = OptimizationConfig(
                optimizer_type=OptimizerType(o.get('optimizer_type', 'pso')),
                population_size=o.get('population_size', 30),
                max_iterations=o.get('max_iterations', 50),
                c1=o.get('c1', 2.0),
                c2=o.get('c2', 2.0),
                seed=o.get('seed', 42),
                n_runs=o.get('n_runs', 5),
            )

        # Load evaluation config
        if 'evaluation' in data:
            e = data['evaluation']
            config.evaluation = EvaluationConfig(
                datasets=e.get('datasets', ['synthetic']),
                max_samples_per_dataset=e.get('max_samples_per_dataset', 1000),
                run_cross_domain=e.get('run_cross_domain', True),
                primary_metric=e.get('primary_metric', 'f1_macro'),
            )

        return config


# =============================================================================
# Preset Configurations
# =============================================================================

def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing."""
    config = ExperimentConfig(
        name="quick_test",
        description="Quick test run with minimal settings",
        experiment_type=ExperimentType.SINGLE_DATASET,
    )
    config.model.use_bert = False
    config.model.use_hybrid_nn = False
    config.optimization.max_iterations = 10
    config.optimization.population_size = 10
    config.optimization.n_runs = 1
    config.evaluation.max_samples_per_dataset = 100
    config.evaluation.run_cross_domain = False
    return config


def get_full_benchmark_config() -> ExperimentConfig:
    """Get configuration for full thesis benchmark."""
    config = ExperimentConfig(
        name="full_benchmark",
        description="Complete thesis benchmark on all datasets",
        experiment_type=ExperimentType.FULL_BENCHMARK,
    )
    config.evaluation.max_samples_per_dataset = 5000
    config.optimization.n_runs = 10
    config.optimization.max_iterations = 100
    return config


def get_cross_domain_config() -> ExperimentConfig:
    """Get configuration for cross-domain analysis."""
    config = ExperimentConfig(
        name="cross_domain",
        description="Cross-domain generalization study",
        experiment_type=ExperimentType.CROSS_DOMAIN,
    )
    config.evaluation.run_cross_domain = True
    config.evaluation.max_samples_per_dataset = 2000
    return config


def get_ablation_config() -> ExperimentConfig:
    """Get configuration for ablation study."""
    config = ExperimentConfig(
        name="ablation_study",
        description="Ablation study on model components",
        experiment_type=ExperimentType.ABLATION_STUDY,
    )
    config.evaluation.max_samples_per_dataset = 1000
    config.optimization.n_runs = 5
    return config
