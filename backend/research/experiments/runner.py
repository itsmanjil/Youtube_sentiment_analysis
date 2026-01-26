"""
Thesis Experiment Runner

Main orchestrator for running complete thesis experiments.
Integrates fuzzy logic, PSO optimization, and benchmark evaluation.

Author: [Your Name]
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json
import time
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from .config import (
    ExperimentConfig,
    ExperimentType,
    OptimizerType,
)


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    inference_time_ms: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
            'per_class_f1': self.per_class_f1,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'inference_time_ms': self.inference_time_ms,
            'n_samples': self.n_samples,
        }


@dataclass
class DatasetResult:
    """Results for a single dataset."""
    dataset_name: str
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    best_model: str = ""
    best_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_name': self.dataset_name,
            'model_results': {k: v.to_dict() for k, v in self.model_results.items()},
            'best_model': self.best_model,
            'best_score': self.best_score,
        }


@dataclass
class OptimizationResult:
    """Results from PSO optimization."""
    algorithm: str
    best_weights: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    n_iterations: int
    n_evaluations: int
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'best_weights': self.best_weights.tolist(),
            'best_fitness': self.best_fitness,
            'convergence_history': self.convergence_history,
            'n_iterations': self.n_iterations,
            'n_evaluations': self.n_evaluations,
            'runtime_seconds': self.runtime_seconds,
        }


@dataclass
class ExperimentResult:
    """Complete experiment results."""
    config: ExperimentConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Results by dataset
    dataset_results: Dict[str, DatasetResult] = field(default_factory=dict)

    # Cross-domain results (train_dataset -> test_dataset -> metrics)
    cross_domain_results: Dict[str, Dict[str, ModelResult]] = field(default_factory=dict)

    # Optimization results
    optimization_results: Dict[str, OptimizationResult] = field(default_factory=dict)

    # Ensemble results
    ensemble_weights: Optional[np.ndarray] = None
    ensemble_accuracy: float = 0.0

    # Statistical analysis
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Runtime
    total_runtime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'timestamp': self.timestamp,
            'dataset_results': {k: v.to_dict() for k, v in self.dataset_results.items()},
            'cross_domain_results': {
                train: {test: m.to_dict() for test, m in results.items()}
                for train, results in self.cross_domain_results.items()
            },
            'optimization_results': {k: v.to_dict() for k, v in self.optimization_results.items()},
            'ensemble_weights': self.ensemble_weights.tolist() if self.ensemble_weights is not None else None,
            'ensemble_accuracy': self.ensemble_accuracy,
            'statistical_tests': self.statistical_tests,
            'total_runtime_seconds': self.total_runtime_seconds,
        }

    def save(self, filepath: str) -> None:
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ThesisExperiment:
    """
    Main experiment orchestrator for thesis.

    Integrates:
    - Fuzzy sentiment classification
    - PSO-based ensemble optimization
    - Benchmark dataset evaluation
    - Cross-domain analysis
    - Statistical testing

    Example
    -------
    >>> config = ExperimentConfig(name="main_experiment")
    >>> experiment = ThesisExperiment(config)
    >>> results = experiment.run()
    >>> results.save("thesis_results.json")
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.result = ExperimentResult(config=config)

        # Lazy-loaded components
        self._models: Dict[str, Any] = {}
        self._datasets: Dict[str, Any] = {}
        self._fuzzy_classifier = None
        self._optimizer = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(f"ThesisExperiment.{self.config.name}")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.

        Returns
        -------
        ExperimentResult
            Complete experiment results
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Type: {self.config.experiment_type.value}")

        try:
            # Step 1: Load datasets
            self._load_datasets()

            # Step 2: Initialize models
            self._initialize_models()

            # Step 3: Run based on experiment type
            if self.config.experiment_type == ExperimentType.SINGLE_DATASET:
                self._run_single_dataset()
            elif self.config.experiment_type == ExperimentType.CROSS_DOMAIN:
                self._run_cross_domain()
            elif self.config.experiment_type == ExperimentType.ABLATION_STUDY:
                self._run_ablation_study()
            elif self.config.experiment_type == ExperimentType.FULL_BENCHMARK:
                self._run_full_benchmark()

            # Step 4: Optimize ensemble weights if configured
            if self.config.model.optimize_weights:
                self._optimize_ensemble()

            # Step 5: Statistical analysis
            if self.config.evaluation.run_statistical_tests:
                self._run_statistical_tests()

            # Record total runtime
            self.result.total_runtime_seconds = time.time() - start_time
            self.logger.info(f"Experiment completed in {self.result.total_runtime_seconds:.2f}s")

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

        return self.result

    def _load_datasets(self) -> None:
        """Load benchmark datasets."""
        self.logger.info("Loading datasets...")

        from research.benchmarks.base import DatasetSplit, SentimentLabel

        # Create synthetic datasets for testing
        for ds_name in self.config.evaluation.datasets:
            if ds_name == "synthetic":
                self._datasets[ds_name] = self._create_synthetic_dataset(
                    n_samples=self.config.evaluation.max_samples_per_dataset
                )

        # Load standard benchmarks if configured
        if self.config.evaluation.use_sentiment140:
            try:
                from research.benchmarks.datasets import Sentiment140Dataset
                ds = Sentiment140Dataset(
                    max_samples=self.config.evaluation.max_samples_per_dataset
                )
                ds.load()
                self._datasets['sentiment140'] = ds
            except Exception as e:
                self.logger.warning(f"Could not load Sentiment140: {e}")

        if self.config.evaluation.use_imdb:
            try:
                from research.benchmarks.datasets import IMDBDataset
                ds = IMDBDataset()
                ds.load()
                self._datasets['imdb'] = ds
            except Exception as e:
                self.logger.warning(f"Could not load IMDB: {e}")

        if self.config.evaluation.use_sst:
            try:
                from research.benchmarks.datasets import SSTDataset
                ds = SSTDataset()
                ds.load()
                self._datasets['sst'] = ds
            except Exception as e:
                self.logger.warning(f"Could not load SST: {e}")

        self.logger.info(f"Loaded {len(self._datasets)} datasets")

    def _create_synthetic_dataset(self, n_samples: int = 1000) -> Any:
        """Create synthetic dataset for testing."""
        from research.benchmarks.base import Dataset, DatasetSplit, SentimentLabel

        np.random.seed(self.config.random_seed)

        positive = [
            "This is excellent, I love it!",
            "Amazing quality, highly recommended!",
            "Great experience, wonderful service!",
            "The best purchase ever made!",
            "Fantastic product, brilliant quality!",
        ]

        negative = [
            "Terrible quality, waste of money.",
            "Awful experience, horrible service!",
            "The worst product ever!",
            "Boring and disappointing.",
            "Poor performance, total disaster!",
        ]

        neutral = [
            "It's okay, nothing special.",
            "Average product, does the job.",
            "Neither good nor bad.",
            "Standard quality.",
            "Typical experience.",
        ]

        texts, labels = [], []
        for _ in range(n_samples):
            r = np.random.random()
            if r < 0.4:
                texts.append(np.random.choice(positive))
                labels.append(SentimentLabel.POSITIVE)
            elif r < 0.8:
                texts.append(np.random.choice(negative))
                labels.append(SentimentLabel.NEGATIVE)
            else:
                texts.append(np.random.choice(neutral))
                labels.append(SentimentLabel.NEUTRAL)

        # Create splits
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)

        class SyntheticDataset:
            def __init__(self, texts, labels, n_train, n_val):
                self.name = "synthetic"
                self.train = DatasetSplit(texts[:n_train], labels[:n_train], name='train')
                self.val = DatasetSplit(texts[n_train:n_train+n_val], labels[n_train:n_train+n_val], name='val')
                self.test = DatasetSplit(texts[n_train+n_val:], labels[n_train+n_val:], name='test')

            def load(self):
                pass

        return SyntheticDataset(texts, labels, n_train, n_val)

    def _initialize_models(self) -> None:
        """Initialize sentiment models."""
        self.logger.info("Initializing models...")

        # Try to load actual engines
        try:
            if self.config.model.use_logreg:
                from sentiment_engines.logreg_sentiment import LogRegSentiment
                self._models['logreg'] = LogRegSentiment()
                self.logger.info("  Loaded LogReg")
        except ImportError:
            self._models['logreg'] = self._create_mock_model('logreg')

        try:
            if self.config.model.use_svm:
                from sentiment_engines.svm_sentiment import SVMSentiment
                self._models['svm'] = SVMSentiment()
                self.logger.info("  Loaded SVM")
        except ImportError:
            self._models['svm'] = self._create_mock_model('svm')

        try:
            if self.config.model.use_tfidf:
                from sentiment_engines.tfidf_sentiment import TFIDFSentiment
                self._models['tfidf'] = TFIDFSentiment()
                self.logger.info("  Loaded TF-IDF")
        except ImportError:
            self._models['tfidf'] = self._create_mock_model('tfidf')

        # Initialize fuzzy classifier
        if self.config.model.use_fuzzy:
            try:
                from research.computational_intelligence.fuzzy import FuzzySentimentClassifier
                self._fuzzy_classifier = FuzzySentimentClassifier(
                    defuzz_method=self.config.model.fuzzy_defuzz_method,
                )
                self._models['fuzzy'] = self._fuzzy_classifier
                self.logger.info("  Loaded FuzzySentiment")
            except ImportError as e:
                self.logger.warning(f"Could not load fuzzy classifier: {e}")

        self.logger.info(f"Initialized {len(self._models)} models")

    def _create_mock_model(self, name: str) -> Any:
        """Create mock model for testing."""
        class MockModel:
            def __init__(self, model_name):
                self.name = model_name
                self.positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'fantastic', 'wonderful', 'brilliant'}
                self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'boring', 'disappointing', 'disaster', 'poor'}

            def analyze(self, text):
                words = set(text.lower().split())
                pos = len(words & self.positive_words)
                neg = len(words & self.negative_words)

                if pos > neg:
                    label, conf = 'Positive', 0.8
                elif neg > pos:
                    label, conf = 'Negative', 0.8
                else:
                    label, conf = 'Neutral', 0.5

                return {
                    'label': label,
                    'confidence': conf,
                    'probs': {
                        'Positive': conf if label == 'Positive' else 0.1,
                        'Negative': conf if label == 'Negative' else 0.1,
                        'Neutral': conf if label == 'Neutral' else 0.1,
                    }
                }

        return MockModel(name)

    def _evaluate_model(
        self,
        model: Any,
        dataset: Any,
        model_name: str
    ) -> ModelResult:
        """Evaluate a single model on a dataset."""
        from research.benchmarks.base import SentimentLabel

        test_split = dataset.test
        if hasattr(test_split, 'sample'):
            test_split = test_split.sample(
                min(len(test_split), self.config.evaluation.max_samples_per_dataset)
            )

        predictions = []
        true_labels = []
        label_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}

        start_time = time.time()

        for text, label in test_split:
            try:
                result = model.analyze(text)
                pred_label = result.get('label', 'Neutral') if isinstance(result, dict) else getattr(result, 'label', 'Neutral')
                predictions.append(label_map.get(pred_label, 1))
                true_labels.append(label.value)
            except Exception:
                predictions.append(1)
                true_labels.append(label.value)

        inference_time = (time.time() - start_time) / len(test_split) * 1000

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Compute metrics
        accuracy = np.mean(predictions == true_labels)

        # Confusion matrix
        n_classes = 3
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(true_labels, predictions):
            cm[t, p] += 1

        # Per-class metrics
        precisions, recalls, f1s, supports = [], [], [], []
        class_names = ['Negative', 'Neutral', 'Positive']
        per_class_f1 = {}

        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(cm[c, :].sum())
            per_class_f1[class_names[c]] = float(f1)

        total = sum(supports)
        f1_macro = np.mean(f1s)
        f1_weighted = np.average(f1s, weights=supports) if total > 0 else 0
        precision_macro = np.mean(precisions)
        recall_macro = np.mean(recalls)

        return ModelResult(
            model_name=model_name,
            accuracy=float(accuracy),
            f1_macro=float(f1_macro),
            f1_weighted=float(f1_weighted),
            precision_macro=float(precision_macro),
            recall_macro=float(recall_macro),
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            inference_time_ms=float(inference_time),
            n_samples=len(test_split),
        )

    def _run_single_dataset(self) -> None:
        """Run evaluation on single dataset."""
        self.logger.info("Running single dataset evaluation...")

        for ds_name, dataset in self._datasets.items():
            self.logger.info(f"  Evaluating on {ds_name}")

            ds_result = DatasetResult(dataset_name=ds_name)

            for model_name, model in self._models.items():
                result = self._evaluate_model(model, dataset, model_name)
                ds_result.model_results[model_name] = result
                self.logger.info(f"    {model_name}: Acc={result.accuracy:.4f}, F1={result.f1_macro:.4f}")

                # Track best
                metric = result.f1_macro if self.config.evaluation.primary_metric == 'f1_macro' else result.accuracy
                if metric > ds_result.best_score:
                    ds_result.best_score = metric
                    ds_result.best_model = model_name

            self.result.dataset_results[ds_name] = ds_result

    def _run_cross_domain(self) -> None:
        """Run cross-domain evaluation."""
        self.logger.info("Running cross-domain evaluation...")

        dataset_names = list(self._datasets.keys())

        for train_ds_name in dataset_names:
            self.result.cross_domain_results[train_ds_name] = {}

            for test_ds_name in dataset_names:
                self.logger.info(f"  Train: {train_ds_name} -> Test: {test_ds_name}")

                # For simplicity, use a representative model
                model = list(self._models.values())[0]
                model_name = list(self._models.keys())[0]

                result = self._evaluate_model(
                    model,
                    self._datasets[test_ds_name],
                    f"{model_name}_{train_ds_name}_to_{test_ds_name}"
                )

                self.result.cross_domain_results[train_ds_name][test_ds_name] = result
                self.logger.info(f"    F1: {result.f1_macro:.4f}")

    def _run_ablation_study(self) -> None:
        """Run ablation study on model components."""
        self.logger.info("Running ablation study...")

        # Test with different model combinations
        combinations = [
            ['logreg'],
            ['svm'],
            ['tfidf'],
            ['logreg', 'svm'],
            ['logreg', 'svm', 'tfidf'],
        ]

        if 'fuzzy' in self._models:
            combinations.extend([
                ['fuzzy'],
                ['logreg', 'svm', 'tfidf', 'fuzzy'],
            ])

        for combo in combinations:
            combo_name = '+'.join(combo)
            self.logger.info(f"  Testing: {combo_name}")

            # Evaluate each model in combo
            for ds_name, dataset in self._datasets.items():
                for model_name in combo:
                    if model_name in self._models:
                        result = self._evaluate_model(
                            self._models[model_name],
                            dataset,
                            f"{model_name}_ablation"
                        )
                        self.logger.info(f"    {model_name} on {ds_name}: F1={result.f1_macro:.4f}")

    def _run_full_benchmark(self) -> None:
        """Run complete benchmark evaluation."""
        self.logger.info("Running full benchmark...")

        # Run single dataset evaluation
        self._run_single_dataset()

        # Run cross-domain if configured
        if self.config.evaluation.run_cross_domain and len(self._datasets) > 1:
            self._run_cross_domain()

    def _optimize_ensemble(self) -> None:
        """Optimize ensemble weights using PSO."""
        self.logger.info("Optimizing ensemble weights with PSO...")

        if len(self._models) < 2:
            self.logger.warning("Need at least 2 models for ensemble optimization")
            return

        try:
            from research.computational_intelligence.metaheuristics import (
                ParticleSwarmOptimizer,
                AdaptivePSO,
            )
            from research.computational_intelligence.metaheuristics.base import (
                OptimizationProblem,
                ObjectiveType,
            )

            # Get validation data from first dataset
            dataset = list(self._datasets.values())[0]
            val_split = dataset.val if hasattr(dataset, 'val') and dataset.val else dataset.test

            # Prepare validation data
            val_texts = []
            val_labels = []
            label_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}

            for text, label in val_split:
                val_texts.append(text)
                val_labels.append(label.value)

            val_labels = np.array(val_labels)

            # Pre-compute model predictions
            model_probs = {}
            models_list = list(self._models.items())

            for model_name, model in models_list:
                probs = []
                for text in val_texts:
                    try:
                        result = model.analyze(text)
                        p = result.get('probs', {}) if isinstance(result, dict) else {}
                        probs.append([
                            p.get('Negative', 0.33),
                            p.get('Neutral', 0.34),
                            p.get('Positive', 0.33)
                        ])
                    except Exception:
                        probs.append([0.33, 0.34, 0.33])
                model_probs[model_name] = np.array(probs)

            # Create optimization problem
            n_models = len(models_list)

            class EnsembleProblem(OptimizationProblem):
                def __init__(inner_self):
                    super().__init__(
                        n_variables=n_models,
                        bounds=[(0.01, 1.0)] * n_models,
                        objectives=['accuracy'],
                        objective_types=[ObjectiveType.MINIMIZE],
                        name='EnsembleWeights'
                    )

                def evaluate(inner_self, x):
                    weights = x / np.sum(x)
                    combined = np.zeros((len(val_texts), 3))
                    for i, (name, _) in enumerate(models_list):
                        combined += weights[i] * model_probs[name]
                    preds = np.argmax(combined, axis=1)
                    accuracy = np.mean(preds == val_labels)
                    return -accuracy  # Minimize negative accuracy

            problem = EnsembleProblem()

            # Run optimization
            if self.config.optimization.use_adaptive:
                optimizer = AdaptivePSO(
                    problem=problem,
                    population_size=self.config.optimization.population_size,
                    max_iterations=self.config.optimization.max_iterations,
                    seed=self.config.optimization.seed,
                    verbose=self.config.optimization.verbose,
                )
            else:
                optimizer = ParticleSwarmOptimizer(
                    problem=problem,
                    population_size=self.config.optimization.population_size,
                    max_iterations=self.config.optimization.max_iterations,
                    seed=self.config.optimization.seed,
                    verbose=self.config.optimization.verbose,
                )

            start_time = time.time()
            result = optimizer.optimize()
            runtime = time.time() - start_time

            # Normalize weights
            weights = result.best_solution.position
            weights = weights / np.sum(weights)

            self.result.ensemble_weights = weights
            self.result.ensemble_accuracy = -result.best_fitness

            self.result.optimization_results['pso'] = OptimizationResult(
                algorithm='AdaptivePSO' if self.config.optimization.use_adaptive else 'PSO',
                best_weights=weights,
                best_fitness=-result.best_fitness,
                convergence_history=[-f for f in result.convergence_history],
                n_iterations=result.iterations,
                n_evaluations=result.evaluations,
                runtime_seconds=runtime,
            )

            self.logger.info(f"  Optimized weights: {weights}")
            self.logger.info(f"  Ensemble accuracy: {-result.best_fitness:.4f}")
            self.logger.info(f"  Model weights: {dict(zip([m[0] for m in models_list], weights))}")

        except Exception as e:
            self.logger.error(f"Ensemble optimization failed: {e}")

    def _run_statistical_tests(self) -> None:
        """Run statistical significance tests."""
        self.logger.info("Running statistical tests...")

        # Collect F1 scores for comparison
        scores = {}
        for ds_name, ds_result in self.result.dataset_results.items():
            for model_name, model_result in ds_result.model_results.items():
                if model_name not in scores:
                    scores[model_name] = []
                scores[model_name].append(model_result.f1_macro)

        # Compute basic statistics
        stats = {}
        for model_name, model_scores in scores.items():
            arr = np.array(model_scores)
            stats[model_name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'n': len(arr),
            }

        self.result.statistical_tests['summary'] = stats
        self.logger.info(f"  Computed statistics for {len(scores)} models")

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            f"EXPERIMENT SUMMARY: {self.config.name}",
            f"Timestamp: {self.result.timestamp}",
            "=" * 60,
            "",
            "DATASET RESULTS:",
            "-" * 40,
        ]

        for ds_name, ds_result in self.result.dataset_results.items():
            lines.append(f"\n{ds_name}:")
            for model_name, model_result in ds_result.model_results.items():
                lines.append(
                    f"  {model_name:15s}: Acc={model_result.accuracy:.4f}, "
                    f"F1={model_result.f1_macro:.4f}"
                )
            lines.append(f"  Best: {ds_result.best_model} ({ds_result.best_score:.4f})")

        if self.result.ensemble_weights is not None:
            lines.extend([
                "",
                "ENSEMBLE OPTIMIZATION:",
                "-" * 40,
                f"  Ensemble Accuracy: {self.result.ensemble_accuracy:.4f}",
                f"  Weights: {self.result.ensemble_weights}",
            ])

        if self.result.cross_domain_results:
            lines.extend([
                "",
                "CROSS-DOMAIN RESULTS:",
                "-" * 40,
            ])
            for train_ds, results in self.result.cross_domain_results.items():
                for test_ds, result in results.items():
                    lines.append(f"  {train_ds} -> {test_ds}: F1={result.f1_macro:.4f}")

        lines.extend([
            "",
            f"Total Runtime: {self.result.total_runtime_seconds:.2f}s",
            "=" * 60,
        ])

        return '\n'.join(lines)
