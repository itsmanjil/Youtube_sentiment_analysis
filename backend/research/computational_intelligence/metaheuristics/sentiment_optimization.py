"""
Sentiment Analysis Optimization Problems

This module provides optimization problem definitions specific to
sentiment analysis, enabling:
1. Ensemble weight optimization
2. Fuzzy parameter tuning
3. Hyperparameter optimization
4. Multi-objective optimization (accuracy vs speed)

Author: [Your Name]
"""

from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
import time

from .base import OptimizationProblem, ObjectiveType, Solution, OptimizationResult
from .pso import ParticleSwarmOptimizer, PSOConfig
from .mopso import MultiObjectivePSO, MOPSOConfig
from .nsga2 import NSGA2, NSGA2Config


class EnsembleWeightProblem(OptimizationProblem):
    """
    Optimization problem for ensemble weight tuning.

    Finds optimal weights for combining multiple sentiment models
    to maximize classification accuracy.

    Decision Variables:
        w = [w1, w2, ..., wn] where wi is weight for model i

    Objective:
        Maximize: accuracy(ensemble_predict(models, normalize(w), X), y)

    Constraint:
        sum(w) = 1 (handled by normalization)

    Parameters
    ----------
    models : list
        List of sentiment model instances
    X_val : array-like
        Validation texts
    y_val : array-like
        Validation labels
    metric : str
        Metric to optimize: 'accuracy', 'f1', 'weighted_f1'
    """

    def __init__(
        self,
        models: List[Any],
        X_val: List[str],
        y_val: List[str],
        metric: str = 'accuracy'
    ):
        n_models = len(models)
        super().__init__(
            n_variables=n_models,
            bounds=[(0.01, 1.0)] * n_models,  # Avoid zero weights
            objectives=[metric],
            objective_types=[ObjectiveType.MAXIMIZE],
            name='EnsembleWeightOptimization'
        )

        self.models = models
        self.X_val = X_val
        self.y_val = np.array(y_val)
        self.metric = metric

        # Cache predictions for efficiency
        self._predictions_cache: Dict[int, np.ndarray] = {}
        self._probs_cache: Dict[int, np.ndarray] = {}

        # Pre-compute model predictions
        self._precompute_predictions()

    def _precompute_predictions(self) -> None:
        """Pre-compute predictions from all models."""
        label_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}

        for i, model in enumerate(self.models):
            probs = []
            for text in self.X_val:
                try:
                    result = model.analyze(text)
                    p = result.probs if hasattr(result, 'probs') else result.get('probs', {})
                    probs.append([
                        p.get('Negative', 0.33),
                        p.get('Neutral', 0.34),
                        p.get('Positive', 0.33)
                    ])
                except Exception:
                    probs.append([0.33, 0.34, 0.33])

            self._probs_cache[i] = np.array(probs)

        # Convert labels to numeric
        self._y_numeric = np.array([label_map.get(y, 1) for y in self.y_val])

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate ensemble with given weights."""
        # Normalize weights to sum to 1
        weights = x / np.sum(x)

        # Weighted combination of predictions
        combined_probs = np.zeros((len(self.X_val), 3))
        for i, w in enumerate(weights):
            combined_probs += w * self._probs_cache[i]

        # Get predicted labels
        predictions = np.argmax(combined_probs, axis=1)

        # Compute metric (return negative for minimization)
        if self.metric == 'accuracy':
            score = np.mean(predictions == self._y_numeric)
        elif self.metric == 'f1':
            score = self._compute_f1(predictions, self._y_numeric)
        else:
            score = np.mean(predictions == self._y_numeric)

        # Return negative because we minimize
        return -score

    def _compute_f1(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute macro F1 score."""
        f1_scores = []
        for c in range(3):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)

        return np.mean(f1_scores)


class FuzzyParameterProblem(OptimizationProblem):
    """
    Optimization problem for fuzzy sentiment classifier parameters.

    Optimizes membership function parameters to improve classification.

    Decision Variables (for 3-class Gaussian MFs):
        x = [neg_mean, neg_sigma, neu_mean, neu_sigma, pos_mean, pos_sigma]

    Constraints:
        neg_mean < neu_mean < pos_mean
        sigma > 0
    """

    def __init__(
        self,
        classifier_factory: Callable,
        X_val: List[str],
        y_val: List[str],
        base_models: Dict[str, Any]
    ):
        # 6 parameters: mean and sigma for each of 3 classes
        super().__init__(
            n_variables=6,
            bounds=[
                (0.0, 0.3),   # neg_mean
                (0.05, 0.3),  # neg_sigma
                (0.3, 0.7),   # neu_mean
                (0.05, 0.3),  # neu_sigma
                (0.7, 1.0),   # pos_mean
                (0.05, 0.3),  # pos_sigma
            ],
            objectives=['accuracy'],
            objective_types=[ObjectiveType.MAXIMIZE],
            name='FuzzyParameterOptimization'
        )

        self.classifier_factory = classifier_factory
        self.X_val = X_val
        self.y_val = y_val
        self.base_models = base_models

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate fuzzy classifier with given parameters."""
        neg_mean, neg_sigma, neu_mean, neu_sigma, pos_mean, pos_sigma = x

        # Create classifier with these parameters
        # (This would require modifying the FuzzySentimentClassifier to accept custom MF params)

        # For now, return negative accuracy placeholder
        # In actual implementation, create classifier and evaluate
        return -0.5  # Placeholder


class MultiObjectiveSentimentProblem(OptimizationProblem):
    """
    Multi-objective optimization for sentiment analysis.

    Objectives:
    1. Maximize accuracy
    2. Minimize inference time

    This finds Pareto-optimal configurations balancing
    accuracy and computational efficiency.
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        X_val: List[str],
        y_val: List[str],
        model_factory: Callable
    ):
        n_configs = len(model_configs)
        super().__init__(
            n_variables=n_configs,
            bounds=[(0, 1)] * n_configs,  # Include/exclude each config
            objectives=['accuracy', 'inference_time'],
            objective_types=[ObjectiveType.MAXIMIZE, ObjectiveType.MINIMIZE],
            name='MultiObjectiveSentiment'
        )

        self.model_configs = model_configs
        self.X_val = X_val
        self.y_val = y_val
        self.model_factory = model_factory

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate configuration (returns [neg_accuracy, time])."""
        # Select models where x > 0.5
        selected = [i for i, xi in enumerate(x) if xi > 0.5]

        if len(selected) == 0:
            return np.array([1.0, 1.0])  # Worst case

        # Build ensemble with selected models
        # ... (implementation depends on model factory)

        # Placeholder return
        accuracy = 0.8 - 0.1 * len(selected)  # More models = lower accuracy (example)
        time_cost = 0.1 * len(selected)  # More models = higher time

        return np.array([-accuracy, time_cost])  # Negative accuracy for minimization


class EnsembleWeightOptimizer:
    """
    High-level optimizer for ensemble weights using PSO.

    Example
    -------
    >>> optimizer = EnsembleWeightOptimizer(
    ...     models=[logreg, svm, tfidf],
    ...     algorithm='pso'
    ... )
    >>> best_weights = optimizer.optimize(X_val, y_val)
    >>> print(f"Optimal weights: {best_weights}")
    """

    def __init__(
        self,
        models: List[Any],
        algorithm: str = 'pso',
        population_size: int = 30,
        max_iterations: int = 50,
        seed: int = 42,
        verbose: bool = False
    ):
        self.models = models
        self.algorithm = algorithm
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.seed = seed
        self.verbose = verbose

        self.result: OptimizationResult = None

    def optimize(
        self,
        X_val: List[str],
        y_val: List[str],
        metric: str = 'accuracy'
    ) -> np.ndarray:
        """
        Find optimal ensemble weights.

        Parameters
        ----------
        X_val : list
            Validation texts
        y_val : list
            Validation labels
        metric : str
            Optimization metric

        Returns
        -------
        np.ndarray
            Normalized optimal weights
        """
        # Create problem
        problem = EnsembleWeightProblem(
            models=self.models,
            X_val=X_val,
            y_val=y_val,
            metric=metric
        )

        # Select optimizer
        if self.algorithm == 'pso':
            optimizer = ParticleSwarmOptimizer(
                problem=problem,
                population_size=self.population_size,
                max_iterations=self.max_iterations,
                seed=self.seed,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Run optimization
        self.result = optimizer.optimize()

        # Extract and normalize weights
        weights = self.result.best_solution.position
        weights = weights / np.sum(weights)

        return weights

    def get_convergence_history(self) -> List[float]:
        """Get optimization convergence history."""
        if self.result is None:
            return []
        # Convert from negative (minimization) to positive (accuracy)
        return [-f for f in self.result.convergence_history]


class FuzzyParameterOptimizer:
    """
    Optimizer for fuzzy classifier parameters.

    Tunes membership function parameters for optimal classification.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        algorithm: str = 'pso',
        population_size: int = 30,
        max_iterations: int = 50,
        verbose: bool = False
    ):
        self.base_models = base_models
        self.algorithm = algorithm
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbose = verbose

    def optimize(
        self,
        X_val: List[str],
        y_val: List[str]
    ) -> Dict[str, Any]:
        """
        Find optimal fuzzy parameters.

        Returns
        -------
        dict
            Optimal MF parameters
        """
        # Implementation would go here
        # Returns optimized parameters for membership functions
        return {
            'negative': {'mean': 0.15, 'sigma': 0.15},
            'neutral': {'mean': 0.5, 'sigma': 0.15},
            'positive': {'mean': 0.85, 'sigma': 0.15},
        }


class HyperparameterTuner:
    """
    General hyperparameter tuner using metaheuristics.

    Supports both single-objective and multi-objective optimization.

    Example
    -------
    >>> tuner = HyperparameterTuner(
    ...     param_space={
    ...         'learning_rate': (0.001, 0.1, 'log'),
    ...         'batch_size': (16, 128, 'int'),
    ...         'dropout': (0.1, 0.5, 'float'),
    ...     },
    ...     objective_fn=train_and_evaluate,
    ...     algorithm='pso'
    ... )
    >>> best_params = tuner.optimize()
    """

    def __init__(
        self,
        param_space: Dict[str, Tuple],
        objective_fn: Callable,
        algorithm: str = 'pso',
        n_objectives: int = 1,
        population_size: int = 20,
        max_iterations: int = 30,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize hyperparameter tuner.

        Parameters
        ----------
        param_space : dict
            Parameter name -> (lower, upper, type) where type is
            'float', 'int', or 'log' (log-uniform)
        objective_fn : callable
            Function that takes params dict and returns objective value(s)
        algorithm : str
            'pso', 'mopso', or 'nsga2'
        n_objectives : int
            Number of objectives (1 for single, >1 for multi)
        """
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.algorithm = algorithm
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.seed = seed
        self.verbose = verbose

        # Extract parameter info
        self.param_names = list(param_space.keys())
        self.param_types = [param_space[name][2] for name in self.param_names]

        self.result: OptimizationResult = None
        self.history: List[Dict[str, Any]] = []

    def _create_problem(self) -> OptimizationProblem:
        """Create optimization problem from param space."""
        bounds = []
        for name in self.param_names:
            lower, upper, ptype = self.param_space[name]
            if ptype == 'log':
                bounds.append((np.log(lower), np.log(upper)))
            else:
                bounds.append((lower, upper))

        class HyperparamProblem(OptimizationProblem):
            def __init__(inner_self):
                objectives = ['objective'] if self.n_objectives == 1 else [f'obj_{i}' for i in range(self.n_objectives)]
                obj_types = [ObjectiveType.MINIMIZE] * len(objectives)

                super(HyperparamProblem, inner_self).__init__(
                    n_variables=len(self.param_names),
                    bounds=bounds,
                    objectives=objectives,
                    objective_types=obj_types,
                    name='HyperparameterTuning'
                )

            def evaluate(inner_self, x: np.ndarray):
                # Convert to params dict
                params = self._decode_params(x)

                # Track history
                start = time.time()
                result = self.objective_fn(params)
                elapsed = time.time() - start

                self.history.append({
                    'params': params,
                    'result': result,
                    'time': elapsed
                })

                if self.verbose:
                    print(f"Eval {len(self.history)}: {params} -> {result:.4f}")

                return result

        return HyperparamProblem()

    def _decode_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert optimization vector to parameter dict."""
        params = {}
        for i, name in enumerate(self.param_names):
            value = x[i]
            ptype = self.param_types[i]

            if ptype == 'log':
                value = np.exp(value)
            elif ptype == 'int':
                value = int(round(value))

            params[name] = value

        return params

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns
        -------
        dict
            Best parameters found
        """
        problem = self._create_problem()

        # Select optimizer
        if self.algorithm == 'pso':
            optimizer = ParticleSwarmOptimizer(
                problem=problem,
                population_size=self.population_size,
                max_iterations=self.max_iterations,
                seed=self.seed,
                verbose=False
            )
        elif self.algorithm == 'mopso':
            optimizer = MultiObjectivePSO(
                problem=problem,
                population_size=self.population_size,
                max_iterations=self.max_iterations,
                seed=self.seed,
                verbose=False
            )
        elif self.algorithm == 'nsga2':
            optimizer = NSGA2(
                problem=problem,
                population_size=self.population_size,
                max_iterations=self.max_iterations,
                seed=self.seed,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Run
        self.result = optimizer.optimize()

        # Return best params
        return self._decode_params(self.result.best_solution.position)

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        if self.result is None:
            raise ValueError("Must call optimize() first")
        return self._decode_params(self.result.best_solution.position)

    def get_pareto_front_params(self) -> List[Dict[str, Any]]:
        """Get Pareto front parameters (for multi-objective)."""
        if self.result is None or not self.result.pareto_front:
            return []
        return [self._decode_params(sol.position) for sol in self.result.pareto_front]
