"""
Base Classes for Metaheuristic Optimization

This module provides the foundational abstractions for implementing
bio-inspired optimization algorithms.

Design Pattern:
    Strategy pattern - different optimizers can be swapped while
    maintaining the same interface.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Tuple, Optional, Callable, Any, Union, TypeVar, Generic
)
import numpy as np
from enum import Enum
import time
import json


class ObjectiveType(Enum):
    """Type of optimization objective."""
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


@dataclass
class Solution:
    """
    Represents a candidate solution in the search space.

    Attributes:
        position: The solution vector (decision variables)
        fitness: Objective function value(s)
        velocity: For PSO-based algorithms
        personal_best: Best position found by this particle
        personal_best_fitness: Fitness at personal best
        metadata: Additional algorithm-specific data
    """
    position: np.ndarray
    fitness: Union[float, np.ndarray] = None
    velocity: np.ndarray = None
    personal_best: np.ndarray = None
    personal_best_fitness: Union[float, np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.personal_best is None:
            self.personal_best = self.position.copy()
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)

    def dominates(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another (for multi-objective).

        Pareto dominance: A dominates B if A is no worse in all objectives
        and strictly better in at least one.
        """
        if self.fitness is None or other.fitness is None:
            return False

        self_fit = np.atleast_1d(self.fitness)
        other_fit = np.atleast_1d(other.fitness)

        no_worse = np.all(self_fit <= other_fit)
        strictly_better = np.any(self_fit < other_fit)

        return no_worse and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'position': self.position.tolist(),
            'fitness': self.fitness.tolist() if isinstance(self.fitness, np.ndarray) else self.fitness,
            'metadata': self.metadata,
        }


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.

    Attributes:
        best_solution: The best solution found
        best_fitness: Fitness of the best solution
        pareto_front: For multi-objective, the non-dominated solutions
        convergence_history: Fitness values over iterations
        runtime: Total optimization time in seconds
        iterations: Number of iterations completed
        evaluations: Number of fitness evaluations
        metadata: Additional information
    """
    best_solution: Solution
    best_fitness: Union[float, np.ndarray]
    pareto_front: List[Solution] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    runtime: float = 0.0
    iterations: int = 0
    evaluations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_solution': self.best_solution.to_dict(),
            'best_fitness': self.best_fitness.tolist() if isinstance(self.best_fitness, np.ndarray) else self.best_fitness,
            'convergence_history': self.convergence_history,
            'runtime': self.runtime,
            'iterations': self.iterations,
            'evaluations': self.evaluations,
            'pareto_front_size': len(self.pareto_front),
            'metadata': self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    Subclass this to define your specific optimization problem
    with custom objective function and constraints.

    Example:
        >>> class EnsembleWeightProblem(OptimizationProblem):
        ...     def __init__(self, models, X_val, y_val):
        ...         super().__init__(
        ...             n_variables=len(models),
        ...             bounds=[(0, 1)] * len(models),
        ...             objectives=['accuracy'],
        ...             objective_types=[ObjectiveType.MAXIMIZE]
        ...         )
        ...         self.models = models
        ...         self.X_val = X_val
        ...         self.y_val = y_val
        ...
        ...     def evaluate(self, x):
        ...         weights = x / x.sum()  # Normalize
        ...         predictions = ensemble_predict(self.models, weights, self.X_val)
        ...         return accuracy_score(self.y_val, predictions)
    """

    def __init__(
        self,
        n_variables: int,
        bounds: List[Tuple[float, float]],
        objectives: List[str] = None,
        objective_types: List[ObjectiveType] = None,
        constraints: List[Callable] = None,
        name: str = "OptimizationProblem"
    ):
        """
        Initialize the optimization problem.

        Parameters
        ----------
        n_variables : int
            Number of decision variables
        bounds : list of tuples
            (lower, upper) bounds for each variable
        objectives : list of str
            Names of objective functions
        objective_types : list of ObjectiveType
            Whether to minimize or maximize each objective
        constraints : list of callables
            Constraint functions (should return <= 0 when satisfied)
        name : str
            Problem name for logging
        """
        self.n_variables = n_variables
        self.bounds = bounds
        self.objectives = objectives or ['objective']
        self.objective_types = objective_types or [ObjectiveType.MINIMIZE]
        self.constraints = constraints or []
        self.name = name

        # Validation
        assert len(bounds) == n_variables, "Bounds must match n_variables"
        assert len(self.objectives) == len(self.objective_types), "Objectives and types must match"

        # Convert bounds to arrays for efficient operations
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        # Tracking
        self.evaluation_count = 0

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate the objective function(s) for a solution.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector

        Returns
        -------
        float or np.ndarray
            Objective value(s). For multi-objective, return array.
        """
        pass

    def evaluate_with_penalty(self, x: np.ndarray, penalty_factor: float = 1e6) -> Union[float, np.ndarray]:
        """
        Evaluate with constraint penalty.

        Adds a penalty term for constraint violations.
        """
        self.evaluation_count += 1

        # Compute base fitness
        fitness = self.evaluate(x)

        # Add constraint penalties
        total_penalty = 0
        for constraint in self.constraints:
            violation = constraint(x)
            if violation > 0:
                total_penalty += penalty_factor * violation ** 2

        if isinstance(fitness, np.ndarray):
            return fitness + total_penalty
        return fitness + total_penalty

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if solution satisfies all constraints."""
        # Check bounds
        if np.any(x < self.lower_bounds) or np.any(x > self.upper_bounds):
            return False

        # Check constraints
        for constraint in self.constraints:
            if constraint(x) > 0:
                return False

        return True

    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to valid bounds."""
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def random_solution(self) -> np.ndarray:
        """Generate a random solution within bounds."""
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def random_population(self, size: int) -> List[np.ndarray]:
        """Generate a random population of solutions."""
        return [self.random_solution() for _ in range(size)]

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective problem."""
        return len(self.objectives) > 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_variables={self.n_variables}, "
            f"objectives={self.objectives})"
        )


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    Subclass this to implement specific metaheuristic algorithms.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        population_size: int = 50,
        max_iterations: int = 100,
        seed: int = None,
        verbose: bool = False
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        problem : OptimizationProblem
            The problem to optimize
        population_size : int
            Number of solutions in the population
        max_iterations : int
            Maximum number of iterations
        seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        self.problem = problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # State
        self.population: List[Solution] = []
        self.best_solution: Solution = None
        self.convergence_history: List[float] = []
        self.iteration = 0

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the population."""
        pass

    @abstractmethod
    def iterate(self) -> None:
        """Perform one iteration of the algorithm."""
        pass

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization algorithm.

        Returns
        -------
        OptimizationResult
            The optimization results
        """
        start_time = time.time()

        # Initialize
        self.initialize()

        # Main loop
        for self.iteration in range(self.max_iterations):
            self.iterate()

            # Track convergence
            if self.best_solution is not None:
                best_fit = self.best_solution.fitness
                if isinstance(best_fit, np.ndarray):
                    self.convergence_history.append(float(best_fit[0]))
                else:
                    self.convergence_history.append(float(best_fit))

            # Verbose output
            if self.verbose and (self.iteration + 1) % 10 == 0:
                print(f"Iteration {self.iteration + 1}/{self.max_iterations}, "
                      f"Best fitness: {self.best_solution.fitness}")

        runtime = time.time() - start_time

        return OptimizationResult(
            best_solution=self.best_solution,
            best_fitness=self.best_solution.fitness,
            convergence_history=self.convergence_history,
            runtime=runtime,
            iterations=self.max_iterations,
            evaluations=self.problem.evaluation_count,
            metadata={
                'algorithm': self.__class__.__name__,
                'population_size': self.population_size,
            }
        )

    def _evaluate_solution(self, solution: Solution) -> None:
        """Evaluate a solution and update its fitness."""
        solution.fitness = self.problem.evaluate_with_penalty(solution.position)

    def _update_best(self, solution: Solution) -> None:
        """Update the global best if this solution is better."""
        if self.best_solution is None:
            self.best_solution = Solution(
                position=solution.position.copy(),
                fitness=solution.fitness
            )
        else:
            # For minimization (default)
            if self._is_better(solution.fitness, self.best_solution.fitness):
                self.best_solution = Solution(
                    position=solution.position.copy(),
                    fitness=solution.fitness
                )

    def _is_better(self, fitness1, fitness2) -> bool:
        """Check if fitness1 is better than fitness2."""
        # Handle single objective
        if isinstance(fitness1, (int, float)) and isinstance(fitness2, (int, float)):
            obj_type = self.problem.objective_types[0]
            if obj_type == ObjectiveType.MINIMIZE:
                return fitness1 < fitness2
            else:
                return fitness1 > fitness2

        # Handle multi-objective (use first objective for comparison)
        f1 = fitness1[0] if isinstance(fitness1, np.ndarray) else fitness1
        f2 = fitness2[0] if isinstance(fitness2, np.ndarray) else fitness2

        obj_type = self.problem.objective_types[0]
        if obj_type == ObjectiveType.MINIMIZE:
            return f1 < f2
        return f1 > f2
