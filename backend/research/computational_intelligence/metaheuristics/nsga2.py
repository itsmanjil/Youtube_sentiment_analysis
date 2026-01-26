"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II

A fast and elitist multi-objective evolutionary algorithm that uses:
1. Fast non-dominated sorting
2. Crowding distance for diversity
3. Binary tournament selection
4. Simulated Binary Crossover (SBX)
5. Polynomial mutation

Reference:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II".
    IEEE TEVC.

Author: [Your Name]
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .base import (
    Optimizer,
    OptimizationProblem,
    OptimizationResult,
    Solution,
)


@dataclass
class NSGA2Config:
    """Configuration for NSGA-II algorithm."""
    # Crossover
    crossover_prob: float = 0.9
    crossover_eta: float = 20  # SBX distribution index

    # Mutation
    mutation_prob: float = None  # Default: 1/n_variables
    mutation_eta: float = 20    # Polynomial mutation index

    # Selection
    tournament_size: int = 2


class NSGA2(Optimizer):
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    NSGA-II is one of the most popular multi-objective evolutionary
    algorithms, known for its computational efficiency and good
    convergence properties.

    Key Operations:
    1. Non-dominated Sorting: Ranks solutions into fronts
    2. Crowding Distance: Measures solution spacing within fronts
    3. Selection: Prefers lower rank and higher crowding distance
    4. SBX Crossover: Simulated binary crossover for offspring
    5. Polynomial Mutation: Maintains diversity

    Parameters
    ----------
    problem : OptimizationProblem
        Multi-objective optimization problem
    population_size : int
        Population size (should be even)
    max_iterations : int
        Number of generations
    config : NSGA2Config
        Algorithm configuration
    seed : int
        Random seed
    verbose : bool
        Print progress

    Example
    -------
    >>> problem = MultiObjectiveProblem()
    >>> nsga2 = NSGA2(problem, population_size=100, max_iterations=200)
    >>> result = nsga2.optimize()
    >>> print(f"Pareto front: {len(result.pareto_front)} solutions")
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        population_size: int = 100,
        max_iterations: int = 100,
        config: NSGA2Config = None,
        seed: int = None,
        verbose: bool = False
    ):
        # Ensure even population size
        population_size = population_size + (population_size % 2)
        super().__init__(problem, population_size, max_iterations, seed, verbose)

        self.config = config or NSGA2Config()

        # Default mutation probability
        if self.config.mutation_prob is None:
            self.config.mutation_prob = 1.0 / problem.n_variables

        # Storage for ranks and crowding
        self.ranks: List[int] = []
        self.crowding: List[float] = []

    def initialize(self) -> None:
        """Initialize random population."""
        self.population = []

        for _ in range(self.population_size):
            position = self.problem.random_solution()
            individual = Solution(position=position)

            # Evaluate
            individual.fitness = self.problem.evaluate_with_penalty(individual.position)
            self.population.append(individual)

        # Perform initial ranking
        self._fast_non_dominated_sort()
        self._compute_crowding_distance()

        # Set best (first front, first solution)
        self.best_solution = self.population[0]

    def iterate(self) -> None:
        """Perform one generation of NSGA-II."""
        # Create offspring population
        offspring = self._create_offspring()

        # Combine parent and offspring
        combined = self.population + offspring

        # Evaluate offspring
        for ind in offspring:
            if ind.fitness is None:
                ind.fitness = self.problem.evaluate_with_penalty(ind.position)

        # Non-dominated sorting on combined population
        fronts = self._fast_non_dominated_sort_combined(combined)

        # Select next generation
        self.population = []
        front_idx = 0

        while len(self.population) + len(fronts[front_idx]) <= self.population_size:
            # Add entire front
            for idx in fronts[front_idx]:
                self.population.append(combined[idx])
            front_idx += 1

            if front_idx >= len(fronts):
                break

        # Fill remaining slots using crowding distance
        if len(self.population) < self.population_size:
            remaining = self.population_size - len(self.population)
            last_front = [combined[idx] for idx in fronts[front_idx]]

            # Compute crowding for last front
            crowding = self._crowding_distance_for_front(last_front)

            # Sort by crowding (descending) and select
            sorted_idx = np.argsort(crowding)[::-1]
            for i in range(remaining):
                self.population.append(last_front[sorted_idx[i]])

        # Update ranks and crowding for new population
        self._fast_non_dominated_sort()
        self._compute_crowding_distance()

        # Update best solution
        self.best_solution = self.population[0]

    def _create_offspring(self) -> List[Solution]:
        """Create offspring through selection, crossover, and mutation."""
        offspring = []

        for _ in range(self.population_size // 2):
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if np.random.random() < self.config.crossover_prob:
                child1_pos, child2_pos = self._sbx_crossover(
                    parent1.position, parent2.position
                )
            else:
                child1_pos = parent1.position.copy()
                child2_pos = parent2.position.copy()

            # Mutation
            child1_pos = self._polynomial_mutation(child1_pos)
            child2_pos = self._polynomial_mutation(child2_pos)

            # Create offspring solutions
            offspring.append(Solution(position=child1_pos))
            offspring.append(Solution(position=child2_pos))

        return offspring

    def _tournament_select(self) -> Solution:
        """Binary tournament selection based on rank and crowding."""
        candidates = np.random.choice(
            len(self.population),
            size=self.config.tournament_size,
            replace=False
        )

        best_idx = candidates[0]
        for idx in candidates[1:]:
            # Prefer lower rank
            if self.ranks[idx] < self.ranks[best_idx]:
                best_idx = idx
            # If same rank, prefer higher crowding
            elif self.ranks[idx] == self.ranks[best_idx]:
                if self.crowding[idx] > self.crowding[best_idx]:
                    best_idx = idx

        return self.population[best_idx]

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        eta = self.config.crossover_eta
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    lb = self.problem.lower_bounds[i]
                    ub = self.problem.upper_bounds[i]

                    # Compute beta
                    beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))

                    u = np.random.random()
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = np.clip(c1, lb, ub)
                    child2[i] = np.clip(c2, lb, ub)
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]

        return child1, child2

    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        eta = self.config.mutation_eta
        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.random() < self.config.mutation_prob:
                y = individual[i]
                lb = self.problem.lower_bounds[i]
                ub = self.problem.upper_bounds[i]

                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)

                u = np.random.random()

                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1))
                    deltaq = val ** (1.0 / (eta + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** (1.0 / (eta + 1))

                mutated[i] = np.clip(y + deltaq * (ub - lb), lb, ub)

        return mutated

    def _fast_non_dominated_sort(self) -> List[List[int]]:
        """Fast non-dominated sorting for current population."""
        n = len(self.population)
        self.ranks = [0] * n

        # Domination counts and dominated sets
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]

        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(self.population[i], self.population[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(self.population[j], self.population[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                self.ranks[i] = 0
                fronts[0].append(i)

        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.ranks[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _fast_non_dominated_sort_combined(
        self,
        combined: List[Solution]
    ) -> List[List[int]]:
        """Fast non-dominated sorting for combined population."""
        n = len(combined)

        # Domination counts and dominated sets
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]

        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(combined[i], combined[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(combined[j], combined[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """Check if sol1 dominates sol2."""
        fit1 = np.atleast_1d(sol1.fitness)
        fit2 = np.atleast_1d(sol2.fitness)

        return np.all(fit1 <= fit2) and np.any(fit1 < fit2)

    def _compute_crowding_distance(self) -> None:
        """Compute crowding distance for current population."""
        n = len(self.population)
        self.crowding = [0.0] * n

        if n == 0:
            return

        n_obj = len(np.atleast_1d(self.population[0].fitness))
        fitness_matrix = np.array([
            np.atleast_1d(sol.fitness) for sol in self.population
        ])

        for obj in range(n_obj):
            sorted_idx = np.argsort(fitness_matrix[:, obj])

            # Boundary solutions
            self.crowding[sorted_idx[0]] = np.inf
            self.crowding[sorted_idx[-1]] = np.inf

            obj_range = (fitness_matrix[sorted_idx[-1], obj] -
                        fitness_matrix[sorted_idx[0], obj])

            if obj_range > 0:
                for i in range(1, n - 1):
                    self.crowding[sorted_idx[i]] += (
                        fitness_matrix[sorted_idx[i + 1], obj] -
                        fitness_matrix[sorted_idx[i - 1], obj]
                    ) / obj_range

    def _crowding_distance_for_front(self, front: List[Solution]) -> np.ndarray:
        """Compute crowding distance for a specific front."""
        n = len(front)
        crowding = np.zeros(n)

        if n <= 2:
            return np.full(n, np.inf)

        n_obj = len(np.atleast_1d(front[0].fitness))
        fitness_matrix = np.array([
            np.atleast_1d(sol.fitness) for sol in front
        ])

        for obj in range(n_obj):
            sorted_idx = np.argsort(fitness_matrix[:, obj])

            crowding[sorted_idx[0]] = np.inf
            crowding[sorted_idx[-1]] = np.inf

            obj_range = (fitness_matrix[sorted_idx[-1], obj] -
                        fitness_matrix[sorted_idx[0], obj])

            if obj_range > 0:
                for i in range(1, n - 1):
                    crowding[sorted_idx[i]] += (
                        fitness_matrix[sorted_idx[i + 1], obj] -
                        fitness_matrix[sorted_idx[i - 1], obj]
                    ) / obj_range

        return crowding

    def optimize(self) -> OptimizationResult:
        """Run NSGA-II and return result with Pareto front."""
        result = super().optimize()

        # Get Pareto front (rank 0 solutions)
        pareto_front = [
            self.population[i] for i in range(len(self.population))
            if self.ranks[i] == 0
        ]
        result.pareto_front = pareto_front

        result.metadata.update({
            'pareto_front_size': len(pareto_front),
            'crossover_prob': self.config.crossover_prob,
            'mutation_prob': self.config.mutation_prob,
        })

        return result
