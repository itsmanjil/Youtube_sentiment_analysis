"""
Multi-Objective Particle Swarm Optimization (MOPSO)

Implementation of MOPSO for problems with multiple conflicting objectives.
Uses an external archive to maintain the Pareto front.

Key Features:
- External archive for non-dominated solutions
- Crowding distance for diversity preservation
- Leader selection based on crowding
- Grid-based archive management

Reference:
    Coello, C.A.C., Pulido, G.T., & Lechuga, M.S. (2004).
    "Handling Multiple Objectives with Particle Swarm Optimization".
    IEEE TEVC.

Author: [Your Name]
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from .base import (
    Optimizer,
    OptimizationProblem,
    OptimizationResult,
    Solution,
    ObjectiveType,
)


@dataclass
class MOPSOConfig:
    """Configuration for MOPSO algorithm."""
    # PSO parameters
    c1: float = 1.5
    c2: float = 1.5
    w_start: float = 0.9
    w_end: float = 0.4
    v_max_ratio: float = 0.2

    # Archive parameters
    archive_size: int = 100
    n_grids: int = 10  # Grid divisions per objective

    # Mutation
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1


class MultiObjectivePSO(Optimizer):
    """
    Multi-Objective Particle Swarm Optimization (MOPSO).

    MOPSO extends PSO to handle multiple conflicting objectives by:
    1. Maintaining an external archive of non-dominated solutions
    2. Using crowding distance for diversity preservation
    3. Selecting leaders from less crowded regions

    The algorithm finds a set of Pareto-optimal solutions representing
    trade-offs between objectives (e.g., accuracy vs. inference time).

    Parameters
    ----------
    problem : OptimizationProblem
        Multi-objective optimization problem
    population_size : int
        Number of particles
    max_iterations : int
        Maximum iterations
    config : MOPSOConfig
        Algorithm configuration
    seed : int
        Random seed
    verbose : bool
        Print progress

    Example
    -------
    >>> # Multi-objective problem: maximize accuracy, minimize inference time
    >>> problem = SentimentMOProblem(objectives=['accuracy', 'speed'])
    >>> mopso = MultiObjectivePSO(problem, population_size=50)
    >>> result = mopso.optimize()
    >>> print(f"Pareto front size: {len(result.pareto_front)}")
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        population_size: int = 50,
        max_iterations: int = 100,
        config: MOPSOConfig = None,
        seed: int = None,
        verbose: bool = False
    ):
        super().__init__(problem, population_size, max_iterations, seed, verbose)
        self.config = config or MOPSOConfig()

        # Velocity limits
        search_range = self.problem.upper_bounds - self.problem.lower_bounds
        self.v_max = self.config.v_max_ratio * search_range
        self.v_min = -self.v_max

        # External archive (Pareto front)
        self.archive: List[Solution] = []

        # Grid for archive management
        self.n_objectives = len(problem.objectives)
        self.grid_bounds: List[Tuple[float, float]] = []

    def initialize(self) -> None:
        """Initialize swarm and archive."""
        self.population = []

        for _ in range(self.population_size):
            position = self.problem.random_solution()
            velocity = np.random.uniform(self.v_min, self.v_max)

            particle = Solution(
                position=position,
                velocity=velocity,
                personal_best=position.copy(),
            )

            # Evaluate (multi-objective)
            particle.fitness = self.problem.evaluate_with_penalty(particle.position)
            particle.personal_best_fitness = particle.fitness.copy()

            self.population.append(particle)

        # Initialize archive with non-dominated solutions
        self._update_archive(self.population)

        # Set initial best (first archive member)
        if self.archive:
            self.best_solution = self.archive[0]

    def iterate(self) -> None:
        """Perform one MOPSO iteration."""
        w = self._get_inertia_weight()

        for particle in self.population:
            # Select leader from archive
            leader = self._select_leader()

            # Random coefficients
            r1 = np.random.random(self.problem.n_variables)
            r2 = np.random.random(self.problem.n_variables)

            # Velocity update
            cognitive = self.config.c1 * r1 * (particle.personal_best - particle.position)
            social = self.config.c2 * r2 * (leader.position - particle.position)

            particle.velocity = w * particle.velocity + cognitive + social
            particle.velocity = np.clip(particle.velocity, self.v_min, self.v_max)

            # Position update
            particle.position = particle.position + particle.velocity
            particle.position = self.problem.clip_to_bounds(particle.position)

            # Mutation (for diversity)
            if np.random.random() < self.config.mutation_rate:
                particle.position = self._mutate(particle.position)

            # Evaluate
            particle.fitness = self.problem.evaluate_with_penalty(particle.position)

            # Update personal best (non-dominated check)
            if self._dominates_or_equal(particle.fitness, particle.personal_best_fitness):
                particle.personal_best = particle.position.copy()
                particle.personal_best_fitness = particle.fitness.copy()

        # Update archive
        self._update_archive(self.population)

        # Update best solution (use first archive member)
        if self.archive:
            self.best_solution = self.archive[0]

    def _get_inertia_weight(self) -> float:
        """Linearly decreasing inertia weight."""
        progress = self.iteration / self.max_iterations
        return self.config.w_start - progress * (self.config.w_start - self.config.w_end)

    def _dominates_or_equal(self, fit1: np.ndarray, fit2: np.ndarray) -> bool:
        """Check if fit1 dominates or equals fit2."""
        return np.all(fit1 <= fit2)

    def _dominates(self, fit1: np.ndarray, fit2: np.ndarray) -> bool:
        """Check if fit1 strictly dominates fit2."""
        return np.all(fit1 <= fit2) and np.any(fit1 < fit2)

    def _update_archive(self, solutions: List[Solution]) -> None:
        """Update archive with new non-dominated solutions."""
        # Add new solutions to archive
        for sol in solutions:
            self._add_to_archive(sol)

        # Trim archive if too large
        if len(self.archive) > self.config.archive_size:
            self._trim_archive()

    def _add_to_archive(self, solution: Solution) -> None:
        """Add solution to archive if non-dominated."""
        # Check if dominated by any archive member
        dominated = False
        to_remove = []

        for i, arch_sol in enumerate(self.archive):
            if self._dominates(arch_sol.fitness, solution.fitness):
                dominated = True
                break
            if self._dominates(solution.fitness, arch_sol.fitness):
                to_remove.append(i)

        # Remove dominated archive members
        for i in reversed(to_remove):
            self.archive.pop(i)

        # Add if not dominated
        if not dominated:
            self.archive.append(Solution(
                position=solution.position.copy(),
                fitness=solution.fitness.copy()
            ))

    def _trim_archive(self) -> None:
        """Trim archive using crowding distance."""
        # Compute crowding distances
        crowding = self._compute_crowding_distance()

        # Sort by crowding (ascending) and remove most crowded
        indices = np.argsort(crowding)
        n_remove = len(self.archive) - self.config.archive_size

        # Remove solutions with smallest crowding distance
        for i in sorted(indices[:n_remove], reverse=True):
            self.archive.pop(i)

    def _compute_crowding_distance(self) -> np.ndarray:
        """Compute crowding distance for archive members."""
        n = len(self.archive)
        if n <= 2:
            return np.full(n, np.inf)

        crowding = np.zeros(n)
        fitness_matrix = np.array([sol.fitness for sol in self.archive])

        for obj in range(self.n_objectives):
            # Sort by this objective
            sorted_idx = np.argsort(fitness_matrix[:, obj])

            # Boundary solutions get infinite distance
            crowding[sorted_idx[0]] = np.inf
            crowding[sorted_idx[-1]] = np.inf

            # Compute distance for middle solutions
            obj_range = fitness_matrix[sorted_idx[-1], obj] - fitness_matrix[sorted_idx[0], obj]
            if obj_range > 0:
                for i in range(1, n - 1):
                    crowding[sorted_idx[i]] += (
                        fitness_matrix[sorted_idx[i + 1], obj] -
                        fitness_matrix[sorted_idx[i - 1], obj]
                    ) / obj_range

        return crowding

    def _select_leader(self) -> Solution:
        """Select leader from archive using crowding-based tournament."""
        if len(self.archive) == 0:
            return self.population[0]

        if len(self.archive) == 1:
            return self.archive[0]

        # Binary tournament based on crowding distance
        crowding = self._compute_crowding_distance()

        i1 = np.random.randint(len(self.archive))
        i2 = np.random.randint(len(self.archive))

        # Select less crowded (higher crowding distance)
        if crowding[i1] > crowding[i2]:
            return self.archive[i1]
        return self.archive[i2]

    def _mutate(self, position: np.ndarray) -> np.ndarray:
        """Apply polynomial mutation."""
        mutated = position.copy()
        for i in range(len(position)):
            if np.random.random() < 0.5:
                delta = self.config.mutation_strength * (
                    self.problem.upper_bounds[i] - self.problem.lower_bounds[i]
                )
                mutated[i] += np.random.uniform(-delta, delta)

        return self.problem.clip_to_bounds(mutated)

    def optimize(self) -> OptimizationResult:
        """Run MOPSO and return result with Pareto front."""
        result = super().optimize()

        # Add Pareto front
        result.pareto_front = self.archive.copy()

        # Add MOPSO-specific metadata
        result.metadata.update({
            'archive_size': len(self.archive),
            'hypervolume': self._compute_hypervolume(),
        })

        return result

    def _compute_hypervolume(self, reference_point: np.ndarray = None) -> float:
        """
        Compute hypervolume indicator (2D approximation).

        For thesis: Hypervolume measures the quality of the Pareto front.
        Larger hypervolume = better coverage of objective space.
        """
        if len(self.archive) == 0:
            return 0.0

        if self.n_objectives != 2:
            # For >2 objectives, use approximation or external library
            return 0.0

        # Get fitness values
        fitness = np.array([sol.fitness for sol in self.archive])

        # Default reference point (worst in each objective + margin)
        if reference_point is None:
            reference_point = np.max(fitness, axis=0) * 1.1

        # Sort by first objective
        sorted_idx = np.argsort(fitness[:, 0])
        sorted_fitness = fitness[sorted_idx]

        # Compute hypervolume (2D)
        hv = 0.0
        prev_x = sorted_fitness[0, 0]

        for i in range(len(sorted_fitness)):
            x, y = sorted_fitness[i]
            width = reference_point[0] - x
            height = reference_point[1] - y

            if i < len(sorted_fitness) - 1:
                next_y = sorted_fitness[i + 1, 1]
                height = next_y - y

            hv += width * max(0, height)

        return hv

    def get_pareto_front(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get Pareto front as list of (position, fitness) tuples."""
        return [(sol.position.copy(), sol.fitness.copy()) for sol in self.archive]
