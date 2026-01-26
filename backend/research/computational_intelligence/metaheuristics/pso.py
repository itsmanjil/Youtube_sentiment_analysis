"""
Particle Swarm Optimization (PSO)

Implementation of the classic PSO algorithm with various enhancements:
- Adaptive inertia weight
- Constriction coefficient
- Velocity clamping
- Local best topology

Reference:
    Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization".
    Proceedings of ICNN'95.

    Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer".
    IEEE ICEC.

Author: [Your Name]
"""

from typing import List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass

from .base import (
    Optimizer,
    OptimizationProblem,
    OptimizationResult,
    Solution,
    ObjectiveType,
)


@dataclass
class PSOConfig:
    """Configuration for PSO algorithm."""
    # Cognitive and social coefficients
    c1: float = 2.0  # Cognitive (personal best attraction)
    c2: float = 2.0  # Social (global best attraction)

    # Inertia weight
    w_start: float = 0.9  # Initial inertia
    w_end: float = 0.4    # Final inertia
    adaptive_inertia: bool = True

    # Velocity limits
    v_max_ratio: float = 0.2  # Max velocity as ratio of search space

    # Constriction
    use_constriction: bool = False
    chi: float = 0.729  # Constriction coefficient

    # Topology
    topology: str = 'global'  # 'global' or 'ring'
    neighborhood_size: int = 3  # For ring topology


class ParticleSwarmOptimizer(Optimizer):
    """
    Particle Swarm Optimization (PSO) Algorithm.

    PSO is a population-based stochastic optimization technique inspired
    by the social behavior of bird flocking or fish schooling.

    Each particle:
    - Has a position (candidate solution) and velocity
    - Remembers its personal best position
    - Is attracted toward its personal best and the swarm's global best

    Velocity Update:
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)

    Position Update:
        x_i(t+1) = x_i(t) + v_i(t+1)

    Parameters
    ----------
    problem : OptimizationProblem
        The optimization problem to solve
    population_size : int
        Number of particles in the swarm
    max_iterations : int
        Maximum number of iterations
    config : PSOConfig, optional
        Algorithm configuration
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Whether to print progress

    Example
    -------
    >>> problem = SomeOptimizationProblem()
    >>> pso = ParticleSwarmOptimizer(
    ...     problem=problem,
    ...     population_size=30,
    ...     max_iterations=100,
    ...     verbose=True
    ... )
    >>> result = pso.optimize()
    >>> print(f"Best solution: {result.best_solution.position}")
    >>> print(f"Best fitness: {result.best_fitness}")
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        population_size: int = 30,
        max_iterations: int = 100,
        config: PSOConfig = None,
        seed: int = None,
        verbose: bool = False
    ):
        super().__init__(problem, population_size, max_iterations, seed, verbose)
        self.config = config or PSOConfig()

        # Compute velocity limits
        search_range = self.problem.upper_bounds - self.problem.lower_bounds
        self.v_max = self.config.v_max_ratio * search_range
        self.v_min = -self.v_max

        # Global best
        self.global_best: Solution = None

        # For ring topology
        self.local_bests: List[Solution] = []

    def initialize(self) -> None:
        """Initialize the swarm with random particles."""
        self.population = []

        for _ in range(self.population_size):
            # Random position
            position = self.problem.random_solution()

            # Random velocity
            velocity = np.random.uniform(self.v_min, self.v_max)

            particle = Solution(
                position=position,
                velocity=velocity,
                personal_best=position.copy(),
            )

            # Evaluate
            self._evaluate_solution(particle)
            particle.personal_best_fitness = particle.fitness

            self.population.append(particle)

            # Update global best
            self._update_global_best(particle)

        # Initialize local bests for ring topology
        if self.config.topology == 'ring':
            self._update_local_bests()

    def iterate(self) -> None:
        """Perform one iteration of PSO."""
        # Compute current inertia weight
        w = self._get_inertia_weight()

        for i, particle in enumerate(self.population):
            # Get the best to follow (global or local)
            if self.config.topology == 'ring':
                best_to_follow = self.local_bests[i]
            else:
                best_to_follow = self.global_best

            # Random coefficients
            r1 = np.random.random(self.problem.n_variables)
            r2 = np.random.random(self.problem.n_variables)

            # Velocity update
            cognitive = self.config.c1 * r1 * (particle.personal_best - particle.position)
            social = self.config.c2 * r2 * (best_to_follow.position - particle.position)

            if self.config.use_constriction:
                particle.velocity = self.config.chi * (
                    particle.velocity + cognitive + social
                )
            else:
                particle.velocity = w * particle.velocity + cognitive + social

            # Velocity clamping
            particle.velocity = np.clip(particle.velocity, self.v_min, self.v_max)

            # Position update
            particle.position = particle.position + particle.velocity

            # Boundary handling (reflection)
            particle.position = self._handle_boundaries(particle.position)

            # Evaluate new position
            self._evaluate_solution(particle)

            # Update personal best
            if self._is_better(particle.fitness, particle.personal_best_fitness):
                particle.personal_best = particle.position.copy()
                particle.personal_best_fitness = particle.fitness

                # Update global best
                self._update_global_best(particle)

        # Update local bests for ring topology
        if self.config.topology == 'ring':
            self._update_local_bests()

        # Update algorithm's best solution
        self.best_solution = self.global_best

    def _get_inertia_weight(self) -> float:
        """Compute inertia weight (linearly decreasing or fixed)."""
        if self.config.adaptive_inertia:
            progress = self.iteration / self.max_iterations
            return self.config.w_start - progress * (self.config.w_start - self.config.w_end)
        return self.config.w_start

    def _handle_boundaries(self, position: np.ndarray) -> np.ndarray:
        """Handle boundary violations using reflection."""
        lower = self.problem.lower_bounds
        upper = self.problem.upper_bounds

        # Reflection for lower bound violations
        below = position < lower
        position[below] = 2 * lower[below] - position[below]

        # Reflection for upper bound violations
        above = position > upper
        position[above] = 2 * upper[above] - position[above]

        # Final clip to ensure within bounds
        return np.clip(position, lower, upper)

    def _update_global_best(self, particle: Solution) -> None:
        """Update global best if particle is better."""
        if self.global_best is None:
            self.global_best = Solution(
                position=particle.position.copy(),
                fitness=particle.fitness
            )
        elif self._is_better(particle.fitness, self.global_best.fitness):
            self.global_best = Solution(
                position=particle.position.copy(),
                fitness=particle.fitness
            )

    def _update_local_bests(self) -> None:
        """Update local bests for ring topology."""
        self.local_bests = []
        k = self.config.neighborhood_size // 2

        for i in range(self.population_size):
            # Get neighbors (ring topology)
            neighbors = []
            for j in range(-k, k + 1):
                idx = (i + j) % self.population_size
                neighbors.append(self.population[idx])

            # Find best in neighborhood
            best = min(neighbors, key=lambda p: p.personal_best_fitness
                      if self.problem.objective_types[0] == ObjectiveType.MINIMIZE
                      else -p.personal_best_fitness)

            self.local_bests.append(Solution(
                position=best.personal_best.copy(),
                fitness=best.personal_best_fitness
            ))

    def optimize(self) -> OptimizationResult:
        """Run PSO optimization with enhanced tracking."""
        result = super().optimize()

        # Add PSO-specific metadata
        result.metadata.update({
            'c1': self.config.c1,
            'c2': self.config.c2,
            'w_start': self.config.w_start,
            'w_end': self.config.w_end,
            'topology': self.config.topology,
        })

        return result


class AdaptivePSO(ParticleSwarmOptimizer):
    """
    Adaptive PSO with self-tuning parameters.

    This variant automatically adjusts c1, c2, and w based on
    swarm diversity and convergence behavior.

    Reference:
        Zhan et al. (2009). "Adaptive Particle Swarm Optimization".
        IEEE TSMC-B.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Adaptation parameters
        self.diversity_history: List[float] = []
        self.stagnation_count = 0
        self.last_best_fitness = None

    def iterate(self) -> None:
        """PSO iteration with adaptive parameters."""
        # Compute swarm diversity
        diversity = self._compute_diversity()
        self.diversity_history.append(diversity)

        # Adapt parameters based on diversity
        self._adapt_parameters(diversity)

        # Check for stagnation
        if self.global_best is not None:
            if self.last_best_fitness is not None:
                if abs(self.global_best.fitness - self.last_best_fitness) < 1e-10:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0
            self.last_best_fitness = self.global_best.fitness

        # Reinitialization if stagnant
        if self.stagnation_count > 20:
            self._reinitialize_worst()
            self.stagnation_count = 0

        # Standard PSO iteration
        super().iterate()

    def _compute_diversity(self) -> float:
        """Compute swarm diversity (average distance from centroid)."""
        positions = np.array([p.position for p in self.population])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)

    def _adapt_parameters(self, diversity: float) -> None:
        """Adapt c1, c2 based on diversity."""
        # Normalize diversity
        search_range = np.linalg.norm(self.problem.upper_bounds - self.problem.lower_bounds)
        norm_diversity = diversity / search_range

        if norm_diversity < 0.1:  # Low diversity - need exploration
            self.config.c1 = 1.5  # Less personal influence
            self.config.c2 = 2.5  # More social influence
            self.config.w_start = 0.9  # Higher inertia
        elif norm_diversity > 0.3:  # High diversity - need exploitation
            self.config.c1 = 2.5  # More personal influence
            self.config.c2 = 1.5  # Less social influence
            self.config.w_start = 0.5  # Lower inertia
        else:  # Balanced
            self.config.c1 = 2.0
            self.config.c2 = 2.0
            self.config.w_start = 0.7

    def _reinitialize_worst(self) -> None:
        """Reinitialize worst particles to escape local optima."""
        # Sort by fitness
        sorted_pop = sorted(
            self.population,
            key=lambda p: p.fitness,
            reverse=(self.problem.objective_types[0] == ObjectiveType.MAXIMIZE)
        )

        # Reinitialize worst 20%
        n_reinit = max(1, self.population_size // 5)
        for i in range(n_reinit):
            particle = sorted_pop[-(i+1)]
            particle.position = self.problem.random_solution()
            particle.velocity = np.random.uniform(self.v_min, self.v_max)
            self._evaluate_solution(particle)
