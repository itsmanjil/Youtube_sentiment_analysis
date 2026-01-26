#!/usr/bin/env python3
"""
Metaheuristics Demo for Thesis

Demonstrates PSO, MOPSO, and NSGA-II on optimization problems
relevant to sentiment analysis.

Usage:
    python -m research.computational_intelligence.metaheuristics.demo

Author: [Your Name]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

import numpy as np

from research.computational_intelligence.metaheuristics.base import (
    OptimizationProblem,
    ObjectiveType,
)
from research.computational_intelligence.metaheuristics.pso import (
    ParticleSwarmOptimizer,
    AdaptivePSO,
    PSOConfig,
)
from research.computational_intelligence.metaheuristics.mopso import (
    MultiObjectivePSO,
    MOPSOConfig,
)
from research.computational_intelligence.metaheuristics.nsga2 import (
    NSGA2,
    NSGA2Config,
)


# =============================================================================
# Test Problems
# =============================================================================

class SphereProblem(OptimizationProblem):
    """Simple sphere function for testing: f(x) = sum(x^2)"""

    def __init__(self, n_dims: int = 10):
        super().__init__(
            n_variables=n_dims,
            bounds=[(-5.12, 5.12)] * n_dims,
            objectives=['sphere'],
            objective_types=[ObjectiveType.MINIMIZE],
            name='Sphere'
        )

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class RastriginProblem(OptimizationProblem):
    """Rastrigin function - multimodal test problem."""

    def __init__(self, n_dims: int = 10):
        super().__init__(
            n_variables=n_dims,
            bounds=[(-5.12, 5.12)] * n_dims,
            objectives=['rastrigin'],
            objective_types=[ObjectiveType.MINIMIZE],
            name='Rastrigin'
        )
        self.A = 10

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return self.A * n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))


class ZDT1Problem(OptimizationProblem):
    """ZDT1 - standard multi-objective test problem."""

    def __init__(self, n_dims: int = 30):
        super().__init__(
            n_variables=n_dims,
            bounds=[(0, 1)] * n_dims,
            objectives=['f1', 'f2'],
            objective_types=[ObjectiveType.MINIMIZE, ObjectiveType.MINIMIZE],
            name='ZDT1'
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])


class EnsembleWeightProblemDemo(OptimizationProblem):
    """Simulated ensemble weight optimization problem."""

    def __init__(self, n_models: int = 3):
        super().__init__(
            n_variables=n_models,
            bounds=[(0.01, 1.0)] * n_models,
            objectives=['neg_accuracy'],
            objective_types=[ObjectiveType.MINIMIZE],
            name='EnsembleWeights'
        )

        # Simulated optimal weights (for demo)
        self.optimal = np.array([0.4, 0.4, 0.2])

    def evaluate(self, x: np.ndarray) -> float:
        # Normalize weights
        w = x / np.sum(x)

        # Simulated accuracy based on distance from optimal
        diff = np.linalg.norm(w - self.optimal)
        accuracy = 0.9 - 0.3 * diff  # Max 90% when weights are optimal

        return -accuracy  # Negative for minimization


# =============================================================================
# Demo Functions
# =============================================================================

def demo_pso():
    """Demo standard PSO on Sphere function."""
    print("\n" + "=" * 60)
    print("DEMO 1: Particle Swarm Optimization (PSO)")
    print("=" * 60)

    problem = SphereProblem(n_dims=10)
    print(f"\nProblem: {problem.name}")
    print(f"Dimensions: {problem.n_variables}")
    print(f"Optimal solution: x* = [0, 0, ..., 0], f(x*) = 0")

    pso = ParticleSwarmOptimizer(
        problem=problem,
        population_size=30,
        max_iterations=100,
        config=PSOConfig(
            c1=2.0, c2=2.0,
            w_start=0.9, w_end=0.4,
            adaptive_inertia=True
        ),
        seed=42,
        verbose=False
    )

    result = pso.optimize()

    print(f"\nResults:")
    print(f"  Best fitness: {result.best_fitness:.6f}")
    print(f"  Best position: [{', '.join(f'{x:.4f}' for x in result.best_solution.position[:5])}...]")
    print(f"  Iterations: {result.iterations}")
    print(f"  Evaluations: {result.evaluations}")
    print(f"  Runtime: {result.runtime:.3f}s")

    # Check convergence
    print(f"\n  Convergence (first 5 iterations): {result.convergence_history[:5]}")
    print(f"  Convergence (last 5 iterations): {result.convergence_history[-5:]}")


def demo_adaptive_pso():
    """Demo Adaptive PSO on Rastrigin (multimodal)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Adaptive PSO on Rastrigin Function")
    print("=" * 60)

    problem = RastriginProblem(n_dims=10)
    print(f"\nProblem: {problem.name} (multimodal)")
    print(f"Dimensions: {problem.n_variables}")
    print(f"Optimal solution: x* = [0, 0, ..., 0], f(x*) = 0")

    apso = AdaptivePSO(
        problem=problem,
        population_size=50,
        max_iterations=200,
        seed=42,
        verbose=False
    )

    result = apso.optimize()

    print(f"\nResults:")
    print(f"  Best fitness: {result.best_fitness:.6f}")
    print(f"  Best position: [{', '.join(f'{x:.4f}' for x in result.best_solution.position[:5])}...]")
    print(f"  Runtime: {result.runtime:.3f}s")


def demo_mopso():
    """Demo Multi-Objective PSO on ZDT1."""
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Objective PSO (MOPSO)")
    print("=" * 60)

    problem = ZDT1Problem(n_dims=10)
    print(f"\nProblem: {problem.name}")
    print(f"Objectives: {problem.objectives}")
    print(f"Expected Pareto front: f2 = 1 - sqrt(f1)")

    mopso = MultiObjectivePSO(
        problem=problem,
        population_size=50,
        max_iterations=100,
        config=MOPSOConfig(archive_size=50),
        seed=42,
        verbose=False
    )

    result = mopso.optimize()

    print(f"\nResults:")
    print(f"  Pareto front size: {len(result.pareto_front)}")
    print(f"  Runtime: {result.runtime:.3f}s")

    # Show some Pareto solutions
    print(f"\n  Sample Pareto solutions (f1, f2):")
    for i, sol in enumerate(result.pareto_front[:5]):
        print(f"    {i+1}. ({sol.fitness[0]:.4f}, {sol.fitness[1]:.4f})")


def demo_nsga2():
    """Demo NSGA-II on ZDT1."""
    print("\n" + "=" * 60)
    print("DEMO 4: NSGA-II Genetic Algorithm")
    print("=" * 60)

    problem = ZDT1Problem(n_dims=10)
    print(f"\nProblem: {problem.name}")
    print(f"Objectives: {problem.objectives}")

    nsga2 = NSGA2(
        problem=problem,
        population_size=100,
        max_iterations=100,
        config=NSGA2Config(
            crossover_prob=0.9,
            mutation_prob=0.1
        ),
        seed=42,
        verbose=False
    )

    result = nsga2.optimize()

    print(f"\nResults:")
    print(f"  Pareto front size: {len(result.pareto_front)}")
    print(f"  Runtime: {result.runtime:.3f}s")

    # Show Pareto solutions
    print(f"\n  Sample Pareto solutions (f1, f2):")
    for i, sol in enumerate(result.pareto_front[:5]):
        f = np.atleast_1d(sol.fitness)
        print(f"    {i+1}. ({f[0]:.4f}, {f[1]:.4f})")


def demo_ensemble_optimization():
    """Demo ensemble weight optimization."""
    print("\n" + "=" * 60)
    print("DEMO 5: Ensemble Weight Optimization")
    print("=" * 60)

    problem = EnsembleWeightProblemDemo(n_models=3)
    print(f"\nProblem: {problem.name}")
    print(f"Simulated optimal weights: {problem.optimal}")

    pso = ParticleSwarmOptimizer(
        problem=problem,
        population_size=20,
        max_iterations=50,
        seed=42,
        verbose=False
    )

    result = pso.optimize()

    # Normalize found weights
    weights = result.best_solution.position
    weights = weights / np.sum(weights)

    print(f"\nResults:")
    print(f"  Found weights: [{', '.join(f'{w:.4f}' for w in weights)}]")
    print(f"  Optimal weights: [{', '.join(f'{w:.4f}' for w in problem.optimal)}]")
    print(f"  Simulated accuracy: {-result.best_fitness:.4f}")
    print(f"  Runtime: {result.runtime:.3f}s")


def demo_comparison():
    """Compare PSO, MOPSO, NSGA-II."""
    print("\n" + "=" * 60)
    print("DEMO 6: Algorithm Comparison")
    print("=" * 60)

    problem = SphereProblem(n_dims=20)
    n_runs = 5

    print(f"\nProblem: Sphere (20D), {n_runs} runs each")
    print(f"Population: 30, Iterations: 100")

    results = {'PSO': [], 'Adaptive PSO': []}

    for run in range(n_runs):
        seed = 42 + run

        # Standard PSO
        pso = ParticleSwarmOptimizer(
            problem=problem,
            population_size=30,
            max_iterations=100,
            seed=seed
        )
        res = pso.optimize()
        results['PSO'].append(res.best_fitness)

        # Adaptive PSO
        apso = AdaptivePSO(
            problem=problem,
            population_size=30,
            max_iterations=100,
            seed=seed
        )
        res = apso.optimize()
        results['Adaptive PSO'].append(res.best_fitness)

    print(f"\nResults (mean ± std):")
    for alg, fits in results.items():
        print(f"  {alg:15s}: {np.mean(fits):.6f} ± {np.std(fits):.6f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("METAHEURISTICS DEMONSTRATION")
    print("Computational Intelligence for Sentiment Analysis")
    print("=" * 60)

    demo_pso()
    demo_adaptive_pso()
    demo_mopso()
    demo_nsga2()
    demo_ensemble_optimization()
    demo_comparison()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
    Summary of Implemented Algorithms:

    1. PSO (Particle Swarm Optimization)
       - Bio-inspired swarm algorithm
       - Fast convergence on unimodal problems
       - Configurable inertia, cognitive/social coefficients

    2. Adaptive PSO
       - Self-tuning parameters based on diversity
       - Better for multimodal problems
       - Automatic stagnation detection

    3. MOPSO (Multi-Objective PSO)
       - Handles multiple conflicting objectives
       - External archive for Pareto front
       - Crowding-based leader selection

    4. NSGA-II
       - Classic multi-objective evolutionary algorithm
       - Fast non-dominated sorting
       - SBX crossover, polynomial mutation

    Applications for Thesis:
    - Ensemble weight optimization
    - Fuzzy parameter tuning
    - Hyperparameter optimization
    - Multi-objective: accuracy vs. speed
    """)


if __name__ == '__main__':
    main()
