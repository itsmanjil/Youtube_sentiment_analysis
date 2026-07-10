"""
Metaheuristics Module for Computational Intelligence

Algorithms Implemented:
1. PSO - Particle Swarm Optimization
2. MOPSO - Multi-Objective PSO
3. NSGA-II - Non-dominated Sorting Genetic Algorithm II

There is no Differential Evolution (DE) implementation in this module or
anywhere in this repository; an earlier revision of this docstring claimed
one existed. Do not cite DE as an implemented method.

Only PSO and NSGA-II are load-bearing for reported thesis results (see
research/analysis/pso_convergence_analysis.py and
research/ci/multi_objective_ensemble.py, which implement the weighted-voting
and NSGA-II optimization used in the pinned runtime artifacts directly,
without going through this module's OptimizationProblem/Optimizer classes).
MOPSO and the EnsembleWeightOptimizer/FuzzyParameterOptimizer/
HyperparameterTuner wrapper classes below are exploratory scaffolding: they
are exercised only by demo.py's synthetic-data demo, are never called by any
script that produces a results/ or results/runtime/ artifact, and must not be
read as claimed contributions.

Reference:
- Kennedy & Eberhart (1995): Particle Swarm Optimization
- Coello et al. (2004): Multi-Objective PSO
- Deb et al. (2002): NSGA-II

Author: Thesis Project — Computational Intelligence & Sentiment Analysis
"""

from .base import (
    OptimizationProblem,
    Solution,
    Optimizer,
)
from .pso import ParticleSwarmOptimizer, AdaptivePSO
from .mopso import MultiObjectivePSO
from .nsga2 import NSGA2
from .sentiment_optimization import (
    EnsembleWeightOptimizer,
    FuzzyParameterOptimizer,
    HyperparameterTuner,
)

__all__ = [
    # Base classes
    'OptimizationProblem',
    'Solution',
    'Optimizer',
    # Algorithms
    'ParticleSwarmOptimizer',
    'AdaptivePSO',
    'MultiObjectivePSO',
    'NSGA2',
    # Sentiment-specific
    'EnsembleWeightOptimizer',
    'FuzzyParameterOptimizer',
    'HyperparameterTuner',
]
