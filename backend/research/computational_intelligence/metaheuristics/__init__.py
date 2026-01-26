"""
Metaheuristics Module for Computational Intelligence

This module provides bio-inspired optimization algorithms for:
- Hyperparameter tuning
- Ensemble weight optimization
- Neural architecture search
- Feature selection

Algorithms Implemented:
1. PSO - Particle Swarm Optimization
2. MOPSO - Multi-Objective PSO
3. NSGA-II - Non-dominated Sorting Genetic Algorithm II
4. DE - Differential Evolution

Reference:
- Kennedy & Eberhart (1995): Particle Swarm Optimization
- Coello et al. (2004): Multi-Objective PSO
- Deb et al. (2002): NSGA-II

Author: [Your Name]
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
