"""
Computational Intelligence Module for YouTube Sentiment Analysis

This module provides bio-inspired and soft computing techniques for enhanced
sentiment analysis:

- Fuzzy Logic: uncertainty-aware ensemble fusion (see .fuzzy)
- Metaheuristics: PSO and NSGA-II ensemble weight optimization (see
  .metaheuristics — the optimizer wrapper classes there are exploratory
  scaffolding, not used in reported results; see that submodule's docstring)

An earlier revision of this docstring additionally claimed "Neural
Architecture Search: Evolutionary architecture optimization" and
"Bio-Inspired Attention: Novel attention mechanisms". Neither exists in this
codebase: research/architectures/attention.py implements standard
multi-head scaled dot-product attention (Vaswani et al. 2017), not a novel
or bio-inspired mechanism, and there is no architecture-search code anywhere
in this repository. Do not cite either claim.

Author: Thesis Project — Computational Intelligence & Sentiment Analysis
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from .fuzzy import FuzzySentimentClassifier, FuzzyEvaluator
from .metaheuristics import (
    ParticleSwarmOptimizer,
    NSGA2,
    EnsembleWeightOptimizer,
    HyperparameterTuner,
)

__all__ = [
    # Fuzzy Logic
    'FuzzySentimentClassifier',
    'FuzzyEvaluator',
    # Metaheuristics
    'ParticleSwarmOptimizer',
    'NSGA2',
    'EnsembleWeightOptimizer',
    'HyperparameterTuner',
]
