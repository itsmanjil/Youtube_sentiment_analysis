"""
Computational Intelligence Module for YouTube Sentiment Analysis

This module provides bio-inspired and soft computing techniques for enhanced
sentiment analysis, including:

- Fuzzy Logic: Handles uncertainty in sentiment classification
- Metaheuristics: Optimization algorithms (PSO, MOPSO, NSGA-II)
- Neural Architecture Search: Evolutionary architecture optimization
- Bio-Inspired Attention: Novel attention mechanisms

Author: [Your Name]
Thesis: Computational Intelligence Approaches for YouTube Sentiment Analysis
"""

from .fuzzy import FuzzySentimentClassifier, FuzzyEvaluator
from .metaheuristics import (
    ParticleSwarmOptimizer,
    MultiObjectivePSO,
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
    'MultiObjectivePSO',
    'NSGA2',
    'EnsembleWeightOptimizer',
    'HyperparameterTuner',
]
