"""
Fuzzy Logic Module for Sentiment Analysis

This module implements fuzzy set theory and fuzzy inference systems
for handling uncertainty in sentiment classification. Key features:

1. Membership Functions: Triangular, Trapezoidal, Gaussian, Sigmoid
2. Fuzzy Inference System: Mamdani-type with customizable rules
3. Defuzzification: Centroid, Bisector, MOM, SOM, LOM methods
4. Integration Layer: Seamless integration with existing ML models

Theoretical Foundation:
- Zadeh, L.A. (1965). Fuzzy Sets. Information and Control.
- Mamdani, E.H. (1974). Application of Fuzzy Logic to Approximate Reasoning.

Usage:
    from computational_intelligence.fuzzy import FuzzySentimentClassifier

    classifier = FuzzySentimentClassifier(
        defuzz_method='centroid',
        alpha_cut=0.3
    )
    result = classifier.classify(sentiment_scores)
"""

from .membership_functions import (
    MembershipFunction,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SigmoidMF,
    BellMF,
    create_three_class_mfs,
    create_sentiment_mfs_triangular,
    create_sentiment_mfs_gaussian,
)
from .fuzzy_inference import (
    FuzzyVariable,
    FuzzyRule,
    FuzzyInferenceSystem,
)
from .fuzzy_sentiment import FuzzySentimentClassifier
from .defuzzification import Defuzzifier
from .fuzzy_evaluation import FuzzyEvaluator

__all__ = [
    # Membership Functions
    'MembershipFunction',
    'TriangularMF',
    'TrapezoidalMF',
    'GaussianMF',
    'SigmoidMF',
    'BellMF',
    'create_three_class_mfs',
    'create_sentiment_mfs_triangular',
    'create_sentiment_mfs_gaussian',
    # Fuzzy Inference
    'FuzzyVariable',
    'FuzzyRule',
    'FuzzyInferenceSystem',
    # Main Classifier
    'FuzzySentimentClassifier',
    # Defuzzification
    'Defuzzifier',
    # Evaluation
    'FuzzyEvaluator',
]
