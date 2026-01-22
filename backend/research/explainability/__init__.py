"""
Explainability (XAI) Module for Sentiment Analysis.

This package provides model interpretability tools required for academic
research. Explainability is crucial for:

1. Understanding model decisions
2. Building trust in predictions
3. Identifying model biases
4. Creating thesis visualizations

Available Explainers
--------------------
- SHAPExplainer: SHAP values for feature importance
- LIMEExplainer: Local interpretable explanations
- AttentionExplainer: Attention weight visualization

References
----------
Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions
    of Any Classifier.
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .attention_explainer import AttentionExplainer

__all__ = [
    "SHAPExplainer",
    "LIMEExplainer",
    "AttentionExplainer",
]
