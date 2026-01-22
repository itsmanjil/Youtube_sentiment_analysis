"""
Evaluation Framework for Thesis Experiments.

This package provides rigorous evaluation tools required for academic research:

- framework.py: ThesisEvaluationFramework with k-fold CV
- statistical_tests.py: McNemar's test, Wilcoxon, Friedman
- cross_domain.py: Cross-domain evaluation (YouTube -> Twitter)
- ablation.py: Ablation study framework

These tools ensure your thesis results meet academic standards for
statistical rigor and reproducibility.
"""

from .statistical_tests import StatisticalSignificanceTester
from .ablation import AblationStudyFramework

__all__ = [
    "StatisticalSignificanceTester",
    "AblationStudyFramework",
]
