"""
Probability-cube I/O utilities.

The encoder training/calibration modules that formerly lived here were removed
with the DeBERTa-v3 arm. The cube format itself is model-agnostic and is still
used by the CI analyses (research/ci/multi_objective_ensemble.py,
neuro_fuzzy_gate.py, prediction_level_reconciliation.py) to hold stacked
per-model probabilities for the classical engines.
"""

from .prob_cube_io import (
    ProbabilityCubeBundle,
    load_probability_cube,
    save_probability_cube,
)

__all__ = [
    "ProbabilityCubeBundle",
    "load_probability_cube",
    "save_probability_cube",
]
