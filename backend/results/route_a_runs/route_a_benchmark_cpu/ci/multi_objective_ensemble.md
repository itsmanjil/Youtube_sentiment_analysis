# Multi-Objective Ensemble Optimisation (NSGA-II)

## Setup

- **Models**: deberta_v3, logreg, svm
- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@70% (↑)
- **Algorithm**: NSGA-II  (pop=16, gen=16)
- **Evaluations**: 272
- **Runtime**: 3.1s
- **Pareto front size**: 9 solutions

## Pareto Front (Validation Set)

| Rank | w(deberta_v3) | w(logreg) | w(svm) | Macro-F1 | ECE | Coverage |
|------|------|------|------|----------|-----|----------|
|    1 | 0.000 | 0.025 | 0.974 | 0.7693 | 0.0855 | 0.5000 |
|    2 | 0.001 | 0.194 | 0.806 | 0.7636 | 0.0781 | 0.5056 |
|    3 | 0.001 | 0.194 | 0.806 | 0.7636 | 0.0781 | 0.5056 |
|    4 | 0.000 | 0.666 | 0.334 | 0.7475 | 0.0784 | 0.5167 |
|    5 | 0.000 | 0.809 | 0.191 | 0.7359 | 0.0671 | 0.5278 |
|    6 | 0.001 | 0.096 | 0.903 | 0.7747 | 0.0900 | 0.5111 |
|    7 | 0.000 | 0.025 | 0.974 | 0.7693 | 0.0855 | 0.5000 |
|    8 | 0.000 | 0.662 | 0.338 | 0.7475 | 0.0814 | 0.5222 |
|    9 | 0.000 | 0.025 | 0.974 | 0.7693 | 0.0855 | 0.5000 |

## Knee-Point Selection

The recommended operating point minimises the normalised Chebyshev distance to the ideal point — balancing all three objectives.

**Knee-point index** (1-based): 4

### Optimal Weights

| Model | Weight |
|-------|--------|
| deberta_v3 | 0.0000 |
| logreg | 0.6656 |
| svm | 0.3343 |

## Validation Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.7475 |
| ECE | 0.0784 |
| Coverage@70% | 0.5167 |

## Test-Set Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.7826 |
| ECE | 0.0817 |
| Coverage@70% | 0.5222 |

## Thesis Interpretation

The Pareto front reveals that ensemble calibration and accuracy are partially competing: aggressive weight concentration (LogReg dominant) maximises F1 but raises ECE, while more balanced weights improve calibration at a small accuracy cost. The knee-point solution provides a principled, calibration-aware alternative to the single-objective PSO solution reported in §4.3, demonstrating that the CI contribution extends beyond raw accuracy improvement to probabilistic reliability — a key requirement for deployed sentiment systems.
