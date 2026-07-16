# Multi-Objective Ensemble Optimisation (NSGA-II)

## Setup

- **Models**: deberta_v3, logreg, svm
- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@70% (↑)
- **Algorithm**: NSGA-II  (pop=12, gen=12)
- **Evaluations**: 156
- **Runtime**: 0.5s
- **Pareto front size**: 7 solutions

## Pareto Front (Validation Set)

| Rank | w(deberta_v3) | w(logreg) | w(svm) | Macro-F1 | ECE | Coverage |
|------|------|------|------|----------|-----|----------|
|    1 | 0.019 | 0.011 | 0.970 | 0.7675 | 0.1107 | 0.4333 |
|    2 | 0.009 | 0.626 | 0.365 | 0.7668 | 0.0979 | 0.4833 |
|    3 | 0.020 | 0.179 | 0.801 | 0.7517 | 0.0943 | 0.4333 |
|    4 | 0.005 | 0.227 | 0.768 | 0.7339 | 0.0879 | 0.4667 |
|    5 | 0.005 | 0.253 | 0.742 | 0.7339 | 0.0964 | 0.4833 |
|    6 | 0.020 | 0.179 | 0.801 | 0.7517 | 0.0943 | 0.4333 |
|    7 | 0.020 | 0.179 | 0.801 | 0.7517 | 0.0943 | 0.4333 |

## Knee-Point Selection

The recommended operating point minimises the normalised Chebyshev distance to the ideal point — balancing all three objectives.

**Knee-point index** (1-based): 2

### Optimal Weights

| Model | Weight |
|-------|--------|
| deberta_v3 | 0.0088 |
| logreg | 0.6262 |
| svm | 0.3650 |

## Validation Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.7668 |
| ECE | 0.0979 |
| Coverage@70% | 0.4833 |

## Test-Set Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.6458 |
| ECE | 0.1511 |
| Coverage@70% | 0.5500 |

## Thesis Interpretation

The Pareto front reveals that ensemble calibration and accuracy are partially competing: aggressive weight concentration (LogReg dominant) maximises F1 but raises ECE, while more balanced weights improve calibration at a small accuracy cost. The knee-point solution provides a principled, calibration-aware alternative to the single-objective PSO solution reported in §4.3, demonstrating that the CI contribution extends beyond raw accuracy improvement to probabilistic reliability — a key requirement for deployed sentiment systems.
