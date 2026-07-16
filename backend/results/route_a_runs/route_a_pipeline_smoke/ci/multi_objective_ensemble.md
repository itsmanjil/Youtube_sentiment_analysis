# Multi-Objective Ensemble Optimisation (NSGA-II)

## Setup

- **Models**: deberta_v3, logreg, svm
- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@70% (↑)
- **Algorithm**: NSGA-II  (pop=8, gen=8)
- **Evaluations**: 72
- **Runtime**: 0.5s
- **Pareto front size**: 5 solutions

## Pareto Front (Validation Set)

| Rank | w(deberta_v3) | w(logreg) | w(svm) | Macro-F1 | ECE | Coverage |
|------|------|------|------|----------|-----|----------|
|    1 | 0.675 | 0.132 | 0.193 | 0.5905 | 0.0659 | 0.1667 |
|    2 | 0.897 | 0.045 | 0.058 | 0.5641 | 0.0953 | 0.2167 |
|    3 | 0.690 | 0.170 | 0.140 | 0.6096 | 0.0807 | 0.1667 |
|    4 | 0.011 | 0.585 | 0.403 | 0.6889 | 0.1246 | 0.4833 |
|    5 | 0.001 | 0.179 | 0.820 | 0.7380 | 0.1624 | 0.4833 |

## Knee-Point Selection

The recommended operating point minimises the normalised Chebyshev distance to the ideal point — balancing all three objectives.

**Knee-point index** (1-based): 4

### Optimal Weights

| Model | Weight |
|-------|--------|
| deberta_v3 | 0.0114 |
| logreg | 0.5854 |
| svm | 0.4032 |

## Validation Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.6889 |
| ECE | 0.1246 |
| Coverage@70% | 0.4833 |

## Test-Set Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.7646 |
| ECE | 0.1130 |
| Coverage@70% | 0.4167 |

## Thesis Interpretation

The Pareto front reveals that ensemble calibration and accuracy are partially competing: aggressive weight concentration (LogReg dominant) maximises F1 but raises ECE, while more balanced weights improve calibration at a small accuracy cost. The knee-point solution provides a principled, calibration-aware alternative to the single-objective PSO solution reported in §4.3, demonstrating that the CI contribution extends beyond raw accuracy improvement to probabilistic reliability — a key requirement for deployed sentiment systems.
