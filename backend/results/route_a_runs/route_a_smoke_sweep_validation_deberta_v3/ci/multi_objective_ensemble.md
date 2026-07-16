# Multi-Objective Ensemble Optimisation (NSGA-II)

## Setup

- **Models**: deberta_v3, logreg, svm
- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@70% (↑)
- **Algorithm**: NSGA-II  (pop=8, gen=8)
- **Evaluations**: 72
- **Runtime**: 0.2s
- **Pareto front size**: 4 solutions

## Pareto Front (Validation Set)

| Rank | w(deberta_v3) | w(logreg) | w(svm) | Macro-F1 | ECE | Coverage |
|------|------|------|------|----------|-----|----------|
|    1 | 0.691 | 0.176 | 0.132 | 0.6096 | 0.0706 | 0.1667 |
|    2 | 0.042 | 0.799 | 0.159 | 0.6889 | 0.1564 | 0.4667 |
|    3 | 0.663 | 0.140 | 0.197 | 0.6069 | 0.0668 | 0.1667 |
|    4 | 0.663 | 0.140 | 0.197 | 0.6069 | 0.0668 | 0.1667 |

## Knee-Point Selection

The recommended operating point minimises the normalised Chebyshev distance to the ideal point — balancing all three objectives.

**Knee-point index** (1-based): 1

### Optimal Weights

| Model | Weight |
|-------|--------|
| deberta_v3 | 0.6915 |
| logreg | 0.1764 |
| svm | 0.1321 |

## Validation Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.6096 |
| ECE | 0.0706 |
| Coverage@70% | 0.1667 |

## Test-Set Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.6971 |
| ECE | 0.1182 |
| Coverage@70% | 0.1667 |

## Thesis Interpretation

The Pareto front reveals that ensemble calibration and accuracy are partially competing: aggressive weight concentration (LogReg dominant) maximises F1 but raises ECE, while more balanced weights improve calibration at a small accuracy cost. The knee-point solution provides a principled, calibration-aware alternative to the single-objective PSO solution reported in §4.3, demonstrating that the CI contribution extends beyond raw accuracy improvement to probabilistic reliability — a key requirement for deployed sentiment systems.
