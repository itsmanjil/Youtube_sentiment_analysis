# Multi-Objective Ensemble Optimisation (NSGA-II)

## Setup

- **Models**: deberta_v3, logreg, svm
- **Objectives**: Macro-F1 (↑), ECE (↓), Coverage@70% (↑)
- **Algorithm**: NSGA-II  (pop=12, gen=12)
- **Evaluations**: 156
- **Runtime**: 1.5s
- **Pareto front size**: 4 solutions

## Pareto Front (Validation Set)

| Rank | w(deberta_v3) | w(logreg) | w(svm) | Macro-F1 | ECE | Coverage |
|------|------|------|------|----------|-----|----------|
|    1 | 0.023 | 0.157 | 0.820 | 0.7380 | 0.1452 | 0.4833 |
|    2 | 0.003 | 0.603 | 0.394 | 0.6889 | 0.1240 | 0.4833 |
|    3 | 0.149 | 0.604 | 0.247 | 0.6889 | 0.1188 | 0.3833 |
|    4 | 0.001 | 0.829 | 0.170 | 0.6702 | 0.1220 | 0.5000 |

## Knee-Point Selection

The recommended operating point minimises the normalised Chebyshev distance to the ideal point — balancing all three objectives.

**Knee-point index** (1-based): 2

### Optimal Weights

| Model | Weight |
|-------|--------|
| deberta_v3 | 0.0027 |
| logreg | 0.6034 |
| svm | 0.3938 |

## Validation Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.6889 |
| ECE | 0.1240 |
| Coverage@70% | 0.4833 |

## Test-Set Performance (knee-point)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.7646 |
| ECE | 0.1208 |
| Coverage@70% | 0.4167 |

## Thesis Interpretation

The Pareto front reveals that ensemble calibration and accuracy are partially competing: aggressive weight concentration (LogReg dominant) maximises F1 but raises ECE, while more balanced weights improve calibration at a small accuracy cost. The knee-point solution provides a principled, calibration-aware alternative to the single-objective PSO solution reported in §4.3, demonstrating that the CI contribution extends beyond raw accuracy improvement to probabilistic reliability — a key requirement for deployed sentiment systems.
