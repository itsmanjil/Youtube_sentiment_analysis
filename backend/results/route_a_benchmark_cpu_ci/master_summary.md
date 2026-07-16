# Route A CI Results — Master Summary

**Models in ensemble**: deberta_v3, logreg, svm

## Model-Level Metrics (Test Set)

| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | T |
|-------|----------|-----------|------------------|---|
| deberta_v3 | 0.6579 | 0.1084 | 0.1084 | 1.000 |
| logreg | 0.7544 | 0.0789 | 0.0621 | 0.861 |
| svm | 0.7863 | 0.1004 | 0.0649 | 0.735 |

## Ensemble-Level Metrics (Test Set)

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Uniform ensemble | 0.7770 | 0.7778 | 0.1385 | 0.3375 |
| Neuro-fuzzy gate | **0.7976** | 0.8000 | **0.0853** | 0.3048 |
| Entropy-gated (neuro_fuzzy) | 0.7976 | 0.8000 | — | — |

## Selective Prediction (AURC / AUCA)

Entropy-gated AURC = **0.0861**

| Model | AUCA | AUC-F1 | Acc@100% |
|-------|------|--------|----------|
| svm | 0.9122 | 0.9095 | 0.7889 |
| neuro_fuzzy | 0.9070 | 0.9015 | 0.8000 |
| ensemble_uniform | 0.9048 | 0.9002 | 0.7778 |
| logreg | 0.9003 | 0.8951 | 0.7556 |
| deberta_v3 | 0.7125 | 0.5642 | 0.6556 |

## NF Gate — Learned MF Parameters

| Model | Fuzzy Set | Center | Width | Alpha |
|-------|-----------|--------|-------|-------|
| deberta_v3 | Low | 0.108 | 0.101 | -5.000 |
| deberta_v3 | Medium | 0.314 | 0.050 | -4.678 |
| deberta_v3 | High | 0.424 | 0.050 | +5.000 |
| logreg | Low | 0.655 | 0.072 | +5.000 |
| logreg | Medium | 0.716 | 0.238 | +3.825 |
| logreg | High | 0.444 | 0.050 | +5.000 |
| svm | Low | 0.335 | 0.064 | +5.000 |
| svm | Medium | 0.115 | 0.235 | +3.735 |
| svm | High | 0.130 | 0.162 | -0.521 |