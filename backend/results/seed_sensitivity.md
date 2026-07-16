# Seed Sensitivity Analysis

Pipeline: TF-IDF(1,2, 50k) + LogReg(balanced)  |  8,000 per class, 80/20 train/val  |  seeds: [0, 7, 13, 42, 123]

> Each seed re-samples the training data, re-splits, and re-seeds the model. This captures the full pipeline variance, not just model initialisation.

| Seed | n_train | n_val | |V| | Accuracy | F1-macro | Cohen κ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19,200 | 4,800 | 48,112 | 0.6438 | 0.6439 | 0.4656 |
| 7 | 19,200 | 4,800 | 47,694 | 0.6523 | 0.6530 | 0.4784 |
| 13 | 19,200 | 4,800 | 47,887 | 0.6490 | 0.6499 | 0.4734 |
| 42 | 19,200 | 4,800 | 48,045 | 0.6506 | 0.6510 | 0.4759 |
| 123 | 19,200 | 4,800 | 48,241 | 0.6406 | 0.6413 | 0.4609 |

## Aggregate over seeds

| Metric | Mean | Std | Range |
|---|---:|---:|---:|
| accuracy | 0.6472 | 0.0044 | 0.6406–0.6523 (0.0117) |
| f1_macro | 0.6478 | 0.0045 | 0.6413–0.6530 (0.0117) |
| cohen_kappa | 0.4709 | 0.0066 | 0.4609–0.4784 (0.0175) |

## Interpretation

The F1 range across seeds is under 0.02, which is small relative to the differences between methods reported in the main tables. The thesis should still note the seed-driven variance when discussing very close comparisons.
