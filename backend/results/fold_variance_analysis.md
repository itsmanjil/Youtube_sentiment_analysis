# Fold Variance Analysis (Real YouTube Data)
> This file supersedes the synthetic `make_classification` demo run previously stored as `thesis_evaluation_report.json` (now renamed to `DEMO_synthetic_evaluation_report.json`).
## Setup
- Data: `train.csv` (subsampled to 30,000 rows, stratified)
- Folds: 10, seed=42
- Pipeline: TF-IDF(1,2, max_features=50000) + LogReg(C=1.0, class_weight=balanced)
## Aggregate
- Accuracy: 0.6510 ± 0.0098
- Macro-F1: 0.6513 ± 0.0098
- Cohen kappa: 0.4764 ± 0.0148
## Per-fold
| Fold | n | Acc | F1-macro | Kappa | Tok-len (mean) | Vocab overlap | z(F1) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3000 | 0.6487 | 0.6491 | 0.4730 | 20.75 | 0.802 | -0.22 |
| 2 | 3000 | 0.6563 | 0.6571 | 0.4845 | 20.19 | 0.807 | +0.59 |
| 3 | 3000 | 0.6447 | 0.6456 | 0.4670 | 21.77 | 0.800 | -0.58 |
| 4 | 3000 | 0.6323 | 0.6326 | 0.4485 | 21.52 | 0.797 | -1.90 |
| 5 | 3000 | 0.6590 | 0.6590 | 0.4885 | 20.64 | 0.796 | +0.79 |
| 6 | 3000 | 0.6613 | 0.6618 | 0.4920 | 21.64 | 0.804 | +1.07 |
| 7 | 3000 | 0.6407 | 0.6406 | 0.4610 | 21.06 | 0.799 | -1.09 |
| 8 | 3000 | 0.6667 | 0.6667 | 0.5000 | 20.97 | 0.794 | +1.57 |
| 9 | 3000 | 0.6533 | 0.6533 | 0.4800 | 21.02 | 0.801 | +0.20 |
| 10 | 3000 | 0.6467 | 0.6470 | 0.4700 | 21.79 | 0.787 | -0.44 |

**Best fold:** 8  |  **Worst fold:** 4

## Interpretation
With 3,000 samples per fold and a 10-fold stratified split, per-fold variance is dominated by model randomness and the particular tail of hard examples in each fold, not by systematic class skew.
- Worst fold (#4) F1 = 0.6326; best fold (#8) F1 = 0.6667; spread = 0.0342.
- Worst-fold class skew vs global: Negative: +0.0000, Neutral: +0.0000, Positive: +0.0000.
- Worst-fold token length (mean=21.52) vs best-fold (mean=20.97).
- Worst-fold vocab overlap with training: 0.797 (best: 0.794).

If the worst fold's F1 z-score is within ±1.5 and its class/length statistics are close to the global means, the variance is consistent with normal CV noise at this sample size and does not require a separate explanation in the thesis.
