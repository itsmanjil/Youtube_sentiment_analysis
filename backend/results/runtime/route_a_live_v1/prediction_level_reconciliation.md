# Prediction-Level Offline vs Live Reconciliation

- Created at: `2026-05-17T07:49:14Z`
- Runtime artifact version: `route_a_live_v1`
- Offline probability cube: `C:\Users\itsma\OneDrive\Documents\GitHub\Youtube_sentiment_analysis\backend\results\prob_cubes\route_a_benchmark_cpu_test_deberta_logreg_svm.npz`
- Text source: `reconstructed_source_csv`
- Samples: `180`
- Probability tolerance: `1e-06`
- Label-equivalence status: `PASS`
- Strict probability-equivalence status: `FAIL`

| Model | Samples | Label Match Rate | Mismatches | Max Prob Delta | Mean Prob Delta | Prob Tol Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logreg | 180 | 1.00000000 | 0 | 0.0102905405 | 0.0042952665 | no |
| svm | 180 | 1.00000000 | 0 | 0.0108004988 | 0.0046647016 | no |

## Interpretation

The live runtime reproduced every offline probability-cube label. Probability deltas are reported separately because calibration or environment-level floating-point differences can change confidence without changing the predicted class.
