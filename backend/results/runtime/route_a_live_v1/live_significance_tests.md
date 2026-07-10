# Live-Runtime Significance Tests & Bootstrap Confidence Intervals

- Runtime artifact: `route_a_live_v1`  |  Dataset: `data/test.csv` (n = 165,110)
- Bootstrap: 2000 resamples (seed 42)
- NSGA-II knee weights: {"logreg": 0.916081, "svm": 0.002738, "tfidf": 0.081181} (served T = 1.0)
- PSO weights: {"logreg": 0.880485, "svm": 0.0, "tfidf": 0.119515} (served T = 1.0)

## Reproduction validation (reconstructed vs pinned benchmark)

| Model | Acc (repro/pinned) | Macro-F1 (repro/pinned) | ECE (repro/pinned) | Validated |
| --- | --- | --- | --- | --- |
| logreg | 0.6946 / 0.6946 | 0.6928 / 0.6928 | 0.003900 / 0.003900 | yes |
| svm | 0.6801 / 0.6801 | 0.6780 / 0.6780 | 0.015690 / 0.015690 | yes |
| tfidf | 0.6622 / 0.6622 | 0.6567 / 0.6567 | 0.017889 / 0.017889 | yes |
| meta_learner | 0.6955 / 0.6955 | 0.6946 / 0.6946 | 0.018303 / 0.018303 | yes |
| ensemble_pso | 0.6961 / 0.6961 | 0.6941 / 0.6941 | 0.006103 / 0.006103 | yes |
| ensemble_nsga2 | 0.6959 / 0.6959 | 0.6940 / 0.6940 | 0.003919 / 0.003919 | yes |
| fuzzy_ensemble | 0.6960 / 0.6960 | 0.6940 / 0.6940 | 0.002987 / 0.002987 | yes |

## Paired McNemar tests (label correctness, Holm-adjusted)

| Model A | Model B | A correct / B wrong | A wrong / B correct | p (raw) | p (Holm) | Significant (a=0.05) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ensemble_nsga2 | meta_learner | 2233 | 2164 | 3.051e-01 | 1.000e+00 | no |
| ensemble_nsga2 | ensemble_pso | 430 | 465 | 2.557e-01 | 1.000e+00 | no |
| ensemble_nsga2 | logreg | 968 | 758 | 4.888e-07 | 3.421e-06 | yes |
| fuzzy_ensemble | ensemble_nsga2 | 1224 | 1202 | 6.698e-01 | 1.000e+00 | no |
| ensemble_nsga2 | fuzzy_ensemble | 1202 | 1224 | 6.698e-01 | 1.000e+00 | no |
| meta_learner | logreg | 2510 | 2369 | 4.504e-02 | 2.702e-01 | no |
| meta_learner | ensemble_nsga2 | 2164 | 2233 | 3.051e-01 | 1.000e+00 | no |

## Paired bootstrap 95% CIs on metric differences (A - B)

| Model A | Model B | Metric | Point diff | 95% CI | Excludes 0 |
| --- | --- | --- | ---: | --- | --- |
| meta_learner | logreg | macro_f1 | +0.00176 | [+0.00095, +0.00260] | yes |
| meta_learner | ensemble_nsga2 | macro_f1 | +0.00062 | [-0.00020, +0.00142] | no |
| ensemble_nsga2 | meta_learner | ece | -0.01438 | [-0.01732, -0.01053] | yes |
| ensemble_nsga2 | logreg | ece | +0.00002 | [-0.00136, +0.00070] | no |
| ensemble_nsga2 | ensemble_pso | ece | -0.00218 | [-0.00263, -0.00084] | yes |
| ensemble_nsga2 | ensemble_pso | macro_f1 | -0.00015 | [-0.00049, +0.00023] | no |
| ensemble_nsga2 | fuzzy_ensemble | macro_f1 | -0.00001 | [-0.00061, +0.00056] | no |
| fuzzy_ensemble | ensemble_nsga2 | ece | -0.00093 | [-0.00235, +0.00135] | no |

_For an ECE difference, a negative value favours Model A (lower calibration error). A CI excluding zero indicates a statistically reliable difference at the 5% level._
