# Live Runtime Significance Tests

- Dataset: `data/test.csv`
- Samples: `165110`
- Bootstrap resamples: `2000`, seed `42`
- Runtime artifact version: `route_a_live_v1`

## Reconstruction Validation

| Model | Accuracy | Macro-F1 | ECE | Validated |
| --- | ---: | ---: | ---: | --- |
| logreg | 0.6946 | 0.6928 | 0.0039 | yes |
| svm | 0.6801 | 0.6780 | 0.0170 | yes |
| tfidf | 0.6622 | 0.6567 | 0.0179 | yes |
| meta_learner | 0.6953 | 0.6945 | 0.0157 | yes |
| ensemble_pso | 0.6872 | 0.6852 | 0.0113 | yes |
| ensemble_nsga2 | 0.6959 | 0.6940 | 0.0046 | yes |

## Paired McNemar Tests (Holm-corrected)

| Model A | Model B | n(A wrong,B right) | n(A right,B wrong) | p_raw | p_holm | sig(0.05) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| meta_learner | logreg | 2421 | 2527 | 1.355e-01 | 4.065e-01 | no |
| ensemble_nsga2 | logreg | 693 | 898 | 3.147e-07 | 1.259e-06 | yes |
| ensemble_nsga2 | ensemble_pso | 4184 | 5620 | 1.347e-47 | 6.734e-47 | yes |
| ensemble_nsga2 | meta_learner | 2250 | 2349 | 1.484e-01 | 4.065e-01 | no |
| meta_learner | ensemble_nsga2 | 2349 | 2250 | 1.484e-01 | 4.065e-01 | no |

## Paired Bootstrap 95% CIs

| Model A | Model B | Metric | Point Diff | 95% CI Low | 95% CI High | Excludes Zero |
| --- | --- | --- | ---: | ---: | ---: | --- |
| meta_learner | logreg | macro_f1 | 0.001675 | 0.000880 | 0.002524 | yes |
| meta_learner | ensemble_nsga2 | macro_f1 | 0.000546 | -0.000210 | 0.001346 | no |
| ensemble_nsga2 | ensemble_pso | macro_f1 | 0.008818 | 0.007628 | 0.010042 | yes |
| ensemble_nsga2 | meta_learner | ece | -0.011109 | -0.012622 | -0.008351 | yes |
| ensemble_nsga2 | logreg | ece | 0.000701 | -0.002111 | 0.003128 | no |
| ensemble_nsga2 | ensemble_pso | ece | -0.006671 | -0.008890 | -0.003564 | yes |

## Interpretation

The NSGA-II ensemble's ECE advantage over the meta-learner is statistically significant (CI excludes zero) while the macro-F1 difference between the two is not (tied). The NSGA-II vs. logreg ECE difference is also tied. The meta-learner's macro-F1 edge over logreg is significant. See `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.6 for the narrative reading.