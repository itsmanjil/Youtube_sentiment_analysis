# Offline vs Live Reconciliation

- Runtime artifact version: `route_a_live_v1`
- Dataset: `/Users/deadshot/Documents/GitHub/Youtube_sentiment_analysis/backend/data/test.csv`
- Samples: `165110`
- Offline source: `backend/results/thesis_model_performance_youtube_filtered.md`
- Live source: `backend/results/runtime/<version>/live_runtime_benchmark_full_test.json`

## Same-Name Models

| Model | Offline Acc | Live Acc | Δ Acc | Offline F1 | Live F1 | Δ F1 | Offline ECE | Live ECE | Δ ECE | Offline Brier | Live Brier | Δ Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.6622 | 0.6622 | +0.000000 | 0.6567 | 0.6567 | +0.000000 | 0.010556 | 0.017889 | +0.007333 | 0.448593 | 0.449058 | +0.000465 |
| logreg | 0.6946 | 0.6946 | +0.000000 | 0.6928 | 0.6928 | +0.000000 | 0.004468 | 0.003900 | -0.000568 | 0.410001 | 0.410009 | +0.000008 |
| svm | 0.6801 | 0.6801 | +0.000000 | 0.6780 | 0.6780 | +0.000000 | 0.015690 | 0.016953 | +0.001263 | 0.429094 | 0.429259 | +0.000165 |
| meta_learner | 0.6955 | 0.6953 | -0.000200 | 0.6946 | 0.6945 | -0.000100 | 0.018303 | 0.015711 | -0.002592 | 0.411778 | 0.411713 | -0.000065 |

## Ensemble Mapping

| Offline Row | Live Row | Offline Acc | Live Acc | Δ Acc | Offline F1 | Live F1 | Δ F1 | Offline ECE | Live ECE | Δ ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble | ensemble_pso | 0.6932 | 0.6872 | -0.006000 | 0.6909 | 0.6852 | -0.005700 | 0.019060 | 0.011272 | -0.007788 |
| ensemble | ensemble_nsga2 | 0.6932 | 0.6959 | +0.002700 | 0.6909 | 0.6940 | +0.003100 | 0.019060 | 0.004601 | -0.014459 |

## Live-Only Rows

| Model | Accuracy | Macro-F1 | ECE | Brier | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| fuzzy_ensemble | 0.6622 | 0.6567 | 0.018516 | 0.448920 | Pinned runtime row with no direct offline counterpart |

## Conclusions

- Direct same-name models remain numerically aligned on accuracy/macro-F1, with only small calibration drift between offline and live artifacts.
- The historical offline `ensemble` row should not be treated as the live runtime default anymore; the live stack now exposes explicit `ensemble_pso` and `ensemble_nsga2` variants.
- The best live ensemble variant is `ensemble_nsga2` with macro-F1 0.6940.
