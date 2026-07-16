# Domain-Shift / Robustness Slice Evaluation

- Created at: `2026-05-18T01:47:02Z`
- Dataset: `C:\Users\itsma\OneDrive\Documents\GitHub\Youtube_sentiment_analysis\backend\data\route_a_domain_10k\test.csv`
- Slice column: `CountryCode`
- Slice source: `metadata`
- Samples: `1641`

## Overall

| Model | Accuracy | Macro-F1 | Worst Slice | Worst Slice F1 | Spread |
| --- | ---: | ---: | --- | ---: | ---: |
| logreg | 0.736746 | 0.736832 | IE | 0.667823 | 0.114707 |
| svm | 0.736137 | 0.735756 | IE | 0.689130 | 0.075685 |
| tfidf | 0.678245 | 0.676433 | IN | 0.628792 | 0.094712 |
| ensemble_nsga2 | 0.735527 | 0.735490 | IE | 0.677481 | 0.095839 |
| meta_learner | 0.731261 | 0.732183 | IE | 0.664289 | 0.099956 |

## logreg Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| IE | 109 | 0.678899 | 0.667823 |
| AU | 248 | 0.697581 | 0.695733 |
| GB | 193 | 0.720207 | 0.716402 |
| CA | 226 | 0.743363 | 0.730337 |
| US | 496 | 0.739919 | 0.740585 |
| DE | 83 | 0.759036 | 0.760361 |
| IN | 163 | 0.797546 | 0.774974 |
| NZ | 113 | 0.778761 | 0.782530 |

## svm Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| IE | 109 | 0.697248 | 0.689130 |
| CA | 226 | 0.716814 | 0.698520 |
| AU | 248 | 0.725806 | 0.720179 |
| GB | 193 | 0.730570 | 0.727002 |
| US | 496 | 0.735887 | 0.736488 |
| DE | 83 | 0.734940 | 0.736497 |
| IN | 163 | 0.785276 | 0.763219 |
| NZ | 113 | 0.761062 | 0.764815 |

## tfidf Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| IN | 163 | 0.674847 | 0.628792 |
| AU | 248 | 0.657258 | 0.650851 |
| GB | 193 | 0.668394 | 0.662602 |
| US | 496 | 0.667339 | 0.666818 |
| IE | 109 | 0.678899 | 0.669285 |
| CA | 226 | 0.707965 | 0.678084 |
| DE | 83 | 0.710843 | 0.719807 |
| NZ | 113 | 0.725664 | 0.723504 |

## ensemble_nsga2 Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| IE | 109 | 0.688073 | 0.677481 |
| AU | 248 | 0.705645 | 0.703920 |
| GB | 193 | 0.715026 | 0.710237 |
| CA | 226 | 0.743363 | 0.726445 |
| US | 496 | 0.737903 | 0.738442 |
| IN | 163 | 0.785276 | 0.756694 |
| DE | 83 | 0.759036 | 0.760361 |
| NZ | 113 | 0.769912 | 0.773320 |

## meta_learner Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| IE | 109 | 0.669725 | 0.664289 |
| AU | 248 | 0.697581 | 0.696491 |
| GB | 193 | 0.720207 | 0.718183 |
| IN | 163 | 0.760736 | 0.719130 |
| US | 496 | 0.733871 | 0.734810 |
| DE | 83 | 0.746988 | 0.747932 |
| CA | 226 | 0.761062 | 0.749744 |
| NZ | 113 | 0.761062 | 0.764245 |

## Interpretation

This is a metadata-backed domain-slice evaluation.
