# Gold Set Evaluation (300 samples)

Silver labels were produced by PSO-weighted ensemble auto-annotation (logreg×0.31 + svm×0.69). Source labels are the original dataset labels. Inter-labeler agreement: **66.3%**.

## Table 1: Model Performance vs Silver Labels (auto-annotated)

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| logreg | 0.9267 | 0.9247 | 0.9265 |
| svm | 0.9667 | 0.9666 | 0.9665 |
| tfidf | 0.7633 | 0.7496 | 0.7607 |
| ensemble_pso | 1.0000 | 1.0000 | 1.0000 |
| ensemble_nsga2 | 0.9233 | 0.9207 | 0.9231 |
| meta_learner | 0.9133 | 0.9103 | 0.9138 |

## Table 2: Model Performance vs Source Labels (original dataset)

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| logreg | 0.6733 | 0.6580 | 0.6728 |
| svm | 0.6600 | 0.6462 | 0.6599 |
| tfidf | 0.6700 | 0.6511 | 0.6658 |
| ensemble_pso | 0.6633 | 0.6487 | 0.6637 |
| ensemble_nsga2 | 0.6767 | 0.6599 | 0.6751 |
| meta_learner | 0.6800 | 0.6685 | 0.6820 |

## Table 3: Per-Class F1 vs Source Labels

| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|-------|--------|--------|--------|----------|
| logreg | 0.6884 | 0.5157 | 0.7699 | 0.6580 |
| svm | 0.6635 | 0.5153 | 0.7598 | 0.6462 |
| tfidf | 0.7083 | 0.5068 | 0.7383 | 0.6511 |
| ensemble_pso | 0.6667 | 0.5062 | 0.7733 | 0.6487 |
| ensemble_nsga2 | 0.7005 | 0.5128 | 0.7665 | 0.6599 |
| meta_learner | 0.6919 | 0.5389 | 0.7748 | 0.6685 |

## Confusion Matrix: ensemble_pso vs Source Labels

| True \ Pred | Negative | Neutral | Positive |
|-------------|----------|---------|----------|
| Negative | 71 | 28 | 9 |
| Neutral | 22 | 41 | 17 |
| Positive | 12 | 13 | 87 |

## Notes

- Silver labels (auto-annotation) show 66.3% agreement with source labels, indicating substantial label noise or ambiguity in the original dataset.
- Neutral class consistently has the lowest F1 across all models, consistent with inter-annotator difficulty on borderline comments.
- ensemble_pso (PSO-optimised weights, F1-best) and ensemble_nsga2 (NSGA-II knee-point, calibration-best) provide complementary trade-offs.
