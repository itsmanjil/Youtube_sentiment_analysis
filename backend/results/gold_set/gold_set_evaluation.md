# Gold Set Evaluation (300 samples)

## Inter-Annotator Agreement

| Metric | Value |
| --- | --- |
| Percent agreement | 0.9700 |
| Krippendorff's Î± | 0.9547 |
| Fleiss' Îº | 0.9546 |
| Cohen's Îº (annotator_1 vs annotator_2) | 0.9546 |

*strong agreement (alpha >= 0.80)*

## Table 1: Model Performance vs Human-Reconciled Gold Labels

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| logreg | 0.6907 | 0.6940 | 0.6931 |
| svm | 0.6942 | 0.6978 | 0.6957 |
| tfidf | 0.7010 | 0.6988 | 0.6995 |
| ensemble_pso | 0.7010 | 0.7042 | 0.7014 |
| ensemble_nsga2 | 0.6976 | 0.7006 | 0.6999 |
| meta_learner | 0.6976 | 0.7001 | 0.7001 |

## Table 2: Model Performance vs Silver Labels (auto-annotated)

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| logreg | 0.9267 | 0.9273 | 0.9268 |
| svm | 0.9733 | 0.9737 | 0.9733 |
| tfidf | 0.7533 | 0.7529 | 0.7492 |
| ensemble_pso | 1.0000 | 1.0000 | 1.0000 |
| ensemble_nsga2 | 0.9267 | 0.9275 | 0.9268 |
| meta_learner | 0.9200 | 0.9192 | 0.9198 |

> **Note:** ensemble_pso scores 1.000 vs silver labels because it *is* the silver labeler (PSO-weighted ensemble). This is expected and does not indicate overfitting.

## Table 3: Per-Class F1 vs Human-Reconciled Gold Labels

| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|-------|--------|--------|--------|----------|
| logreg | 0.6927 | 0.6168 | 0.7725 | 0.6940 |
| svm | 0.7086 | 0.6140 | 0.7708 | 0.6978 |
| tfidf | 0.6734 | 0.5946 | 0.8283 | 0.6988 |
| ensemble_pso | 0.7232 | 0.6226 | 0.7668 | 0.7042 |
| ensemble_nsga2 | 0.6966 | 0.6262 | 0.7789 | 0.7006 |
| meta_learner | 0.6901 | 0.6393 | 0.7708 | 0.7001 |

## Confusion Matrix: ensemble_pso vs Human-Reconciled Gold Labels

| True \ Pred | Negative | Neutral | Positive |
|-------------|----------|---------|----------|
| Negative | 64 | 12 | 3 |
| Neutral | 30 | 66 | 12 |
| Positive | 4 | 26 | 74 |

## Notes

- Human-reconciled gold labels used as primary reference (disputed items excluded: 9).

- Neutral class consistently has the lowest F1 across all models, consistent with inter-annotator difficulty on borderline comments.
- ensemble_pso (PSO-optimised weights, F1-best) and ensemble_nsga2 (NSGA-II knee-point, calibration-best) provide complementary trade-offs.
