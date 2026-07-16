# ROC-AUC Evaluation (One-vs-Rest)

- Dataset: `data\test.csv`
- Samples: `5000`
- Method: One-vs-Rest (OvR) per class; macro-average across classes

## Macro and Weighted AUC

| Model | Macro AUC | Weighted AUC |
|-------|-----------|--------------|
| ensemble_pso | 0.8597 | 0.8608 |
| ensemble_nsga2 | 0.8596 | 0.8606 |
| fuzzy_ensemble | 0.8591 | 0.8602 |
| logreg | 0.8589 | 0.8600 |
| meta_learner | 0.8575 | 0.8585 |
| svm | 0.8435 | 0.8447 |
| tfidf | 0.8315 | 0.8323 |

## Per-Class AUC (OvR)

| Model | Positive AUC | Neutral AUC | Negative AUC |
|-------|-------------|------------|--------------|
| ensemble_pso | 0.8949 | 0.8186 | 0.8656 |
| ensemble_nsga2 | 0.8949 | 0.8183 | 0.8655 |
| fuzzy_ensemble | 0.8943 | 0.8182 | 0.8649 |
| logreg | 0.8945 | 0.8172 | 0.8652 |
| meta_learner | 0.8919 | 0.8186 | 0.8621 |
| svm | 0.8846 | 0.7956 | 0.8501 |
| tfidf | 0.8684 | 0.7925 | 0.8335 |

## Thesis Interpretation

ROC-AUC is threshold-independent and measures how well the model's
probability scores separate each sentiment class from the rest.
A macro AUC of 1.0 is perfect; 0.5 is no better than random.

Neutral class typically has the lowest per-class AUC, consistent
with its lower F1 scores observed across all models — reflecting
the inherent ambiguity of neutral comments.
