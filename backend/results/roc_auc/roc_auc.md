# ROC-AUC Evaluation (One-vs-Rest)

- Dataset: `data\test.csv`
- Samples: `5000`
- Method: One-vs-Rest (OvR) per class; macro-average across classes

## Macro and Weighted AUC

| Model | Macro AUC | Weighted AUC |
|-------|-----------|--------------|
| ensemble_nsga2 | 0.8596 | 0.8606 |
| logreg | 0.8589 | 0.8600 |
| meta_learner | 0.8577 | 0.8587 |
| ensemble_pso | 0.8511 | 0.8522 |
| svm | 0.8434 | 0.8446 |
| tfidf | 0.8315 | 0.8323 |
| fuzzy_ensemble | 0.8314 | 0.8322 |

## Per-Class AUC (OvR)

| Model | Positive AUC | Neutral AUC | Negative AUC |
|-------|-------------|------------|--------------|
| ensemble_nsga2 | 0.8948 | 0.8184 | 0.8655 |
| logreg | 0.8945 | 0.8172 | 0.8652 |
| meta_learner | 0.8922 | 0.8186 | 0.8623 |
| ensemble_pso | 0.8897 | 0.8060 | 0.8575 |
| svm | 0.8847 | 0.7954 | 0.8501 |
| tfidf | 0.8684 | 0.7925 | 0.8335 |
| fuzzy_ensemble | 0.8680 | 0.7923 | 0.8337 |

## Thesis Interpretation

ROC-AUC is threshold-independent and measures how well the model's
probability scores separate each sentiment class from the rest.
A macro AUC of 1.0 is perfect; 0.5 is no better than random.

Neutral class typically has the lowest per-class AUC, consistent
with its lower F1 scores observed across all models — reflecting
the inherent ambiguity of neutral comments.
