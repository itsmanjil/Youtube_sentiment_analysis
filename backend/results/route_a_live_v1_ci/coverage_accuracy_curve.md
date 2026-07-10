# Coverage-Accuracy Curve Analysis

## Method

Samples are sorted by model confidence (max class probability, descending). At each coverage level the accuracy and macro-F1 are computed on the covered subset. AUCA = Area Under the Coverage-Accuracy curve (trapezoidal); higher = model's confidence better predicts its own correctness.

## Summary: AUCA and AUC-F1 (Test Set)

| Rank | Model | AUCA | AUC-F1 | Acc@100% | F1@100% |
|------|-------|------|--------|----------|---------|
| 1 | ensemble_pso | 0.8311 | 0.8137 | 0.6966 | 0.6951 |
| 2 | meta_learner | 0.8309 | 0.7479 | 0.6973 | 0.6967 |
| 3 | ensemble_nsga2 | 0.8308 | 0.8128 | 0.6964 | 0.6949 |
| 4 | fuzzy_ensemble | 0.8307 | 0.8135 | 0.6976 | 0.6960 |
| 5 | logreg | 0.8300 | 0.8123 | 0.6957 | 0.6943 |
| 6 | svm | 0.8152 | 0.8029 | 0.6835 | 0.6817 |
| 7 | tfidf | 0.7972 | 0.7667 | 0.6630 | 0.6579 |

## Coverage-Accuracy at Key Coverage Levels

| Model | Acc@10% | Acc@25% | Acc@50% | Acc@75% | Acc@100% |
|-------|---------|---------|---------|---------|----------|
| ensemble_pso | 0.9780 | 0.9335 | 0.8518 | 0.7736 | 0.6966 |
| meta_learner | 0.9765 | 0.9387 | 0.8481 | 0.7737 | 0.6973 |
| ensemble_nsga2 | 0.9775 | 0.9344 | 0.8514 | 0.7735 | 0.6964 |
| fuzzy_ensemble | 0.9775 | 0.9323 | 0.8504 | 0.7740 | 0.6976 |
| logreg | 0.9780 | 0.9313 | 0.8501 | 0.7730 | 0.6957 |
| svm | 0.9665 | 0.9148 | 0.8306 | 0.7589 | 0.6835 |
| tfidf | 0.9620 | 0.9054 | 0.8079 | 0.7328 | 0.6630 |

## Macro-F1 at Key Coverage Levels

| Model | F1@10% | F1@25% | F1@50% | F1@75% | F1@100% |
|-------|--------|--------|--------|--------|---------|
| ensemble_pso | 0.9297 | 0.9091 | 0.8381 | 0.7677 | 0.6951 |
| meta_learner | 0.3294 | 0.9088 | 0.8357 | 0.7690 | 0.6967 |
| ensemble_nsga2 | 0.9289 | 0.9107 | 0.8383 | 0.7678 | 0.6949 |
| fuzzy_ensemble | 0.9253 | 0.9089 | 0.8369 | 0.7682 | 0.6960 |
| logreg | 0.9334 | 0.9084 | 0.8373 | 0.7676 | 0.6943 |
| svm | 0.9433 | 0.8929 | 0.8186 | 0.7531 | 0.6817 |
| tfidf | 0.8879 | 0.8632 | 0.7816 | 0.7191 | 0.6579 |

## Thesis Interpretation

The best selective predictor by AUCA is **ensemble_pso** (AUCA=0.8311), meaning its confidence scores are most informative about its own correctness.

The weakest is **tfidf** (AUCA=0.7972), indicating its confidence is less discriminative.

The gap between AUCA values across models reveals that **confidence quality varies significantly by architecture**, independent of raw accuracy. A model with lower full-coverage F1 can still have higher AUCA if it abstains intelligently — this is the core argument for selective prediction in the thesis CI chapter.

### Key Findings

- **tfidf** shows the largest accuracy lift at 10% coverage: 0.9620 vs 0.6630 full coverage (+0.2990)
- `fuzzy_ensemble` AUCA vs `ensemble_nsga2` AUCA: 0.8307 vs 0.8308