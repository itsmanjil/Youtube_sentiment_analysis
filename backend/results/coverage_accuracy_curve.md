# Coverage-Accuracy Curve Analysis

## Method

Samples are sorted by model confidence (max class probability, descending). At each coverage level the accuracy and macro-F1 are computed on the covered subset. AUCA = Area Under the Coverage-Accuracy curve (trapezoidal); higher = model's confidence better predicts its own correctness.

## Summary: AUCA and AUC-F1 (Test Set)

| Rank | Model | AUCA | AUC-F1 | Acc@100% | F1@100% |
|------|-------|------|--------|----------|---------|
| 1 | meta_learner | 0.8309 | 0.7479 | 0.6973 | 0.6967 |
| 2 | neuro_fuzzy | 0.8304 | 0.8138 | 0.6972 | 0.6955 |
| 3 | logreg | 0.8300 | 0.8124 | 0.6957 | 0.6943 |
| 4 | ensemble | 0.8288 | 0.8135 | 0.6956 | 0.6938 |
| 5 | svm | 0.8152 | 0.8029 | 0.6835 | 0.6817 |
| 6 | tfidf | 0.7975 | 0.7672 | 0.6630 | 0.6579 |

## Coverage-Accuracy at Key Coverage Levels

| Model | Acc@10% | Acc@25% | Acc@50% | Acc@75% | Acc@100% |
|-------|---------|---------|---------|---------|----------|
| meta_learner | 0.9765 | 0.9387 | 0.8481 | 0.7737 | 0.6973 |
| neuro_fuzzy | 0.9760 | 0.9317 | 0.8511 | 0.7732 | 0.6972 |
| logreg | 0.9780 | 0.9313 | 0.8500 | 0.7726 | 0.6957 |
| ensemble | 0.9790 | 0.9304 | 0.8468 | 0.7714 | 0.6956 |
| svm | 0.9665 | 0.9148 | 0.8306 | 0.7589 | 0.6835 |
| tfidf | 0.9630 | 0.9056 | 0.8082 | 0.7329 | 0.6630 |

## Macro-F1 at Key Coverage Levels

| Model | F1@10% | F1@25% | F1@50% | F1@75% | F1@100% |
|-------|--------|--------|--------|--------|---------|
| meta_learner | 0.3294 | 0.9088 | 0.8357 | 0.7690 | 0.6967 |
| neuro_fuzzy | 0.9216 | 0.9075 | 0.8383 | 0.7674 | 0.6955 |
| logreg | 0.9334 | 0.9084 | 0.8373 | 0.7672 | 0.6943 |
| ensemble | 0.9507 | 0.9058 | 0.8326 | 0.7650 | 0.6938 |
| svm | 0.9433 | 0.8929 | 0.8186 | 0.7531 | 0.6817 |
| tfidf | 0.8919 | 0.8628 | 0.7821 | 0.7194 | 0.6579 |

## Thesis Interpretation

The best selective predictor by AUCA is **meta_learner** (AUCA=0.8309), meaning its confidence scores are most informative about its own correctness.

The weakest is **tfidf** (AUCA=0.7975), indicating its confidence is less discriminative.

The gap between AUCA values across models reveals that **confidence quality varies significantly by architecture**, independent of raw accuracy. A model with lower full-coverage F1 can still have higher AUCA if it abstains intelligently — this is the core argument for selective prediction in the thesis CI chapter.

### Key Findings

- **tfidf** shows the largest accuracy lift at 10% coverage: 0.9630 vs 0.6630 full coverage (+0.3000)
- Neuro-fuzzy gate AUCA vs static ensemble: 0.8304 vs 0.8288