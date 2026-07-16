# Coverage-Accuracy Curve Analysis

## Method

Samples are sorted by model confidence (max class probability, descending). At each coverage level the accuracy and macro-F1 are computed on the covered subset. AUCA = Area Under the Coverage-Accuracy curve (trapezoidal); higher = model's confidence better predicts its own correctness.

## Summary: AUCA and AUC-F1 (Test Set)

| Rank | Model | AUCA | AUC-F1 | Acc@100% | F1@100% |
|------|-------|------|--------|----------|---------|
| 1 | svm | 0.8806 | 0.8780 | 0.7889 | 0.7863 |
| 2 | ensemble | 0.8781 | 0.8749 | 0.8000 | 0.7988 |
| 3 | neuro_fuzzy | 0.8747 | 0.8694 | 0.7889 | 0.7870 |
| 4 | logreg | 0.8679 | 0.8624 | 0.7556 | 0.7544 |
| 5 | meta_learner | 0.8663 | 0.8615 | 0.7556 | 0.7554 |
| 6 | tfidf | 0.8243 | 0.8127 | 0.7167 | 0.7169 |

## Coverage-Accuracy at Key Coverage Levels

| Model | Acc@10% | Acc@25% | Acc@50% | Acc@75% | Acc@100% |
|-------|---------|---------|---------|---------|----------|
| svm | 1.0000 | 1.0000 | 0.9444 | 0.8593 | 0.7889 |
| ensemble | 1.0000 | 1.0000 | 0.9556 | 0.8593 | 0.8000 |
| neuro_fuzzy | 1.0000 | 1.0000 | 0.9333 | 0.8593 | 0.7889 |
| logreg | 1.0000 | 1.0000 | 0.9222 | 0.8593 | 0.7556 |
| meta_learner | 1.0000 | 0.9778 | 0.9333 | 0.8444 | 0.7556 |
| tfidf | 1.0000 | 0.9778 | 0.8889 | 0.8074 | 0.7167 |

## Macro-F1 at Key Coverage Levels

| Model | F1@10% | F1@25% | F1@50% | F1@75% | F1@100% |
|-------|--------|--------|--------|--------|---------|
| svm | 1.0000 | 1.0000 | 0.9398 | 0.8519 | 0.7863 |
| ensemble | 1.0000 | 1.0000 | 0.9524 | 0.8520 | 0.7988 |
| neuro_fuzzy | 1.0000 | 1.0000 | 0.9238 | 0.8514 | 0.7870 |
| logreg | 1.0000 | 1.0000 | 0.9127 | 0.8539 | 0.7544 |
| meta_learner | 1.0000 | 0.9581 | 0.9234 | 0.8391 | 0.7554 |
| tfidf | 1.0000 | 0.9575 | 0.8718 | 0.8008 | 0.7169 |

## Thesis Interpretation

The best selective predictor by AUCA is **svm** (AUCA=0.8806), meaning its confidence scores are most informative about its own correctness.

The weakest is **tfidf** (AUCA=0.8243), indicating its confidence is less discriminative.

The gap between AUCA values across models reveals that **confidence quality varies significantly by architecture**, independent of raw accuracy. A model with lower full-coverage F1 can still have higher AUCA if it abstains intelligently — this is the core argument for selective prediction in the thesis CI chapter.

### Key Findings

- **tfidf** shows the largest accuracy lift at 10% coverage: 1.0000 vs 0.7167 full coverage (+0.2833)
- Neuro-fuzzy gate AUCA vs static ensemble: 0.8747 vs 0.8781