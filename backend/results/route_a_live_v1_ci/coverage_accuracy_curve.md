# Coverage-Accuracy Curve Analysis

## Method

Samples are sorted by model confidence (max class probability, descending). At each coverage level the accuracy and macro-F1 are computed on the covered subset. AUCA = Area Under the Coverage-Accuracy curve (trapezoidal); higher = model's confidence better predicts its own correctness.

## Summary: AUCA and AUC-F1 (Test Set)

| Rank | Model | AUCA | AUC-F1 | Acc@100% | F1@100% |
|------|-------|------|--------|----------|---------|
| 1 | meta_learner | 0.8009 | 0.7387 | 0.6970 | 0.6965 |
| 2 | logreg | 0.8003 | 0.7859 | 0.6957 | 0.6943 |
| 3 | ensemble | 0.7937 | 0.7822 | 0.6912 | 0.6896 |
| 4 | svm | 0.7856 | 0.7739 | 0.6835 | 0.6817 |
| 5 | tfidf | 0.7676 | 0.7406 | 0.6630 | 0.6579 |

## Coverage-Accuracy at Key Coverage Levels

| Model | Acc@10% | Acc@25% | Acc@50% | Acc@75% | Acc@100% |
|-------|---------|---------|---------|---------|----------|
| meta_learner | 0.9765 | 0.9322 | 0.8481 | 0.7702 | 0.6970 |
| logreg | 0.9780 | 0.9280 | 0.8501 | 0.7701 | 0.6957 |
| ensemble | 0.9740 | 0.9212 | 0.8429 | 0.7627 | 0.6912 |
| svm | 0.9670 | 0.9118 | 0.8308 | 0.7553 | 0.6835 |
| tfidf | 0.9620 | 0.9020 | 0.8079 | 0.7294 | 0.6630 |

## Macro-F1 at Key Coverage Levels

| Model | F1@10% | F1@25% | F1@50% | F1@75% | F1@100% |
|-------|--------|--------|--------|--------|---------|
| meta_learner | 0.3294 | 0.9030 | 0.8360 | 0.7660 | 0.6965 |
| logreg | 0.9334 | 0.9057 | 0.8373 | 0.7650 | 0.6943 |
| ensemble | 0.9521 | 0.9001 | 0.8310 | 0.7572 | 0.6896 |
| svm | 0.9446 | 0.8903 | 0.8189 | 0.7497 | 0.6817 |
| tfidf | 0.8879 | 0.8599 | 0.7816 | 0.7165 | 0.6579 |

## Thesis Interpretation

The best selective predictor by AUCA is **meta_learner** (AUCA=0.8009), meaning its confidence scores are most informative about its own correctness.

The weakest is **tfidf** (AUCA=0.7676), indicating its confidence is less discriminative.

The gap between AUCA values across models reveals that **confidence quality varies significantly by architecture**, independent of raw accuracy. A model with lower full-coverage F1 can still have higher AUCA if it abstains intelligently — this is the core argument for selective prediction in the thesis CI chapter.

### Key Findings

- **tfidf** shows the largest accuracy lift at 10% coverage: 0.9620 vs 0.6630 full coverage (+0.2990)
