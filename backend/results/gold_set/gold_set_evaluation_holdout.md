# Gold Set Evaluation, Held-Out-Only Subset

- Full gold set: 300 items (95 are exact-text members of the training split and excluded here; see `gold_set_train_membership.py`).
- Held-out subset (val + test + no-match): **205 items** (200 with a non-disputed human label).

## Model Performance vs Human-Reconciled Gold Labels (held-out subset)

| Model | Accuracy | Macro F1 | Weighted F1 | N |
|-------|----------|----------|-------------|---|
| logreg | 0.6950 | 0.6990 | 0.6968 | 200 |
| svm | 0.7050 | 0.7091 | 0.7067 | 200 |
| tfidf | 0.7050 | 0.7049 | 0.7027 | 200 |
| ensemble_pso | 0.7050 | 0.7088 | 0.7067 | 200 |
| ensemble_nsga2 | 0.7000 | 0.7039 | 0.7018 | 200 |
| meta_learner | 0.7000 | 0.7036 | 0.7017 | 200 |

> Compare against `results/gold_set/gold_set_evaluation.md` Table 1 (all 300 items, including 95 that overlap the training split). If accuracy/macro-F1 drop materially here, the full-gold-set figures were inflated by training-set memorisation.
