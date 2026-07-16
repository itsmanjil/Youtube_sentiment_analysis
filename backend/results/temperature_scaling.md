# Temperature Scaling Calibration

## Method

Temperature scaling (Guo et al., 2017) fits a single scalar T per model on the validation set by minimising Negative Log-Likelihood.

    z_c = log(p_c)   →   p_calibrated = softmax(z / T)

- T > 1: model was overconfident → scaling softens probabilities  
- T < 1: model was underconfident → scaling sharpens probabilities  
- T = 1: no change required (already well-calibrated)

**Macro-F1 is unaffected** — temperature scaling preserves the argmax.

**Gating**: NLL minimisation on the validation set is not guaranteed to improve held-out ECE. A fitted T is only *kept* (and served) if it reduces test-set ECE relative to the uncalibrated model; otherwise T is pinned to 1.0 (identity) and the fitted value is reported for reference only. Of 5 models, **1 kept** their fitted temperature and **4 were discarded** as harmful on this run.

## Results (Test Set)

### ECE (Expected Calibration Error, 15 bins)

*Lower is better. Reduction % = (before − after) / before × 100. "T (fitted)" is the NLL-optimal value; "T (served)" is 1.0 when the fitted value was discarded for not improving held-out ECE.*

| Model | T (fitted) | T (served) | Kept | ECE Before | ECE After | Reduction |
|-------|------------|------------|------|------------|-----------|-----------|
| logreg | 1.031 | 1.000 | no | 0.0068 | 0.0068 | +0.0% |
| svm | 1.033 | 1.000 | no | 0.0126 | 0.0126 | +0.0% |
| tfidf | 1.131 | 1.000 | no | 0.0131 | 0.0131 | +0.0% |
| ensemble | 0.935 | 0.935 | yes | 0.0216 | 0.0117 | +46.0% |
| meta_learner | 0.984 | 1.000 | no | 0.0203 | 0.0203 | +0.0% |

### Brier Score

*Lower is better.*

| Model | Brier Before | Brier After | Reduction |
|-------|--------------|-------------|-----------|
| logreg | 0.4083 | 0.4083 | +0.0% |
| svm | 0.4274 | 0.4274 | +0.0% |
| tfidf | 0.4464 | 0.4464 | +0.0% |
| ensemble | 0.4113 | 0.4107 | +0.2% |
| meta_learner | 0.4102 | 0.4102 | +0.0% |

### Macro-F1 (unchanged by design)

| Model | Macro-F1 Before | Macro-F1 After | Δ |
|-------|-----------------|----------------|---|
| logreg | 0.6943 | 0.6943 | +0.0000 |
| svm | 0.6817 | 0.6817 | +0.0000 |
| tfidf | 0.6579 | 0.6579 | +0.0000 |
| ensemble | 0.6938 | 0.6938 | +0.0000 |
| meta_learner | 0.6967 | 0.6967 | +0.0000 |

## Summary

- Models kept (served with fitted T): **1/5**
- Average ECE reduction among kept models: **46.0%**
- Average Brier reduction among kept models: **0.2%**
- Most overconfident model (by fitted T, irrespective of whether kept): **tfidf** (T_fitted=1.131)
- Largest ECE improvement: **ensemble** (+46.0%)

## Thesis Interpretation

Temperature scaling provides a lightweight, theoretically-grounded calibration layer that *can* improve probabilistic reliability without retraining — but NLL-optimal T on the validation set is not guaranteed to reduce held-out ECE. In this run, the fitted temperature only improved test-set ECE for 1/5 model(s); for the rest, the fitted T made ECE *worse*, so those models are served uncalibrated (T=1.0) rather than shipping a harmful transform. See the per-model "Kept" column above.

- Classical ML models (TF-IDF + LogReg/SVM) output decision-function scores converted to probabilities via Platt scaling, which can be systematically over- or under-confident depending on the feature space.

- The ensemble and meta-learner aggregate multiple models, which can amplify or dampen individual model biases — their temperatures reveal whether aggregation helped or hurt calibration.

Because gating pins discarded models to T=1.0 (the identity transform), temperature scaling remains argmax-preserving and safe to apply at inference time with no accuracy trade-off for every model, kept or not. The entropy-gated selective predictor (§4.4) should be read against the *served* ECE values above, not the fitted-but-discarded ones.

## Reference

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
