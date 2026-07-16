# Temperature Scaling Calibration

## Method

Temperature scaling (Guo et al., 2017) fits a single scalar T per model on the validation set by minimising Negative Log-Likelihood.

    z_c = log(p_c)   →   p_calibrated = softmax(z / T)

- T > 1: model was overconfident → scaling softens probabilities  
- T < 1: model was underconfident → scaling sharpens probabilities  
- T = 1: no change required (already well-calibrated)

**Macro-F1 is unaffected** — temperature scaling preserves the argmax.

**Gating**: a fitted T is only *kept* (and served) if it reduces **validation**-set ECE relative to the uncalibrated model; otherwise T is pinned to 1.0 (identity) and the fitted value is reported for reference only. This keep/discard decision never inspects the test set — the test-set numbers below are a read-only evaluation of a configuration already fixed on validation. Of 6 models, **2 kept** their fitted temperature and **4 were discarded** as not improving validation ECE on this run.

## Results (Test Set)

### ECE (Expected Calibration Error, 15 bins)

*Lower is better. Reduction % = (before − after) / before × 100. "T (fitted)" is the NLL-optimal value; "T (served)" is 1.0 when the fitted value was discarded for not improving held-out ECE.*

| Model | T (fitted) | T (served) | Kept | ECE Before | ECE After | Reduction |
|-------|------------|------------|------|------------|-----------|-----------|
| logreg | 1.031 | 1.031 | yes | 0.0068 | 0.0074 | -9.1% |
| svm | 1.033 | 1.000 | no | 0.0126 | 0.0126 | +0.0% |
| tfidf | 1.131 | 1.131 | yes | 0.0131 | 0.0174 | -33.1% |
| ensemble_pso | 0.990 | 1.000 | no | 0.0073 | 0.0073 | +0.0% |
| ensemble_nsga2 | 1.001 | 1.000 | no | 0.0060 | 0.0060 | -0.0% |
| meta_learner | 0.984 | 1.000 | no | 0.0203 | 0.0203 | -0.0% |

### Brier Score

*Lower is better.*

| Model | Brier Before | Brier After | Reduction |
|-------|--------------|-------------|-----------|
| logreg | 0.4083 | 0.4084 | -0.0% |
| svm | 0.4274 | 0.4274 | +0.0% |
| tfidf | 0.4464 | 0.4469 | -0.1% |
| ensemble_pso | 0.4073 | 0.4073 | +0.0% |
| ensemble_nsga2 | 0.4075 | 0.4075 | +0.0% |
| meta_learner | 0.4102 | 0.4102 | +0.0% |

### Macro-F1 (unchanged by design)

| Model | Macro-F1 Before | Macro-F1 After | Δ |
|-------|-----------------|----------------|---|
| logreg | 0.6943 | 0.6943 | +0.0000 |
| svm | 0.6817 | 0.6817 | +0.0000 |
| tfidf | 0.6579 | 0.6579 | +0.0000 |
| ensemble_pso | 0.6951 | 0.6951 | +0.0000 |
| ensemble_nsga2 | 0.6949 | 0.6949 | +0.0000 |
| meta_learner | 0.6967 | 0.6967 | +0.0000 |

## Summary

- Models kept (served with fitted T): **2/6**
- Average ECE reduction among kept models: **-21.1%**
- Average Brier reduction among kept models: **-0.1%**
- Most overconfident model (by fitted T, irrespective of whether kept): **tfidf** (T_fitted=1.131)
- Largest ECE improvement: **svm** (+0.0%)

## Thesis Interpretation

Temperature scaling provides a lightweight, theoretically-grounded calibration layer that *can* improve probabilistic reliability without retraining — but NLL-optimal T is not guaranteed to reduce ECE on the same split it was fitted on. In this run, the fitted temperature only improved validation-set ECE for 2/6 model(s); for the rest, the fitted T made validation ECE *worse*, so those models are served uncalibrated (T=1.0) rather than shipping a harmful transform. This decision is made entirely on validation data; the test-set ECE/Brier columns above are a read-only check of the already-fixed serving configuration, not part of the gating decision. See the per-model "Kept" column above.

- Classical ML models (TF-IDF + LogReg/SVM) output decision-function scores converted to probabilities via Platt scaling, which can be systematically over- or under-confident depending on the feature space.

- The ensemble and meta-learner aggregate multiple models, which can amplify or dampen individual model biases — their temperatures reveal whether aggregation helped or hurt calibration.

Because gating pins discarded models to T=1.0 (the identity transform), temperature scaling remains argmax-preserving and safe to apply at inference time with no accuracy trade-off for every model, kept or not. The entropy-gated selective predictor (§4.4) should be read against the *served* ECE values above, not the fitted-but-discarded ones.

## Reference

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
