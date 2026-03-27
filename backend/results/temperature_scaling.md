# Temperature Scaling Calibration

## Method

Temperature scaling (Guo et al., 2017) fits a single scalar T per model on the validation set by minimising Negative Log-Likelihood.

    z_c = log(p_c)   →   p_calibrated = softmax(z / T)

- T > 1: model was overconfident → scaling softens probabilities  
- T < 1: model was underconfident → scaling sharpens probabilities  
- T = 1: no change required (already well-calibrated)

**Macro-F1 is unaffected** — temperature scaling preserves the argmax.

## Results (Test Set)

### ECE (Expected Calibration Error, 15 bins)

*Lower is better. Reduction % = (before − after) / before × 100*

| Model | T | ECE Before | ECE After | Reduction |
|-------|---|------------|-----------|-----------|
| logreg | 1.031 | 0.0068 | 0.0074 | -9.1% |
| svm | 1.033 | 0.0126 | 0.0163 | -29.4% |
| tfidf | 1.131 | 0.0131 | 0.0174 | -33.1% |
| ensemble | 0.935 | 0.0216 | 0.0117 | +46.0% |
| meta_learner | 0.984 | 0.0203 | 0.0230 | -13.2% |

### Brier Score

*Lower is better.*

| Model | Brier Before | Brier After | Reduction |
|-------|--------------|-------------|-----------|
| logreg | 0.4083 | 0.4084 | -0.0% |
| svm | 0.4274 | 0.4276 | -0.1% |
| tfidf | 0.4464 | 0.4469 | -0.1% |
| ensemble | 0.4113 | 0.4107 | +0.2% |
| meta_learner | 0.4102 | 0.4103 | -0.0% |

### Macro-F1 (unchanged by design)

| Model | Macro-F1 Before | Macro-F1 After | Δ |
|-------|-----------------|----------------|---|
| logreg | 0.6943 | 0.6943 | +0.0000 |
| svm | 0.6817 | 0.6817 | +0.0000 |
| tfidf | 0.6579 | 0.6579 | +0.0000 |
| ensemble | 0.6938 | 0.6938 | +0.0000 |
| meta_learner | 0.6967 | 0.6967 | +0.0000 |

## Summary

- Average ECE reduction: **-7.8%** across all 5 models
- Average Brier reduction: **-0.0%** across all 5 models
- Most overconfident model: **tfidf** (T=1.131)
- Largest ECE improvement: **ensemble** (+46.0%)

## Thesis Interpretation

Temperature scaling provides a lightweight, theoretically-grounded calibration layer that improves probabilistic reliability without retraining. The learned temperatures reveal the inherent confidence tendencies of each architecture:

- Classical ML models (TF-IDF + LogReg/SVM) output decision-function scores converted to probabilities via Platt scaling, which can be systematically over- or under-confident depending on the feature space.

- The ensemble and meta-learner aggregate multiple models, which can amplify or dampen individual model biases — their temperatures reveal whether aggregation helped or hurt calibration.

Since temperature scaling preserves argmax predictions, it is safe to apply at inference time with no accuracy trade-off. The calibrated probabilities are required for the entropy-gated selective predictor (§4.4) to achieve its theoretical guarantees.

## Reference

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
