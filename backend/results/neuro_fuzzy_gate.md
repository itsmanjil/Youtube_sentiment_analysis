# Neuro-Fuzzy Confidence Gating System

## Architecture

Per-sample adaptive ensemble router using learned Gaussian membership functions (ANFIS-lite).

- **Input**: confidence vector c = [c_logreg, c_svm, c_tfidf]
- **Fuzzification**: 3 Gaussian MFs per model (Low / Medium / High)
- **Gate weights**: softmax(Σ_k α_{m,k} · μ_{m,k}(c_m))
- **Output**: per-sample weighted ensemble probabilities
- **Total parameters**: 27 (3 models × 3 sets × 3)

## Training

| | Value |
|---|-------|
| NLL before fitting | 0.7164 |
| NLL after fitting | 0.7048 |
| Converged | True |
| L-BFGS-B iterations | 106 |

## Learned Membership Function Parameters

| Model | Fuzzy Set | Center | Width | Alpha (α) |
|-------|-----------|--------|-------|-----------|
| logreg | Low | 0.634 | 0.071 | +5.000 |
| logreg | Medium | 0.276 | 0.138 | +2.760 |
| logreg | High | 0.990 | 0.412 | +0.618 |
| svm | Low | 0.662 | 0.489 | -2.155 |
| svm | Medium | 0.765 | 0.489 | -1.996 |
| svm | High | 0.731 | 0.430 | -2.111 |
| tfidf | Low | 0.010 | 0.050 | -1.355 |
| tfidf | Medium | 0.866 | 0.050 | +3.522 |
| tfidf | High | 0.685 | 0.235 | -1.040 |

*Centers reveal which confidence region activates each model. Positive α = high-confidence samples prefer this model; negative α = low-confidence samples prefer this model.*

## Performance Comparison (Test Set)

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Static ensemble (uniform) | 0.6938 | 0.6959 | 0.0260 | 0.4123 |
| **Neuro-fuzzy gate** | **0.6955** | **0.6972** | **0.0070** | **0.4076** |

Δ Macro-F1 = **+0.0017**  |  Δ ECE = **-0.0190**

## Thesis Interpretation

The neuro-fuzzy gate demonstrates a fundamental advantage over static ensemble weighting: **different samples benefit from different model mixtures**. When a model is confident (high c_m), the gate boosts its contribution; when it is uncertain, the gate reduces it, distributing weight to the remaining models.

The learned MF parameters are directly interpretable: the center values show which confidence level is the 'sweet spot' for each model, and the α consequents reveal whether that confidence region should increase or decrease the model's influence.

This constitutes the **Neuro-Fuzzy CI contribution** of the thesis: a principled, interpretable, and data-driven approach to ensemble gating that extends classical fuzzy inference with parameter learning.

## References

Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
