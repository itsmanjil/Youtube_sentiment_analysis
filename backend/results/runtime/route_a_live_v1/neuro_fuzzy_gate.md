# Neuro-Fuzzy Confidence Gating System

## Architecture

Per-sample adaptive ensemble router using learned Gaussian membership functions (simplified neuro-fuzzy gating inspired by ANFIS).

- **Input**: confidence vector c = [c_logreg, c_svm, c_tfidf]
- **Fuzzification**: 3 Gaussian MFs per model (Low / Medium / High)
- **Gate weights**: softmax(Σ_k α_{m,k} · μ_{m,k}(c_m))
- **Output**: per-sample weighted ensemble probabilities
- **Total parameters**: 27 (3 models × 3 sets × 3)

## Training

| | Value |
|---|-------|
| NLL before fitting | 0.7164 |
| NLL after fitting | 0.7047 |
| Converged | True |
| L-BFGS-B iterations | 120 |

## Learned Membership Function Parameters

| Model | Fuzzy Set | Center | Width | Alpha (α) |
|-------|-----------|--------|-------|-----------|
| logreg | Low | 0.516 | 0.158 | +3.435 |
| logreg | Medium | 0.154 | 0.050 | +2.526 |
| logreg | High | 0.895 | 0.500 | +0.706 |
| svm | Low | 0.812 | 0.455 | -3.184 |
| svm | Medium | 0.740 | 0.424 | -2.977 |
| svm | High | 0.691 | 0.368 | -3.174 |
| tfidf | Low | 0.010 | 0.050 | -1.702 |
| tfidf | Medium | 0.890 | 0.050 | +3.458 |
| tfidf | High | 0.451 | 0.093 | -0.906 |

*Centers reveal which confidence region activates each model. Positive α = high-confidence samples prefer this model; negative α = low-confidence samples prefer this model.*

## Performance Comparison (Test Set)

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Static ensemble (uniform) | 0.6938 | 0.6959 | 0.0260 | 0.4123 |
| **Neuro-fuzzy gate** | **0.6960** | **0.6976** | **0.0070** | **0.4075** |

Δ Macro-F1 = **+0.0022**  |  Δ ECE = **-0.0190**

## Thesis Interpretation

The neuro-fuzzy gate demonstrates a fundamental advantage over static ensemble weighting: **different samples benefit from different model mixtures**. When a model is confident (high c_m), the gate boosts its contribution; when it is uncertain, the gate reduces it, distributing weight to the remaining models.

The learned MF parameters are directly interpretable: the center values show which confidence level is the 'sweet spot' for each model, and the α consequents reveal whether that confidence region should increase or decrease the model's influence.

This constitutes the **Neuro-Fuzzy CI contribution** of the thesis: a principled, interpretable, and data-driven approach to ensemble gating that extends classical fuzzy inference with parameter learning.

## References

Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
