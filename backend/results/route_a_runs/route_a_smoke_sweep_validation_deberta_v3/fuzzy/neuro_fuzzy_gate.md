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
| NLL before fitting | 0.7009 |
| NLL after fitting | 0.5884 |
| Converged | True |
| L-BFGS-B iterations | 106 |

## Learned Membership Function Parameters

| Model | Fuzzy Set | Center | Width | Alpha (α) |
|-------|-----------|--------|-------|-----------|
| deberta_v3 | Low | 0.183 | 0.050 | -5.000 |
| deberta_v3 | Medium | 0.404 | 0.092 | -5.000 |
| deberta_v3 | High | 0.198 | 0.054 | -4.144 |
| logreg | Low | 0.277 | 0.124 | -5.000 |
| logreg | Medium | 0.727 | 0.177 | +5.000 |
| logreg | High | 0.502 | 0.052 | +5.000 |
| svm | Low | 0.105 | 0.050 | +3.688 |
| svm | Medium | 0.107 | 0.050 | +2.516 |
| svm | High | 0.990 | 0.178 | +0.158 |

*Centers reveal which confidence region activates each model. Positive α = high-confidence samples prefer this model; negative α = low-confidence samples prefer this model.*

## Performance Comparison (Test Set)

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Static ensemble (uniform) | 0.7625 | 0.7667 | 0.1566 | 0.3693 |
| **Neuro-fuzzy gate** | **0.7811** | **0.7833** | **0.1499** | **0.3484** |

Δ Macro-F1 = **+0.0186**  |  Δ ECE = **-0.0067**

## Thesis Interpretation

The neuro-fuzzy gate demonstrates a fundamental advantage over static ensemble weighting: **different samples benefit from different model mixtures**. When a model is confident (high c_m), the gate boosts its contribution; when it is uncertain, the gate reduces it, distributing weight to the remaining models.

The learned MF parameters are directly interpretable: the center values show which confidence level is the 'sweet spot' for each model, and the α consequents reveal whether that confidence region should increase or decrease the model's influence.

This constitutes the **Neuro-Fuzzy CI contribution** of the thesis: a principled, interpretable, and data-driven approach to ensemble gating that extends classical fuzzy inference with parameter learning.

## References

Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
