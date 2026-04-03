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
| NLL before fitting | 0.6358 |
| NLL after fitting | 0.5517 |
| Converged | False |
| L-BFGS-B iterations | 200 |

## Learned Membership Function Parameters

| Model | Fuzzy Set | Center | Width | Alpha (α) |
|-------|-----------|--------|-------|-----------|
| deberta_v3 | Low | 0.108 | 0.101 | -5.000 |
| deberta_v3 | Medium | 0.314 | 0.050 | -4.678 |
| deberta_v3 | High | 0.424 | 0.050 | +5.000 |
| logreg | Low | 0.655 | 0.072 | +5.000 |
| logreg | Medium | 0.716 | 0.238 | +3.825 |
| logreg | High | 0.444 | 0.050 | +5.000 |
| svm | Low | 0.335 | 0.064 | +5.000 |
| svm | Medium | 0.115 | 0.235 | +3.735 |
| svm | High | 0.130 | 0.162 | -0.521 |

*Centers reveal which confidence region activates each model. Positive α = high-confidence samples prefer this model; negative α = low-confidence samples prefer this model.*

## Performance Comparison (Test Set)

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Static ensemble (uniform) | 0.7770 | 0.7778 | 0.1385 | 0.3375 |
| **Neuro-fuzzy gate** | **0.7976** | **0.8000** | **0.0853** | **0.3048** |

Δ Macro-F1 = **+0.0206**  |  Δ ECE = **-0.0532**

## Thesis Interpretation

The neuro-fuzzy gate demonstrates a fundamental advantage over static ensemble weighting: **different samples benefit from different model mixtures**. When a model is confident (high c_m), the gate boosts its contribution; when it is uncertain, the gate reduces it, distributing weight to the remaining models.

The learned MF parameters are directly interpretable: the center values show which confidence level is the 'sweet spot' for each model, and the α consequents reveal whether that confidence region should increase or decrease the model's influence.

This constitutes the **Neuro-Fuzzy CI contribution** of the thesis: a principled, interpretable, and data-driven approach to ensemble gating that extends classical fuzzy inference with parameter learning.

## References

Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
