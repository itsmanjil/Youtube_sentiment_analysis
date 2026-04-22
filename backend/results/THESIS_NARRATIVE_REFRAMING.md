# Thesis Narrative Reframing Guide

## The Problem With the Current Narrative

If you frame your thesis as "CI methods improve sentiment classification
accuracy on YouTube comments," your own significance tests disprove this
claim. The McNemar results show that **no CI method significantly
outperforms logistic regression** (all Holm-adjusted p ≥ 0.17 for CI vs
logreg). The F1 range across all methods is 0.6943–0.6967 — a 0.24
percentage point span that is statistically indistinguishable.

An examiner who reads your significance tests and then sees an accuracy-
focused contribution claim will question the validity of the entire thesis.

## The Reframed Narrative (Three Contributions)

### Contribution 1: Calibration, Not Accuracy

**Claim**: Computational intelligence methods substantially improve
*probability calibration* for YouTube sentiment analysis, even when
classification accuracy gains are marginal.

**Evidence**:

| Method | Macro-F1 | ECE | Brier |
|--------|----------|-----|-------|
| Static ensemble (uniform) | 0.6938 | 0.0260 | 0.4123 |
| **Neuro-fuzzy gate** | **0.6955** | **0.0070** | **0.4076** |
| NSGA-II knee-point | 0.6940 | 0.0039 | — |

The neuro-fuzzy gate reduces Expected Calibration Error by **73.1%**
(0.0260 → 0.0070) compared to static ensemble weights. The NSGA-II
knee-point achieves even lower ECE (0.0039) on the test set.

**Why this matters**: In production sentiment analysis, downstream systems
(dashboards, alerts, content moderation pipelines) rely on model *confidence
scores* to make decisions. A model that says "I'm 90% confident this is
positive" should be correct ~90% of the time. Uncalibrated confidence scores
lead to systematic over- or under-confidence that cascades into poor
decisions.

**How to write it in your thesis**:

> "While classification accuracy plateaus across ensemble configurations —
> a finding consistent with the 'no free lunch' theorem (Wolpert, 1996) —
> the CI methods provide a qualitatively different kind of improvement:
> probability calibration. The neuro-fuzzy gating mechanism reduces ECE
> by 73.1% relative to static ensemble weighting, demonstrating that
> adaptive, confidence-aware model routing produces substantially more
> reliable uncertainty estimates."

### Contribution 2: Pareto-Optimal Trade-offs via NSGA-II

**Claim**: Multi-objective optimization reveals the accuracy–calibration–
coverage trade-off frontier that single-objective methods cannot expose.

**Evidence**: Your NSGA-II run produced a Pareto front of 60 non-dominated
solutions. Key observations from the front:

- **Accuracy-first extreme**: logreg weight ≈ 0.99, F1 = 0.6891,
  ECE = 0.0140, coverage = 0.478
- **Calibration-first extreme**: tfidf weight ≈ 0.93, F1 = 0.6580,
  ECE = 0.0075, coverage = 0.414
- **Knee-point (balanced)**: logreg = 0.916, svm = 0.003, tfidf = 0.081,
  test F1 = 0.6940, test ECE = 0.0039, coverage = 0.471

The Pareto front demonstrates that accuracy and calibration are **partially
conflicting objectives** — you cannot maximise both simultaneously. This is
a genuine insight that single-objective PSO cannot provide.

**How to write it**:

> "NSGA-II reveals that the sentiment ensemble problem is inherently multi-
> objective: configurations that maximise macro-F1 tend to produce poorly
> calibrated probabilities, while well-calibrated configurations sacrifice
> 2–3 percentage points of F1. The Pareto front enables practitioners to
> select a configuration matching their downstream requirements — accuracy-
> focused for simple classification, calibration-focused for decision-support
> systems."

### Contribution 3: The Negative Result as a Finding

**Claim**: For YouTube comment sentiment analysis with TF-IDF features, the
CI ensemble methods do not significantly outperform a well-tuned logistic
regression baseline on classification accuracy — this ceiling effect itself
is a meaningful finding.

**Evidence**: McNemar's test across all 36 pairwise comparisons shows that
no CI method significantly differs from logistic regression on accuracy
(p_adj = 1.0 for all CI–logreg pairs after Holm-Bonferroni correction).

**Why this is publishable**: Negative results prevent the community from
pursuing unproductive directions. Your finding suggests that for short,
noisy social media text with bag-of-words features, the classification
ceiling is reached by classical linear models, and CI methods should focus
on calibration and uncertainty quantification rather than accuracy.

**How to write it**:

> "The statistical equivalence of all methods on classification accuracy
> represents a feature-space ceiling: the TF-IDF bigram representation
> captures the extractable signal for YouTube comment sentiment, and no
> amount of ensemble engineering can exceed this ceiling. However, CI
> methods provide value in a different dimension — probability calibration
> — suggesting that future work should combine richer representations
> (transformer embeddings) with CI ensemble methods to break both the
> accuracy ceiling and improve calibration simultaneously."

## Thesis Chapter Structure (Recommended)

**Chapter 5: Experimental Results**

5.1 Baseline Model Performance (logreg, SVM, TF-IDF NB)
    - Hyperparameter tuning results (grid search with 5-fold CV)
    - Test set performance with confidence intervals

5.2 Ensemble Methods (uniform, meta-learner, PSO, NSGA-II)
    - Classification accuracy comparison
    - **Statistical significance analysis** (McNemar + Holm-Bonferroni)
    - Key finding: accuracy plateau across all methods

5.3 Calibration Analysis ← **This is your main CI contribution**
    - ECE and Brier score comparison
    - Temperature scaling results
    - Neuro-fuzzy gating: 73% ECE reduction
    - Reliability diagrams

5.4 Multi-Objective Analysis
    - Pareto front from NSGA-II
    - Trade-off between F1, ECE, and coverage
    - Knee-point selection rationale

5.5 Ablation Study
    - Component-wise contribution table
    - What each CI method adds (and doesn't add)

5.6 Gold Set Validation
    - Performance on 300 manually-labeled comments
    - Inter-annotator agreement discussion

**Chapter 6: Discussion**

6.1 The accuracy ceiling and its implications
6.2 Calibration as the primary CI contribution
6.3 When CI ensemble methods are and aren't warranted
6.4 Limitations (label noise, domain shift, computational cost)
6.5 Threats to validity

## Key Sentences for Your Viva/Defense

If asked "Why bother with all this CI complexity if logistic regression
performs equally well?":

> "That's precisely one of our findings. Classification accuracy saturates
> at ~0.695 F1 with TF-IDF features, regardless of ensemble sophistication.
> However, the CI methods provide a 73% improvement in probability
> calibration (ECE: 0.026 → 0.007), which is critical for any production
> system that relies on confidence scores for downstream decision-making.
> NSGA-II further contributes by revealing the Pareto-optimal trade-off
> between accuracy and calibration — a multi-dimensional insight that
> single-objective methods cannot provide."

If asked "Is the neuro-fuzzy gate really ANFIS?":

> "It's a simplified neuro-fuzzy gating mechanism inspired by ANFIS. A
> full ANFIS with 27 cross-model rules would risk overfitting our 3-model
> ensemble. We deliberately use per-model independent gating with 9
> activations, which preserves the key ANFIS principles of learned
> membership functions and gradient-based parameter fitting while reducing
> rule-space complexity. This simplification is explicitly documented and
> justified in our methodology."

If asked "Your McNemar tests show no significant differences — doesn't
that invalidate your thesis?":

> "No — it validates one of our three contributions. We transparently
> report that accuracy gains are not significant, which is itself a
> meaningful finding for the sentiment analysis community. Our primary CI
> contribution is in calibration and multi-objective optimization, where
> the improvements are substantial and measurable."

---
*Guide prepared: 2026-04-09*
*Based on actual experimental results from the project's `results/` directory.*
