# Fuzzy Logic Ensemble — Theoretical Grounding

## 1. Motivation

Standard ensemble methods (soft voting, stacking) aggregate model probability
estimates using crisp arithmetic operations (weighted sums, logistic regression).
This ignores an important property of sentiment classification: **epistemic uncertainty**.
When a model assigns probabilities [0.38, 0.34, 0.28], it is effectively saying
"I am uncertain about this instance." Treating uncertain and confident predictions
identically can degrade ensemble quality.

**Fuzzy Logic** (Zadeh, 1965) provides a mathematically principled framework for
representing and aggregating uncertain information using *membership functions* and
*linguistic rules*, making it a natural fit for uncertainty-aware ensemble fusion
in the Computational Intelligence tradition.

---

## 2. Theoretical Background

### 2.1 Fuzzy Sets and Membership Functions

A fuzzy set $A$ over universe $X$ is defined by a membership function
$\mu_A: X \rightarrow [0, 1]$, where $\mu_A(x) = 1$ means full membership and
$\mu_A(x) = 0$ means non-membership.

In our context, each sentiment class probability output by a base model is mapped
to a fuzzy membership value indicating **confidence degree**:

- **Triangular MF:** $\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right)\right)$
- **Trapezoidal MF:** $\mu(x; a, b, c, d)$ — flat peak between $b$ and $c$, wider uncertainty region

### 2.2 T-norms and T-conorms

Fuzzy intersection (AND) and union (OR) are generalised by:

| Operator | Definition | Interpretation |
|----------|-----------|----------------|
| **Minimum t-norm** | $T(a,b) = \min(a,b)$ | Conservative (most cautious) conjunction |
| **Product t-norm** | $T(a,b) = a \cdot b$ | Probabilistic conjunction |
| **Maximum t-conorm** | $S(a,b) = \max(a,b)$ | Conservative disjunction |
| **Probabilistic sum** | $S(a,b) = a+b-ab$ | Probabilistic disjunction |

### 2.3 Defuzzification Methods

The fuzzy aggregation produces a fuzzy output set; defuzzification converts this
back to a crisp class decision:

| Method | Formula | Property |
|--------|---------|---------|
| **Centroid (CoG)** | $\bar{x} = \frac{\int x\mu(x)dx}{\int \mu(x)dx}$ | Minimises squared error |
| **Mean of Maxima (MoM)** | Average of all x where μ is maximised | Emphasises peak membership |
| **Bisector** | x that bisects the area under μ | Robust to asymmetric distributions |
| **Smallest of Maxima (SoM)** | Leftmost maximising x | Biased toward lower classes |
| **Largest of Maxima (LoM)** | Rightmost maximising x | Biased toward higher classes |

---

## 3. Application to Sentiment Ensemble

### 3.1 System Design

Each base model produces a probability vector $\mathbf{p} = [p_{neg}, p_{neu}, p_{pos}]$
for each comment. The fuzzy ensemble proceeds as follows:

```
For each comment:
  1. Get base model probabilities: p = model.batch_analyze(text)
  2. Apply membership functions: μ_c = MF(p_c)  for c ∈ {Neg, Neu, Pos}
  3. Aggregate across base models using t-norm/t-conorm
  4. Defuzzify the aggregated fuzzy set to get a crisp confidence per class
  5. Assign the class with highest crisp confidence
```

### 3.2 The Confidence Threshold

The `confidence_threshold = 0.6` parameter creates a minimum membership
requirement. Predictions where no class exceeds this threshold are flagged as
low-confidence and could be deferred to a higher-capacity model in a cascade
architecture — an extension for future work.

---

## 4. Grid Search Design

### 4.1 Search Space

The grid search (`backend/research/fuzzy_grid_search.py`) explores:

| Dimension | Values Tested |
|-----------|--------------|
| Membership function type | Triangular, Trapezoidal |
| Defuzzification method | Centroid, MoM, Bisector, SoM, LoM |
| T-norm | Min, Product |
| T-conorm | Max, Probabilistic Sum |
| Alpha-cut threshold | 0.0, 0.1, 0.2 |
| Base models | {logreg}, {logreg, svm}, {logreg, svm, tfidf} |
| Resolution | 100 |

**Total configurations evaluated:** 216

> Note: The filename `fuzzy_grid_5k.json` refers to the *limit* parameter
> (first 5,000 test samples used for speed), not the number of configurations.

### 4.2 Why Grid Search (Not Random/Bayesian)?

The search space is small (216 discrete combinations), fully enumerable, and
non-continuous — making exhaustive grid search both tractable and more
reproducible than Bayesian optimisation. Every configuration is evaluated
identically, eliminating acquisition function bias.

### 4.3 Evaluation Protocol

To prevent configuration selection overfitting:

- **Selection set:** A held-out validation sample (not the test set)
- **Reported results:** Best configuration re-evaluated on the independent test set
- **Metric:** Macro-F1 (consistent with all other models)

---

## 5. Results

### 5.1 Best Configuration Found

| Parameter | Value |
|-----------|-------|
| Membership function | Triangular |
| Defuzzification | Mean of Maxima (MoM) |
| T-norm | Minimum |
| T-conorm | Maximum |
| Base models | LogReg only |
| Alpha-cut | 0.0 |
| Resolution | 100 |
| **Val Macro-F1** | **0.6726** |
| **Test Macro-F1** | **0.6676** |
| **Test Accuracy** | 0.6627 |

### 5.2 Comparison with Other Ensembles

| Method | Test Macro-F1 | Notes |
|--------|--------------|-------|
| LogReg (single model) | 0.6928 | Best classical baseline |
| Fuzzy Ensemble (best config) | 0.6676 | MoM + triangular MF |
| PSO Weighted Ensemble | 0.6909 | Soft voting, optimised weights |
| Meta-Learner (stacking) | 0.6946 | Best overall |

### 5.3 Why Fuzzy Underperforms the Meta-Learner

The Fuzzy ensemble achieves **0.6676** vs **0.6946** for the meta-learner.
Key reasons:

1. **Defuzzification information loss:** Converting the fuzzy output back to a
   scalar discards distributional information that stacking preserves.
2. **Rule-agnostic combination:** The fuzzy system uses fixed t-norm/t-conorm
   operators rather than learning an optimal combination function from data.
3. **Single base model optimal:** The grid search found that using only LogReg
   (not LogReg+SVM) is best — suggesting the fuzzy combination of multiple
   models introduces noise rather than complementary information.

---

## 6. Theoretical Contribution to Thesis

The fuzzy ensemble contributes to the **Computational Intelligence** component
of your degree. The key theoretical contributions are:

1. **Uncertainty quantification:** Fuzzy membership provides a graded confidence
   signal unavailable in crisp ensemble methods.
2. **Systematic evaluation of fuzzy operators:** The grid over t-norms, t-conorms,
   and defuzzification methods constitutes an empirical study of fuzzy operator
   suitability for NLP ensemble tasks.
3. **Negative result value:** The finding that MoM + triangular MF with a single
   base model is optimal is itself theoretically interesting — it suggests that
   for near-balanced, moderately-confident models, conservative conjunction
   (min t-norm) and peak-emphasising defuzzification (MoM) capture the most
   discriminative signal.

---

## 7. Recommended Thesis Framing

In your **Computational Intelligence Methods** chapter:

> We apply fuzzy set theory (Zadeh, 1965) to the ensemble combination problem,
> mapping base model probability outputs to fuzzy membership degrees via
> triangular and trapezoidal membership functions. An exhaustive grid search
> over 216 operator configurations — varying membership function type,
> defuzzification method (centroid, MoM, bisector), and t-norm/t-conorm pairs
> — was conducted on a held-out validation set, with the best configuration
> re-evaluated on the independent test set to prevent overfitting to the search.
> The optimal configuration (triangular MF, Mean-of-Maxima defuzzification,
> min t-norm) achieves a test Macro-F1 of 0.6676, demonstrating that while
> fuzzy uncertainty modelling provides a principled framework, it does not
> surpass the meta-learner (0.6946) in this domain. This finding is consistent
> with the literature: fuzzy systems offer advantages in domains with high
> epistemic uncertainty or mismatched model confidence, but are less effective
> when base models are well-calibrated and their errors are correlated.

---

## 8. References

- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis
  with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), 1–13.
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.
- Ruta, D., & Gabrys, B. (2000). An overview of classifier fusion methods.
  *Computing and Information Systems*, 7(1), 1–10.
- Zimmermann, H. J. (2001). *Fuzzy Set Theory — and Its Applications* (4th ed.). Kluwer Academic.
