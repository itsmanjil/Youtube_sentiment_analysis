# Should-do Deliverables — Summary

Delivery date: 2026-04-11

This file rolls up the six thesis-grade "should-do" items that were agreed after
the Must-do phase closed. For each item it states the question, the artefacts
produced, and the one-line finding relevant to the thesis defense.

---

## 1. Fold 4 "anomaly" investigation

**Question.** The existing `thesis_evaluation_report.json` showed a 10-fold CV
run where fold 4 dropped to 60% accuracy while other folds hit 74%. An examiner
would reasonably ask why.

**Finding.** *The anomaly was an artefact of the reporting script itself.* The
only producer of `thesis_evaluation_report.json` was the `__main__` demo block
in `research/evaluation_framework.py`, which called
`sklearn.datasets.make_classification(n_samples=1000, …)` — synthetic data, not
YouTube comments. Fold size was ~100 samples, so a single bad draw could swing
accuracy 14 pp.

**Actions taken.**
- Renamed the synthetic file to `DEMO_synthetic_evaluation_report.json` so it
  can never be mistaken for a real result.
- Built `research/fold_variance_analysis.py`, which runs a 10-fold stratified
  CV on a 30k balanced subsample of the real `train.csv` with the same
  TF-IDF(1,2) + LogReg pipeline.
- Output: `results/fold_variance_analysis.{json,md}`.

**Real-data result.** All 10 folds land in F1 ∈ [0.6326, 0.6667] — a spread of
just 0.034. The worst fold (#4) has class skew of exactly 0.0 against the
global distribution (stratification holds), vocab overlap 0.797 vs the best
fold's 0.794, and token length within one token of the mean. The original
"anomaly" does **not** replicate at realistic sample sizes.

---

## 2. Effect-size reporting (Cohen's g, risk difference, odds ratio)

**Question.** `ci_significance_tests.md` reports Holm-Bonferroni-corrected
McNemar p-values, but n_test = 20,000 makes p-values almost automatic. Modern
reporting standards expect effect sizes alongside p-values.

**Deliverables.**
- `research/effect_sizes.py` — computes Cohen's g, odds ratio, and
  pp-accuracy risk difference for every method pair.
- `results/effect_sizes.{json,md}`.

**Finding.** Every CI-vs-LogReg comparison lands at Cohen's g < 0.04, which is
below Cohen's (1988) threshold of 0.05 for "negligible":

| Comparison | ΔF1 | Acc gain (pp) | Cohen's g | Magnitude |
|---|---:|---:|---:|:---:|
| logreg → meta_learner | +0.0025 | +0.16 | +0.028 | negligible |
| logreg → ts_meta      | +0.0025 | +0.16 | +0.028 | negligible |
| logreg → neuro_fuzzy  | +0.0013 | +0.14 | +0.035 | negligible |
| logreg → pso          | +0.0009 | +0.10 | +0.029 | negligible |
| logreg → nsga2        | +0.0006 | +0.07 | +0.030 | negligible |
| logreg → ensemble     | −0.0005 | −0.01 | +0.001 | negligible |

This quantitatively confirms the reframed thesis narrative: defend CI on
**calibration**, **Pareto trade-offs**, and the **negative result**, not on
F1 gains.

---

## 3. Error characterization on the real test set

**Question.** What kinds of comments does the LogReg baseline actually fail
on? Is there a concrete story for the Limitations section?

**Deliverables.**
- `research/error_characterization.py` — loads `models/logreg/model.sav`,
  runs it on all 165,110 test rows, and slices the errors by text property.
- `results/error_characterization.{json,md}`.

**Findings (overall accuracy = 0.6946).**
1. **Neutral is the weakest class** (accuracy 0.6232 vs Negative 0.7242,
   Positive 0.7283).
2. **Neutral↔Negative confusion makes up 47.3% of all errors**
   (Neutral→Negative 24.0% + Negative→Neutral 23.3%).
3. **Longer comments are monotonically harder**: very-short 0.7335 → long
   0.6717. Counter to the naive assumption that "more text = more signal".
4. **Negation penalty: −5.29 pp** accuracy on comments with negation markers
   (0.6567 vs 0.7096) — directly validates keeping negators out of the
   stopword list in `src/preprocessing/classical.py`.
5. **Confidence gap = 0.1503** between correct and wrong predictions. The
   baseline is only modestly more confident when right than when wrong —
   exactly the calibration weakness the neuro-fuzzy gate targets.

---

## 4. Preprocessing knob ablation

**Question.** The existing `thesis_preprocess_ablation.md` ablates *dataset-level*
cleaning stages (raw / youtube_clean / youtube_filtered). It does not isolate
the knobs inside `ClassicalPreprocessConfig`.

**Deliverables.**
- `research/preprocessing_knob_ablation.py` — 2³ ablation over the three knobs
  (expand negation contractions, negation tag, remove stopwords) on an 18k
  balanced subsample with 80/20 train/val.
- `results/preprocessing_knob_ablation.{json,md}`.

**Finding.** All eight configurations land in F1 ∈ [0.6264, 0.6367] — a spread
of 0.0103 and a best-vs-baseline delta of only +0.0018. *The knobs are
behaviour-preserving, not accuracy-enhancing.* Thesis-safe framing: the
preprocessing module stabilises preprocessing between training and inference
(no train/inference skew — see `preprocessing_consistency_audit.md`) at
essentially zero F1 cost, which is the honest contribution.

Note: the `negation_tag` knob has a slightly negative main effect when
enabled alone and should be defended as a *design choice for interpretability*,
not as a performance optimisation.

---

## 5. Seed sensitivity analysis

**Question.** Is the 0.694 macro-F1 headline a lucky draw, or does it hold
across random seeds?

**Deliverables.**
- `research/seed_sensitivity.py` — re-samples, re-splits, and re-seeds the
  baseline pipeline for five seeds (0, 7, 13, 42, 123) with 24k balanced
  samples each.
- `results/seed_sensitivity.{json,md}`.

**Finding.** F1 across seeds = 0.6478 ± 0.0045, range = 0.0117
([0.6413, 0.6530]). The range is small in absolute terms **but it is larger
than every single CI-vs-LogReg delta from Should-do #2** (all under 0.003).
This is a strong additional data point for the negative-result narrative:
the seed wobble of the baseline alone *exceeds* the method-choice improvement
from any CI layer.

---

## 6. Pareto front visualisation

**Question.** The NSGA-II Pareto front in `multi_objective_ensemble.json` is
the most defensible CI contribution (it shows real trade-offs). It needed a
thesis-ready figure.

**Deliverables.**
- `research/pareto_visualization.py` — reads the existing JSON and renders
  three standalone SVG scatter plots plus a combined HTML viewer. No
  matplotlib dependency, so the SVGs can be embedded directly in LaTeX via
  `\includegraphics`.
- `results/pareto_f1_vs_ece.svg`
- `results/pareto_f1_vs_coverage.svg`
- `results/pareto_ece_vs_coverage.svg`
- `results/pareto_visualization.html`  (combined viewer, knee point highlighted)

**Knee point.** Index 26 of the 60-point Pareto front. Weights: logreg 0.916,
svm 0.003, tfidf 0.081. Val F1 = 0.6903, ECE = 0.0122, coverage = 0.4679.
Test F1 = 0.6940, ECE = 0.0039. This is the recommended headline configuration
for the thesis.

---

## Cross-item synthesis (for the thesis narrative)

Three independent measurements now converge on the same conclusion:

1. **Effect size (Should-do #2)** — all CI methods are Cohen's-g-negligible
   against LogReg on F1 (g < 0.04 vs the 0.05 "negligible" threshold).
2. **Seed variance (Should-do #5)** — the F1 range of LogReg across seeds
   (0.012) is ~4× larger than the largest CI-vs-LogReg F1 gain (0.003).
3. **Pareto front (Should-do #6)** — the CI methods that *do* win (knee test
   ECE 0.004 vs LogReg ECE much higher, see `neuro_fuzzy_gate.json`) win on
   calibration, not accuracy.

Combined with the preprocessing audit and ANFIS-wording fix from the Must-do
phase, the thesis now has a coherent and defensible story: *the CI layer
delivers calibration and operational trade-offs, not higher F1, and this is
a positive contribution in a scientific sense — a negative result that
rules out several fashionable CI techniques as F1 improvements on this
dataset.*
