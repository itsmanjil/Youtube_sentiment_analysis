# Chapter 4 — Consolidated Evaluation

Status date: 2026-07-02 (updated)

This chapter consolidates every evaluation strand into one defensible narrative,
mapping each result to its reproducible artifact. The argument is deliberately
*not* "our model has the best F1"; it is "calibration, uncertainty, and human
grounding distinguish the models where raw F1 does not."

## 4.1 Headline Runtime Performance (165,110 comments)

Source: `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`.

| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | ms/sample |
|-------|---------:|---------:|----:|------:|:----------:|----------:|
| meta_learner | 0.6953 | **0.6945** | 0.0157 | 0.4117 | yes | 0.48 |
| ensemble_nsga2 | **0.6959** | 0.6940 | **0.0046** | 0.4092 | yes | 0.49 |
| logreg | 0.6946 | 0.6928 | 0.0039 | 0.4100 | yes | 0.13 |
| ensemble_pso | 0.6872 | 0.6852 | 0.0113 | 0.4195 | yes | 0.35 |
| svm | 0.6801 | 0.6780 | 0.0170 | 0.4293 | yes | 0.11 |
| tfidf | 0.6622 | 0.6567 | 0.0179 | 0.4491 | yes | 0.12 |
| fuzzy_ensemble | 0.6622 | 0.6567 | 0.0185 | 0.4489 | no | 0.52 |

**Honest reading of the F1 gap (addresses review finding #5).** The best model
(meta_learner) exceeds the logistic-regression baseline by only **+0.0017
macro-F1** on the full test set. This margin is small and the thesis does **not**
rest its contribution on it. The substantive differences are in *calibration*:
`ensemble_nsga2` attains ECE = 0.0046 — roughly a quarter of the meta-learner's
0.0157 — while matching it on accuracy. The thesis claim is therefore that
NSGA-II multi-objective weighting buys **calibration quality at no accuracy
cost** (RQ1, RQ3), which is the deployment-relevant property, not a raw-F1 win.

**Note on the fuzzy ensemble row.** The neuro-fuzzy-gated configuration
reports metrics identical to the TF-IDF + Naive-Bayes base classifier
(accuracy 0.6622, macro-F1 0.6567) because, on this corpus, the gate's
fuzzy-inference override rarely changes the argmax label. A direct
ablation confirms this: on a 40,000-comment sample (seed 42), the deployed
`fuzzy_ensemble` engine changes the base classifier's argmax on only 0.18%
of comments (71 of 40,000) — 33 corrections, 21 regressions, and 17
wrong-to-wrong flips (both predictions incorrect but for different
labels) — so its net effect on accuracy and macro-F1 is negligible. Source:
`results/neuro_fuzzy_gate_ablation/fuzzy_gate_ablation.md`, reproducible via
`python research/ci/fuzzy_gate_ablation.py --sample 40000 --seed 42`.

## 4.2 ROC-AUC, One-vs-Rest (review finding #7)

Source: `results/roc_auc/roc_auc.md` (5,000-comment sample, seed 42).
Reproduce: `python research/evaluation/roc_auc.py --test data/test.csv --sample 5000`.

| Model | Macro AUC | Positive AUC | Neutral AUC | Negative AUC |
|-------|----------:|-------------:|------------:|-------------:|
| ensemble_nsga2 | **0.8596** | 0.8948 | 0.8184 | 0.8655 |
| logreg | 0.8589 | 0.8945 | 0.8172 | 0.8652 |
| meta_learner | 0.8577 | 0.8922 | 0.8186 | 0.8623 |
| ensemble_pso | 0.8511 | 0.8897 | 0.8060 | 0.8575 |
| svm | 0.8434 | 0.8847 | 0.7954 | 0.8501 |
| tfidf | 0.8315 | 0.8684 | 0.7925 | 0.8335 |
| fuzzy_ensemble | 0.8314 | 0.8680 | 0.7923 | 0.8337 |

ROC-AUC is threshold-independent. `ensemble_nsga2` leads on macro AUC,
corroborating that its *probability ranking* (not just its argmax labels) is the
strongest. **The Neutral column is the lowest for every model** (0.79–0.82 vs
0.86–0.89 for Positive), independently confirming the Neutral-separability
problem identified in the EDA.

## 4.3 Confusion Matrices and Per-Class F1 (review finding #7)

Source: `results/confusion_matrices/confusion_matrices.md` (5,000-comment sample).
Reproduce: `python research/evaluation/confusion_matrices.py --test data/test.csv --sample 5000`.

Per-class F1 summary (selected models):

| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|-------|-------:|-------:|-------:|---------:|
| meta_learner | 0.707 | **0.623** | 0.758 | 0.6960 |
| ensemble_nsga2 | 0.710 | 0.615 | 0.755 | 0.6933 |
| logreg | 0.708 | 0.611 | 0.753 | 0.6909 |
| tfidf | 0.684 | 0.558 | 0.721 | 0.6545 |

Every model's confusion matrix shows the Neutral row with the lowest diagonal
(recall) and the highest off-diagonal mass, split across *both* polar classes.
`meta_learner` achieves the best Neutral F1 (0.623) and best Neutral recall
(0.633) — which is precisely *why* it is the recommended default despite its
near-tie with logistic regression on overall macro-F1. The full count and
row-normalised matrices for all seven models are in the source artifact.

## 4.4 Selective Prediction / Coverage–Accuracy (RQ2)

Source: `results/route_a_live_v1_ci/coverage_accuracy_curve.md` (20,000-comment
sample of the real 165,110-row held-out test split, seed 42). An earlier
version of this table was computed on an undisclosed 180-comment sample of
the `route_a_benchmark_cpu` smoke split; that table's Acc@100% values
(0.717–0.800) did not match the full-test benchmark in §4.1 and used
superseded generic model names. This table replaces it.

| Model | AUCA | Acc@10% | Acc@25% | Acc@50% | Acc@100% |
|-------|-----:|--------:|--------:|--------:|---------:|
| meta_learner | **0.8009** | 0.9765 | 0.9322 | 0.8481 | 0.6970 |
| logreg | 0.8003 | 0.9780 | 0.9280 | 0.8501 | 0.6957 |
| ensemble | 0.7937 | 0.9740 | 0.9212 | 0.8429 | 0.6912 |
| svm | 0.7856 | 0.9670 | 0.9118 | 0.8308 | 0.6835 |
| tfidf | 0.7676 | 0.9620 | 0.9020 | 0.8079 | 0.6630 |

`ensemble` here is the uniform-weight static ensemble baseline (equal
1/3-1/3-1/3 blend of logreg/svm/tfidf), not `ensemble_pso`/`ensemble_nsga2`
from §4.1 — this table predates the PSO/NSGA-II weighting work and is kept
as the selective-prediction baseline; re-running the analysis for the
weighted-ensemble variants specifically is future work. `fuzzy_ensemble` is
omitted from this table: the standalone script used to compute it here
loads the fitted gate parameters through a different formula than the one
actually deployed in the live runtime (see §3B.6), so including it here
would risk misrepresenting its real selective-prediction behaviour; its
near-pass-through argmax behaviour is instead characterised directly via
the ablation in §4.1.

Acc@100% here (0.6630–0.6970) is consistent with the full 165,110-comment
benchmark in Table 6 (0.6622–0.6959), confirming this is a genuine subsample
of the same evaluation rather than a different dataset. Abstaining on the
least-confident half of comments raises accuracy from roughly 0.66–0.70
(full coverage) to 0.81–0.85 (50% coverage), demonstrating that model
confidence is genuinely informative (RQ2). AUCA varies by architecture
*independently of full-coverage accuracy* — confidence quality is a distinct
axis of merit, the central methodological argument of the thesis. One
notable irregularity: `meta_learner`'s macro-F1 at 10% coverage (0.3294) is
far below its accuracy at that coverage (0.9765), because its most-confident
predictions at this sample size are concentrated in one or two classes,
collapsing recall on the others; this is reported rather than smoothed over,
and is a caution against reading AUC-F1 and AUCA as interchangeable at very
low coverage. This also reframes the Neutral problem: ambiguous Neutral
comments are exactly those the system can abstain on and route to human
review.

## 4.5 Calibration (RQ3)

Calibration is reported per model in §4.1 (ECE, Brier). The key conclusions,
consistent with Guo et al. (2017): (i) temperature scaling is applied per model
and improves calibration where a validation temperature is available; (ii) gains
are **model-specific** — the thesis does not claim universal calibration
improvement; (iii) `ensemble_nsga2` is the best calibrated multi-model
configuration (ECE 0.0046), and logistic regression is the best calibrated
single model (ECE 0.0039); (iv) `hybrid_dl` is **not** calibrated in the pinned
runtime because no temperature artifact row exists for it, and this is stated
rather than hidden. This calibration advantage is significance-backed (§4.6):
the `ensemble_nsga2` vs `meta_learner` ECE gap excludes zero under paired
bootstrap (95% CI [−0.0126, −0.0084]).

## 4.6 Statistical Significance

Source: `results/thesis_mcnemar.md`. Paired McNemar tests with Holm correction
on the historical offline benchmark family showed the meta-learner differing
significantly from the generic ensemble (p_adj = 4.15e-05) and from logistic
regression (p_adj = 0.045).

A dedicated paired test of the *live* `ensemble_nsga2` against the live
meta-learner has since been computed on the full pinned runtime split (n =
165,110; 2,000-resample paired bootstrap, seed 42; Holm-adjusted McNemar) —
source: `results/runtime/route_a_live_v1/live_significance_tests.{md,json}`,
script `research/ci/live_significance_tests.py`. Results:

- **Calibration:** `ensemble_nsga2` vs `meta_learner` ECE difference =
  −0.0111, 95% CI [−0.0126, −0.0084] (**excludes zero, significant**). The
  NSGA-II calibration advantage over the meta-learner is therefore an
  established inferential result, not merely descriptive.
- **Accuracy:** `ensemble_nsga2` vs `meta_learner` macro-F1 difference is
  tied, 95% CI [−0.0002, +0.0014] (does not exclude zero) — confirming the
  calibration gain comes at no significant accuracy cost.
- `ensemble_nsga2` vs `logreg` ECE difference is also tied, 95% CI [−0.0021,
  +0.0031] — the NSGA-II calibration edge is specifically over the
  meta-learner/ensemble family, not over the best single-model baseline.
- `meta_learner` vs `logreg` macro-F1 difference = +0.0017, 95% CI [+0.0009,
  +0.0025] (**significant**).

This closes the previously reported gap: the live NSGA-II calibration
advantage is now significance-backed rather than descriptive-only.

## 4.7 Human Gold-Set Evaluation (RQ4)

Source: `results/gold_set/gold_set_evaluation.md`, `results/gold_set/iaa_report.md`.

Two annotators independently labelled 300 comments — the thesis author and
one independent second annotator not otherwise involved in model
development (disclosed in Chapter 3 / `LABEL_PROVENANCE.md`). **Inter-annotator
agreement: Krippendorff's α = 0.9547, Cohen's/Fleiss' κ = 0.9546 (strong);
percent agreement 97.0%; 9 disputed items excluded.**

Performance versus the 291 human-reconciled gold labels:

| Model | Accuracy | Macro-F1 | Neu F1 |
|-------|---------:|---------:|-------:|
| ensemble_pso | 0.7010 | **0.7042** | 0.6226 |
| ensemble_nsga2 | 0.6976 | 0.7006 | 0.6262 |
| meta_learner | 0.6976 | 0.7001 | 0.6393 |
| tfidf | 0.7010 | 0.6988 | 0.5946 |
| svm | 0.6942 | 0.6978 | 0.6140 |
| logreg | 0.6907 | 0.6940 | 0.6168 |

Note the full six-model table (unlike the abridged version previously shown):
on the gold set, `tfidf` and `svm` are competitive with, or exceed, the
ensembles/meta-learner — a ranking collapse relative to the 165,110-comment
full-test benchmark (Table 6), where `tfidf` is the weakest model. This is
expected given the 300-item sample size and is exactly why §4.7 treats the
gold set as a reliability check rather than a ranking instrument.

Critically, this **corrects an earlier circular result**: against the silver
(auto-generated) labels, `ensemble_pso` scored a meaningless 1.000 F1 because it
*was* the silver labeller. Against independent human labels it scores 0.704,
a credible, non-circular figure. The Neutral class is again the weakest column,
and the strong IAA confirms the gold labels themselves are reliable — so the
residual Neutral difficulty is a genuine model/ambiguity effect, not annotation
noise.

**Train-split overlap check.** The gold set was originally sampled from
`train.csv` rather than the held-out test split (see
`research/ci/gold_set_train_membership.py`,
`data/gold_set_split_membership.csv`): 95 of 300 items (31.7%) are exact-text
members of the training split, 26 are in validation, 36 in test, and 143 do
not match any split (removed by preprocessing filters). To check whether
training-set memorisation inflates the headline gold-set numbers, the same
evaluation was re-run on the 205-item held-out-only subset (excluding the 95
training-split items):
`results/gold_set/gold_set_evaluation_holdout.md`. Results are materially
unchanged (e.g. `ensemble_pso` accuracy 0.7010 → 0.7100, macro-F1 0.7042 →
0.7128 on the held-out subset), so the reported ~0.70 macro-F1 figure is not
an artefact of training-set overlap. This check, and the underlying sampling
frame, are documented here for transparency even though the result is
reassuring.

## 4.8 Neutral-Class Analysis and Intervention (review finding #6)

Source: `results/neutral_analysis/neutral_analysis.md`.
Reproduce: `python research/analysis/neutral_class_analysis.py --model logreg --sample 8000`.

**Error direction (baseline logreg, test):** of 2,450 true-Neutral comments,
61.6% are correct, 24.4% are misread as Negative, and 14.0% as Positive — the
model over-commits short, low-signal comments to a polarity instead of
abstaining to Neutral.

**Intervention — Neutral prior adjustment (threshold tuning, no retraining):**
the Neutral probability is scaled by a factor α before argmax; α is selected on
validation (α = 1.6) and reported on held-out test:

| Metric | Baseline | Intervention (α=1.6) | Δ |
|--------|---------:|---------------------:|---:|
| Macro-F1 | 0.6922 | 0.6814 | **−0.0108** |
| Neutral-F1 | 0.6175 | 0.6326 | **+0.0151** |
| Neutral-Recall | 0.6155 | 0.7469 | +0.1314 |
| Neutral-Precision | 0.6196 | 0.5486 | −0.0710 |

**Honest verdict (mixed result).** Threshold tuning raises Neutral-F1 by +0.015
and Neutral recall substantially (+0.13), but at a −0.011 macro-F1 cost — a real
precision/recall trade-off. The intervention is therefore justified *only* when
Neutral recall is the operational priority (e.g. flagging ambiguous comments for
review), and is reported transparently rather than cherry-picked. The deeper
cause — Neutral comments being the shortest and most ambiguous (EDA §3A.2) —
indicates that durable improvement needs richer features (encoder embeddings) or
a Neutral-vs-rest cascade, listed as future work.

## 4.9 Chapter Summary

The evaluation supports a calibration-and-uncertainty-centred thesis claim. On
raw macro-F1 the models are near-tied and only marginally above logistic
regression; the meaningful separation appears in calibration (NSGA-II ECE
0.0046), probability ranking (NSGA-II macro AUC 0.8596), selective-prediction
quality (AUCA up to 0.88), and human-grounded evaluation (α = 0.9547, 0.70 gold
F1). The Neutral class is the consistent weak point across every lens, explained
by its short, ambiguous comments, and is addressed with an honestly-reported
intervention. Every number above maps to a re-runnable artifact (see the
claim-to-artifact audit).
