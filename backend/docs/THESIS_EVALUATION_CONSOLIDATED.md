# Chapter 4 — Consolidated Evaluation

Status date: 2026-07-02 (updated)

This chapter consolidates every evaluation strand into one defensible narrative,
mapping each result to its reproducible artifact. The argument is deliberately
*not* "our model has the best F1"; it is "calibration, uncertainty, and human
grounding distinguish the models where raw F1 does not."

## 4.1 Headline Runtime Performance (165,110 comments)

Source: `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`.
Updated 2026-07-10 after fixing three bugs in the ensemble/calibration/gate
pipeline (stale first-commit PSO weights, a single shared ensemble
temperature applied to both PSO and NSGA-II variants, and base-model engines
being built with output calibration enabled inside weight/gate-fitting code
— see `results/runtime/route_a_live_v1/manifest.json`'s `update_note` for
the full list). All numbers below are post-fix.

| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | ms/sample |
|-------|---------:|---------:|----:|------:|:----------:|----------:|
| ensemble_pso | 0.6961 | 0.6941 | 0.0061 | 0.4090 | yes | 0.18 |
| meta_learner | 0.6955 | **0.6946** | 0.0183 | 0.4118 | yes | 0.19 |
| fuzzy_ensemble | 0.6960 | 0.6940 | **0.0030** | 0.4093 | no | 0.20 |
| ensemble_nsga2 | 0.6959 | 0.6940 | 0.0039 | 0.4092 | yes | 0.18 |
| logreg | 0.6946 | 0.6928 | 0.0039 | 0.4100 | yes | 0.06 |
| svm | 0.6801 | 0.6780 | 0.0157 | 0.4291 | yes | 0.05 |
| tfidf | 0.6622 | 0.6567 | 0.0179 | 0.4491 | yes | 0.06 |

**Honest reading of the F1 gap (addresses review finding #5).** The best model
(meta_learner) exceeds the logistic-regression baseline by only **+0.0018
macro-F1** on the full test set — a paired bootstrap 95% CI of
[+0.00095, +0.00260] confirms this is a real but small effect (excludes
zero). This margin is small and the thesis does **not** rest its
contribution on it. Once the PSO/NSGA-II/gate-fitting bugs above were fixed,
**every** ensemble strategy (PSO, NSGA-II, stacking, the neuro-fuzzy gate)
converges to being LogReg-dominated (weights ≈0.85–0.92 on LogReg across
methods) and lands within 0.0006 macro-F1 of each other — none of the
pairwise differences among {ensemble_pso, ensemble_nsga2, meta_learner,
fuzzy_ensemble} is statistically significant on macro-F1 (see
`live_significance_tests.md`). The substantive differences are in
*calibration*: `fuzzy_ensemble` and `ensemble_nsga2` attain ECE ≈0.003–0.004
— roughly a fifth of the meta-learner's 0.0183 — while matching it on
accuracy (`ensemble_nsga2` vs `meta_learner` ECE difference: 95% CI
[-0.01732, -0.01053], excludes zero). The thesis claim is therefore that
multi-objective / uncertainty-aware ensemble combination buys **calibration
quality at no accuracy cost** (RQ1, RQ3), which is the deployment-relevant
property, not a raw-F1 win — and this is now true of both the NSGA-II
ensemble and the neuro-fuzzy gate, not NSGA-II alone.

**Note on the fuzzy ensemble row (revised).** An earlier version of this
chapter reported that the neuro-fuzzy-gated configuration was numerically
identical to the TF-IDF + Naive-Bayes base classifier, attributing this to
the gate rarely overriding the argmax. That reading rested on two bugs that
have since been fixed: (1) the deployed blend and the fitting script used
non-equivalent formulas for how the fitted `alpha` parameter entered the
Gaussian membership activation, and (2) the base models feeding the gate
were scored with output calibration enabled, differing from what the gate
was actually fitted against (see `docs/THESIS_CHAPTER3B_SYSTEM_DESIGN.md`
§3B.6 "Deployment note" for the full account). With both fixed, the gate is
now LogReg-dominant rather than a TF-IDF pass-through, and the
`fuzzy_ensemble` row reaches macro-F1 0.6940 / ECE 0.0030 — the best ECE of
any row in the table above, and statistically tied with `ensemble_nsga2` on
both macro-F1 and ECE (bootstrap 95% CIs include zero for both). A direct
ablation against LogReg (the gate's now-dominant base model) on a
40,000-comment sample (seed 42) shows the gate changes the base classifier's
argmax on 2.74% of comments (1,096 of 40,000) — 456 corrections, 412
regressions, and 228 wrong-to-wrong flips (both predictions incorrect but
for different labels) — a small net-positive edit rate, not the near-total
pass-through the pre-fix numbers suggested. Source:
`results/neuro_fuzzy_gate_ablation/fuzzy_gate_ablation.md`, reproducible via
`python research/ci/fuzzy_gate_ablation.py --sample 40000 --seed 42 --base_model logreg`.

## 4.2 ROC-AUC, One-vs-Rest (review finding #7)

Source: `results/roc_auc/roc_auc.md` (5,000-comment sample, seed 42).
Reproduce: `python research/evaluation/roc_auc.py --test data/test.csv --sample 5000`.

| Model | Macro AUC | Positive AUC | Neutral AUC | Negative AUC |
|-------|----------:|-------------:|------------:|-------------:|
| ensemble_pso | **0.8597** | 0.8949 | 0.8186 | 0.8656 |
| ensemble_nsga2 | 0.8596 | 0.8949 | 0.8183 | 0.8655 |
| fuzzy_ensemble | 0.8591 | 0.8943 | 0.8182 | 0.8649 |
| logreg | 0.8589 | 0.8945 | 0.8172 | 0.8652 |
| meta_learner | 0.8575 | 0.8919 | 0.8186 | 0.8621 |
| svm | 0.8435 | 0.8846 | 0.7956 | 0.8501 |
| tfidf | 0.8315 | 0.8684 | 0.7925 | 0.8335 |

Re-run 2026-07-10 after fixing the PSO/NSGA-II/gate calibration-scoping bugs
described in §4.1; `ensemble_pso` previously showed macro AUC 0.8511 (last
of the ensemble/meta rows) due to the same stale first-commit PSO weights
that depressed its macro-F1, and `fuzzy_ensemble` previously showed 0.8314
(lowest of the table) due to the same gate bugs. ROC-AUC is
threshold-independent; the top five rows above (all ensemble/meta variants
plus logreg) now sit within 0.0022 macro AUC of each other, so no single
model's probability ranking is meaningfully strongest — a materially
different reading from the pre-fix table, where `ensemble_nsga2` appeared to
lead by a wider margin. **The Neutral column is the lowest for every model**
(0.79–0.82 vs 0.86–0.89 for Positive), independently confirming the
Neutral-separability problem identified in the EDA; this observation is
unaffected by the bug fixes.

## 4.3 Confusion Matrices and Per-Class F1 (review finding #7)

Source: `results/confusion_matrices/confusion_matrices.md` (5,000-comment
sample, seed 42). Reproduce:
`python research/evaluation/confusion_matrices.py --test data/test.csv --sample 5000`.
Re-run 2026-07-10 against the fixed engines described in §4.1/§0 of
`THESIS_VIVA_DEFENSE_BRIEF.md`; this section previously showed only 4 of 7
models and predates the PSO-weights/calibration-scoping/gate-formula fixes.

Full per-class F1 summary (all seven models, re-run):

| Model | Neg F1 | Neu F1 | Pos F1 | Macro F1 |
|-------|-------:|-------:|-------:|---------:|
| meta_learner | 0.707 | **0.623** | 0.759 | 0.6964 |
| ensemble_pso | 0.710 | 0.616 | 0.756 | 0.6940 |
| ensemble_nsga2 | 0.710 | 0.616 | 0.755 | 0.6937 |
| fuzzy_ensemble | 0.710 | 0.613 | 0.755 | 0.6927 |
| logreg | 0.708 | 0.611 | 0.753 | 0.6909 |
| svm | 0.699 | 0.600 | 0.739 | 0.6791 |
| tfidf | 0.684 | 0.558 | 0.721 | 0.6545 |

Every model's confusion matrix shows the Neutral row with the lowest diagonal
(recall) and the highest off-diagonal mass, split across *both* polar classes.
`meta_learner` retains the best Neutral F1 (0.623) and best Neutral recall
— which is precisely *why* it is the recommended default despite its
near-tie with the ensemble family on overall macro-F1. `ensemble_pso` and
`fuzzy_ensemble` are now present in this table for the first time (they were
previously omitted or dragged below `logreg` by the bugs in §0): both now
sit essentially on top of `ensemble_nsga2`, consistent with the §4.1 finding
that every ensemble/gate method converges to a similar LogReg-dominant
blend once fitted and served on the same probability distribution. The full
count and row-normalised matrices for all seven models are in the source
artifact.

## 4.4 Selective Prediction / Coverage–Accuracy (RQ2)

Source: `results/route_a_live_v1_ci/coverage_accuracy_curve.md` (20,000-comment
sample of the real 165,110-row held-out test split, seed 42). An earlier
version of this table was computed on an undisclosed 180-comment sample of
the `route_a_benchmark_cpu` smoke split; that table's Acc@100% values
(0.717–0.800) did not match the full-test benchmark in §4.1 and used
superseded generic model names. Re-run 2026-07-10 against the fixed engines:
the script previously scored a bare `get_sentiment_engine("ensemble")` (which
silently resolves to the PSO weights internally, not a uniform baseline as
earlier text here claimed) and reconstructed the neuro-fuzzy gate as a
second, standalone implementation loading a `neuro_fuzzy_gate.json` path
that did not exist at the location this script wrote to — silently dropping
`fuzzy_ensemble` from the table with no error, rather than because of a
formula mismatch as previously stated here. Both are fixed:
`research/ci/coverage_accuracy_curve.py` now scores every row, including
both ensemble variants and the fuzzy gate, by calling
`get_sentiment_engine(...)` exactly as the live runtime does, with no
standalone re-implementation.

| Model | AUCA | AUC-F1 | Acc@10% | Acc@25% | Acc@50% | Acc@100% |
|-------|-----:|-------:|--------:|--------:|--------:|---------:|
| ensemble_pso | **0.8311** | 0.8137 | 0.9780 | 0.9335 | 0.8518 | 0.6966 |
| meta_learner | 0.8309 | 0.7479 | 0.9765 | 0.9387 | 0.8481 | 0.6973 |
| ensemble_nsga2 | 0.8308 | 0.8128 | 0.9775 | 0.9344 | 0.8514 | 0.6964 |
| fuzzy_ensemble | 0.8307 | 0.8135 | 0.9775 | 0.9323 | 0.8504 | 0.6976 |
| logreg | 0.8300 | 0.8123 | 0.9780 | 0.9313 | 0.8501 | 0.6957 |
| svm | 0.8152 | 0.8029 | 0.9665 | 0.9148 | 0.8306 | 0.6835 |
| tfidf | 0.7972 | 0.7667 | 0.9620 | 0.9054 | 0.8079 | 0.6630 |

`fuzzy_ensemble` is present in this table for the first time — it was
previously either omitted (as here) or, in §4.1's ablation-only
characterisation, described as a near-pass-through of a single weak base
model. It is not: AUCA 0.8307 is within 0.0004 of `ensemble_nsga2` and
0.0002 of `logreg`, and its Acc@100% (0.6976) is the highest of any model in
this table. `ensemble_pso` moved from AUCA 0.7937 (stale PSO weights, §0)
to **0.8311**, now the top-ranked model by this metric. All five ensemble/
meta/gate/logreg rows sit within 0.0011 AUCA of each other — the same
convergence-to-a-similar-configuration pattern noted in §4.1 and §4.3.

Acc@100% here (0.6630–0.6976) is consistent with the full 165,110-comment
benchmark in Table 6 (0.6567–0.6946 — note the pinned-artifact row order
differs slightly by sample), confirming this is a genuine subsample of the
same evaluation rather than a different dataset. Abstaining on the
least-confident half of comments raises accuracy from roughly 0.66–0.70
(full coverage) to 0.83–0.85 (50% coverage), demonstrating that model
confidence is genuinely informative (RQ2). AUCA varies by architecture
*independently of full-coverage accuracy* — confidence quality is a distinct
axis of merit, the central methodological argument of the thesis. One
notable irregularity, confirmed unchanged by this re-run: `meta_learner`'s
macro-F1 at 10% coverage (0.3294) is far below its accuracy at that coverage
(0.9765), because its most-confident predictions at this sample size are
concentrated in one or two classes, collapsing recall on the others — visible
in its AUC-F1 (0.7479) sitting well below every other model's (0.77–0.81)
despite a competitive AUCA; this is reported rather than smoothed over, and
is a caution against reading AUC-F1 and AUCA as interchangeable at very low
coverage. This also reframes the Neutral problem: ambiguous Neutral comments
are exactly those the system can abstain on and route to human review.

## 4.5 Calibration (RQ3)

Calibration is reported per model in §4.1 (ECE, Brier); numbers below refer
to the 2026-07-10 re-run described there. The key conclusions, consistent
with Guo et al. (2017): (i) temperature scaling is applied per model and
kept only when it improves **validation**-set ECE — the keep/discard
decision never inspects the test set (see
`results/runtime/route_a_live_v1/temperature_scaling.md`, "Gating"); (ii)
gains are **model-specific** — the thesis does not claim universal
calibration improvement; (iii) `fuzzy_ensemble` is the best calibrated
multi-model configuration (ECE 0.0030), narrowly ahead of `ensemble_nsga2`
(0.0039, statistically tied per §4.6) and logistic regression (ECE 0.0039,
also tied); (iv) `hybrid_dl` is **not** calibrated in the pinned runtime
because no temperature artifact row exists for it, and this is stated
rather than hidden. The calibration advantage of the ensemble/gate family
over the meta-learner is significance-backed (§4.6): the `ensemble_nsga2`
vs `meta_learner` ECE gap excludes zero under paired bootstrap (95% CI
[−0.01732, −0.01053]).

## 4.6 Statistical Significance

Source: `results/thesis_mcnemar.md`. Paired McNemar tests with Holm correction
on the historical offline benchmark family showed the meta-learner differing
significantly from the generic ensemble (p_adj = 4.15e-05) and from logistic
regression (p_adj = 0.045).

A dedicated paired test of the *live* ensemble/gate family against the live
meta-learner and logreg has since been computed on the full pinned runtime
split (n = 165,110; 2,000-resample paired bootstrap, seed 42; Holm-adjusted
McNemar) — source:
`results/runtime/route_a_live_v1/live_significance_tests.{md,json}`, script
`research/ci/live_significance_tests.py`. As of the 2026-07-10 re-run (after
fixing the PSO-weights/calibration-scoping/gate-formula bugs described in
§4.1), every model row is reconstructed by calling the exact same
`get_sentiment_engine(...)` code path the live benchmark uses — not a
second, independently re-implemented ensembling formula — so "reproduction
validation" now checks artifact reproducibility across runs rather than
agreement between two implementations. Results:

- **Calibration:** `ensemble_nsga2` vs `meta_learner` ECE difference =
  −0.0144, 95% CI [−0.01732, −0.01053] (**excludes zero, significant**). The
  ensemble-family calibration advantage over the meta-learner is therefore
  an established inferential result, not merely descriptive.
- **Accuracy:** `ensemble_nsga2` vs `meta_learner` macro-F1 difference is
  tied, 95% CI [−0.00020, +0.00142] (does not exclude zero) — confirming
  the calibration gain comes at no significant accuracy cost.
- `ensemble_nsga2` vs `logreg` ECE difference is also tied, 95% CI
  [−0.00136, +0.00070] — the ensemble-family calibration edge is
  specifically over the meta-learner, not over the best single-model
  baseline.
- `ensemble_nsga2` vs `ensemble_pso`: ECE difference −0.00218, 95% CI
  [−0.00263, −0.00084] (**excludes zero, significant** — NSGA-II is
  slightly but reliably better calibrated than PSO); macro-F1 difference is
  tied, 95% CI [−0.00049, +0.00023].
- `ensemble_nsga2` vs `fuzzy_ensemble`: both macro-F1 (95% CI [−0.00061,
  +0.00056]) and ECE (95% CI [−0.00235, +0.00135]) are tied — the two best
  calibration stories in the table are statistically indistinguishable from
  each other.
- `meta_learner` vs `logreg` macro-F1 difference = +0.00176, 95% CI
  [+0.00095, +0.00260] (**significant**). Note the paired McNemar test for
  this same pair is *not* significant after Holm correction (p_holm =
  0.270) — the two tests disagree because McNemar only uses discordant-pair
  counts while the bootstrap CI uses the full macro-F1 statistic; both
  numbers are reported rather than only the one that is significant.

This closes the previously reported gap: the live ensemble-family
calibration advantage is significance-backed rather than descriptive-only,
and now extends to the neuro-fuzzy gate as well as NSGA-II.

## 4.7 Human Gold-Set Evaluation (RQ4)

Source: `results/gold_set/gold_set_evaluation.md`, `results/gold_set/iaa_report.md`.
Re-run 2026-07-10 against the fixed engines described in §4.1/§0 of
`THESIS_VIVA_DEFENSE_BRIEF.md`.

Two annotators independently labelled 300 comments — the thesis author and
one independent second annotator not otherwise involved in model
development (disclosed in Chapter 3 / `LABEL_PROVENANCE.md`). **Inter-annotator
agreement: Krippendorff's α = 0.9547, Cohen's/Fleiss' κ = 0.9546 (strong);
percent agreement 97.0%; 9 disputed items excluded.** (IAA is computed from
the human annotations directly and is unaffected by any model-side fix.)

Performance versus the 291 human-reconciled gold labels:

| Model | Accuracy | Macro-F1 | Neu F1 |
|-------|---------:|---------:|-------:|
| ensemble_pso | 0.7045 | **0.7075** | 0.6291 |
| ensemble_nsga2 | 0.6976 | 0.7006 | 0.6262 |
| meta_learner | 0.6976 | 0.7001 | **0.6393** |
| tfidf | 0.7010 | 0.6988 | 0.5946 |
| svm | 0.6942 | 0.6978 | 0.6140 |
| logreg | 0.6907 | 0.6940 | 0.6168 |

`ensemble_pso` moved from accuracy 0.7010 / macro-F1 0.7042 (stale weights,
pre-fix) to 0.7045 / **0.7075** (now the best macro-F1 in this table) — the
same stale-PSO-weights bug from §0 depressed this row too, on an entirely
different evaluation set from the one that first surfaced it.
`ensemble_nsga2`, `meta_learner`, `tfidf`, `svm`, and `logreg` are unchanged
(none of their served predictions or calibration depend on the fixed
components — `ensemble_nsga2`'s weights were already correct and its
temperature change is argmax-invariant).

Note the full six-model table: on the gold set, `tfidf` and `svm` remain
competitive with, or close to, the meta-learner/NSGA-II ensemble — a ranking
compression relative to the 165,110-comment full-test benchmark (Table 6),
where `tfidf` is the weakest model by a wide margin. This is expected given
the 300-item sample size and is exactly why §4.7 treats the gold set as a
reliability check rather than a ranking instrument.

Critically, this **corrects an earlier circular result**: against the silver
(auto-generated) labels, `ensemble_pso` scored a meaningless 1.000 F1 because it
*was* the silver labeller. Against independent human labels it scores 0.708,
a credible, non-circular figure. The Neutral class is again the weakest column
for most models (`meta_learner` is the exception, retaining the best Neutral
F1 on the gold set as it does on the full test benchmark in §4.3), and the
strong IAA confirms the gold labels themselves are reliable — so the residual
Neutral difficulty is a genuine model/ambiguity effect, not annotation noise.

**Train-split overlap check.** The gold set was originally sampled from
`train.csv` rather than the held-out test split (see
`research/ci/gold_set_train_membership.py`,
`data/gold_set_split_membership.csv`): 95 of 300 items (31.7%) are exact-text
members of the training split, 26 are in validation, 36 in test, and 143 do
not match any split (removed by preprocessing filters). To check whether
training-set memorisation inflates the headline gold-set numbers, the same
evaluation was re-run on the 205-item held-out-only subset (excluding the 95
training-split items):
`results/gold_set/gold_set_evaluation_holdout.md`. Results (re-run
2026-07-10) are materially unchanged (`ensemble_pso` accuracy 0.7045 →
0.7050, macro-F1 0.7075 → 0.7088 on the held-out subset; every other model
moves by at most ±0.006 macro-F1), so the reported ~0.70 macro-F1 figure is
not an artefact of training-set overlap, and this remains true after the §0
fixes. This check, and the underlying sampling frame, are documented here
for transparency even though the result is reassuring.

## 4.8 Neutral-Class Analysis and Intervention (review finding #6)

Source: `results/neutral_analysis/neutral_analysis.md`.
Reproduce: `python research/analysis/neutral_class_analysis.py --model logreg --sample 8000`.
Re-run 2026-07-10 to confirm it is unaffected by the §0/§4.1 fixes: this
analysis only ever scores `logreg` in isolation (no ensemble or gate
component), and `logreg`'s served predictions and calibration are identical
before and after those fixes (§4.1's ECE for `logreg` is unchanged at
0.0039). The numbers below are confirmed byte-for-byte identical to the
prior run.

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
regression; the meaningful separation appears in calibration (`fuzzy_ensemble`
and `ensemble_nsga2` ECE ≈0.003–0.004, statistically tied with each other and
both significantly better calibrated than the meta-learner), probability
ranking (top five models within 0.0022 macro AUC of each other, all ≈0.858–0.860),
selective-prediction quality (AUCA up to 0.8311, §4.4 — a pre-existing
"0.88" figure here in an earlier revision did not match any number in the
coverage-accuracy table and has been corrected), and human-grounded evaluation
(α = 0.9547, 0.70 gold F1). Once the PSO-weights/calibration-scoping/gate-formula
bugs described in §4.1 and §3B.6 (`THESIS_CHAPTER3B_SYSTEM_DESIGN.md`) were
fixed, the neuro-fuzzy gate joined NSGA-II as a second, statistically
indistinguishable best-calibrated configuration — this is a materially
different and stronger CI-contribution story than the pre-fix numbers
supported, where the gate appeared to be a near-total pass-through of its
weakest base model. The Neutral class is the consistent weak point across
every lens, explained by its short, ambiguous comments, and is addressed with
an honestly-reported intervention. Every number above maps to a re-runnable
artifact (see the claim-to-artifact audit). §4.3, §4.4, §4.7, and §4.8 have
all been re-run against the fixed engines as of 2026-07-10: §4.3 (full-test
per-class F1), §4.4 (selective prediction), and §4.7 (human gold-set) all
moved — `ensemble_pso` improved across all three (full-test macro-F1
0.6852→0.6941, AUCA 0.7937→0.8311, gold-set macro-F1 0.7042→0.7075) and
`fuzzy_ensemble` newly appears in both the §4.3 and §4.4 tables instead of
being omitted, in §4.4's case because of a silent file-path bug rather than
the formula-mismatch reason previously given there — while §4.8
(Neutral-class analysis, `logreg`-only) was confirmed unchanged, since it
does not exercise any of the fixed ensemble/gate components. No
section of this chapter still reflects pre-fix ensemble numbers.
