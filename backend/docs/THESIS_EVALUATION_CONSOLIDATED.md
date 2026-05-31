# Chapter 4 — Consolidated Evaluation

Status date: 2026-05-31

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

Source: `results/route_a_benchmark_cpu_ci/coverage_accuracy_curve.md`.

| Model | AUCA | Acc@25% | Acc@50% | Acc@100% |
|-------|-----:|--------:|--------:|---------:|
| svm | **0.8806** | 1.000 | 0.944 | 0.789 |
| ensemble | 0.8781 | 1.000 | 0.956 | 0.800 |
| neuro_fuzzy | 0.8747 | 1.000 | 0.933 | 0.789 |
| logreg | 0.8679 | 1.000 | 0.922 | 0.756 |
| meta_learner | 0.8663 | 0.978 | 0.933 | 0.756 |
| tfidf | 0.8243 | 0.978 | 0.889 | 0.717 |

Abstaining on the least-confident half of comments raises accuracy from ~0.79 to
~0.94, demonstrating that model confidence is genuinely informative (RQ2). AUCA
varies by architecture *independently of full-coverage accuracy* — confidence
quality is a distinct axis of merit, the central methodological argument of the
thesis. This also reframes the Neutral problem: ambiguous Neutral comments are
exactly those the system can abstain on and route to human review.

## 4.5 Calibration (RQ3)

Calibration is reported per model in §4.1 (ECE, Brier). The key conclusions,
consistent with Guo et al. (2017): (i) temperature scaling is applied per model
and improves calibration where a validation temperature is available; (ii) gains
are **model-specific** — the thesis does not claim universal calibration
improvement; (iii) `ensemble_nsga2` is the best calibrated multi-model
configuration (ECE 0.0046), and logistic regression is the best calibrated
single model (ECE 0.0039); (iv) `hybrid_dl` is **not** calibrated in the pinned
runtime because no temperature artifact row exists for it, and this is stated
rather than hidden.

## 4.6 Statistical Significance

Source: `results/thesis_mcnemar.md`. Paired McNemar tests with Holm correction
on the historical offline benchmark family showed the meta-learner differing
significantly from the generic ensemble (p_adj = 4.15e-05) and from logistic
regression (p_adj = 0.045). A dedicated paired test of the *live* `ensemble_nsga2`
against the live meta-learner has **not** yet been computed; accordingly the live
NSGA-II calibration advantage is reported as a strong descriptive result, not an
established inferential one. This honesty is itself a threat-to-validity control.

## 4.7 Human Gold-Set Evaluation (RQ4)

Source: `results/gold_set/gold_set_evaluation.md`, `results/gold_set/iaa_report.md`.

Two annotators independently labelled 300 comments. **Inter-annotator
agreement: Krippendorff's α = 0.9547, Cohen's/Fleiss' κ = 0.9546 (strong);
percent agreement 97.0%; 9 disputed items excluded.**

Performance versus the 291 human-reconciled gold labels:

| Model | Accuracy | Macro-F1 | Neu F1 |
|-------|---------:|---------:|-------:|
| ensemble_pso | 0.7010 | **0.7042** | 0.6226 |
| meta_learner | 0.6976 | 0.7001 | 0.6393 |
| ensemble_nsga2 | 0.6976 | 0.7006 | 0.6262 |
| logreg | 0.6907 | 0.6940 | 0.6168 |

Critically, this **corrects an earlier circular result**: against the silver
(auto-generated) labels, `ensemble_pso` scored a meaningless 1.000 F1 because it
*was* the silver labeller. Against independent human labels it scores 0.704,
a credible, non-circular figure. The Neutral class is again the weakest column,
and the strong IAA confirms the gold labels themselves are reliable — so the
residual Neutral difficulty is a genuine model/ambiguity effect, not annotation
noise.

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
