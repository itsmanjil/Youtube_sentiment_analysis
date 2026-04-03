# Thesis Results Draft

Date: 2026-04-02

This document contains paste-ready thesis wording aligned with the pinned live
runtime artifacts in `backend/results/runtime/route_a_live_v1/`.

## Abstract Draft

This thesis presents a reproducible sentiment analysis system for YouTube
comments that combines classical machine learning, ensemble learning, and
computational-intelligence components within a single runtime pipeline. The
system was evaluated on a fixed held-out test set of 165,110 comments using a
pinned runtime artifact version (`route_a_live_v1`). On the full test set, the
best macro-F1 was achieved by the stacked meta-learner (macro-F1 = 0.6945,
accuracy = 0.6953), while the best calibrated ensemble was the NSGA-II-weighted
ensemble (macro-F1 = 0.6940, accuracy = 0.6959, ECE = 0.004601). Logistic
regression remained the best single-model calibration baseline (ECE = 0.003900,
macro-F1 = 0.6928). Historical offline and pinned live results were reconciled,
showing that same-name models remained numerically stable, while the live
ensemble should now be interpreted through explicit `ensemble_pso` and
`ensemble_nsga2` variants rather than a single generic ensemble row. These
results support a benchmark-scoped claim of a reproducible, calibration-aware
YouTube sentiment analysis pipeline, rather than a generic state-of-the-art NLP
claim.

## Results Chapter Draft

### Primary Results Paragraph

The primary full-test evaluation was conducted using the pinned live runtime
artifact version `route_a_live_v1` on a held-out test set of 165,110 YouTube
comments. Under this deployed configuration, the highest macro-F1 was obtained
by the meta-learner (macro-F1 = 0.6945, accuracy = 0.6953), whereas the highest
accuracy and strongest calibrated ensemble performance were obtained by the
NSGA-II-weighted ensemble (accuracy = 0.6959, macro-F1 = 0.6940, ECE = 0.004601,
Brier = 0.409204). Logistic regression remained highly competitive as a single
model, achieving macro-F1 = 0.6928 with the lowest ECE among the non-ensemble
models (ECE = 0.003900). In contrast, the fuzzy ensemble did not improve the
full-test headline metrics in the current pinned runtime configuration
(macro-F1 = 0.6567, accuracy = 0.6622).

### Runtime Interpretation Paragraph

These findings indicate that the current deployed system is best described as a
calibrated classical/ensemble pipeline with strong runtime reproducibility. The
meta-learner provides the best overall macro-F1, while the NSGA-II ensemble
provides the strongest accuracy-calibration trade-off for deployment-oriented
use. Therefore, when the thesis emphasizes classification effectiveness, the
meta-learner should be cited as the main runtime headline; when the emphasis is
reliability under probabilistic decision-making, the NSGA-II ensemble is the
more appropriate focal model.

### Offline vs Live Reconciliation Paragraph

To verify that the deployed runtime remained consistent with the historical
offline benchmark tables, an explicit reconciliation was performed between the
historical offline results and the pinned live runtime benchmark. Same-name
models were numerically stable across the two evaluation paths: TF-IDF, logistic
regression, and SVM were unchanged on accuracy and macro-F1, while the
meta-learner differed only marginally (Δ accuracy = -0.0002, Δ macro-F1 =
-0.0001). The main difference concerned the ensemble row. The historical offline
benchmark reported a single generic ensemble result, whereas the live runtime
now exposes two distinct ensemble variants: `ensemble_pso` and
`ensemble_nsga2`. Relative to the historical ensemble row, the live
NSGA-II ensemble improved accuracy by +0.0027, macro-F1 by +0.0031, and reduced
ECE by -0.014459, whereas the PSO ensemble traded lower macro-F1 for better
calibration than the historical ensemble.

### Statistical Significance Draft

For the historical offline benchmark, paired McNemar testing showed that the
meta-learner significantly outperformed the historical ensemble
(`p_adj = 4.15e-05`) and significantly differed from logistic regression
(`p_adj = 0.045`) on the same held-out test split. These tests remain useful for
interpreting the historical benchmark family. However, because paired
significance testing has not yet been rerun specifically for the pinned live
`ensemble_nsga2` runtime variant against the meta-learner, the live NSGA-II
advantages should currently be interpreted as descriptive runtime findings
rather than as final inferential claims.

## Discussion Draft

The present findings do not support a claim that the fuzzy ensemble is the best
overall system on the full held-out test set. Instead, the evidence supports a
more precise conclusion: the repository currently delivers a reproducible live
runtime in which the meta-learner gives the best macro-F1, the NSGA-II ensemble
gives the best calibrated ensemble behavior, and logistic regression remains a
strong single-model baseline. This is a stronger and more defensible thesis
position than a generic claim of “state-of-the-art” performance, because it is
tied to a pinned artifact version, a fixed held-out split, and explicit runtime
evaluation outputs.

## Limitations Draft

Several limitations should be stated explicitly. First, the strongest runtime
results are still classical-first rather than transformer-first, so the Route A
goal of establishing a validated transformer-centered CI contribution remains
incomplete. Second, the full-test significance analysis has not yet been rerun
for the pinned live `ensemble_nsga2` variant, which limits inferential claims
about its superiority over the meta-learner. Third, the fuzzy ensemble is
implemented and active in the runtime path, but it does not currently improve
the full-test headline metrics. Finally, although runtime artifact pinning now
substantially improves reproducibility, any new thesis claim should continue to
cite the exact artifact version (`route_a_live_v1`) and associated manifest.

## Conclusion Draft

In conclusion, the current repository supports a defensible thesis claim of a
reproducible, calibration-aware YouTube sentiment analysis system evaluated on a
large held-out test set. Within the pinned live runtime, the meta-learner is the
best macro-F1 model, the NSGA-II ensemble is the best calibrated ensemble, and
logistic regression remains the strongest single-model calibration baseline. The
appropriate interpretation is therefore not that the project has established a
generic state-of-the-art NLP system, but that it has produced a benchmark-scoped
and deployment-aware sentiment analysis pipeline with traceable runtime
artifacts and stable offline-to-live behavior.

## Safe Claim Wording

Use one of these in the thesis:

- “On the pinned live runtime benchmark (`route_a_live_v1`), the stacked
  meta-learner achieved the highest macro-F1 (0.6945) on the held-out YouTube
  test set.”
- “The NSGA-II-weighted ensemble achieved the strongest deployment-oriented
  trade-off in the live runtime, combining 0.6959 accuracy with ECE 0.004601.”
- “The live runtime results are consistent with the historical offline
  benchmark, but the ensemble result must now be interpreted through explicit
  PSO and NSGA-II variants rather than a single generic ensemble row.”

Avoid these:

- “The system is state-of-the-art.”
- “The fuzzy system is the best overall model.”
- “All models improved after calibration.”
