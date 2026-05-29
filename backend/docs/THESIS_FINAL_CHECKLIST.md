# Thesis Final Checklist

Status date: 2026-05-18

This checklist ranks the remaining work by thesis impact, not engineering
convenience. Completed items below are limited to work that is now backed by
repo state, tests, or thesis-facing documentation.

If time is limited, finish `P0` first, then `P1`. `P2` is useful, but not the
core of a defensible final submission.

## P0 - Must Finish Before Final Thesis

- [x] Freeze the final thesis claim and make it narrower than the original
  proposal.
  The repo now has a consistent headline story: a validated, uncertainty-aware,
  artifact-pinned YouTube sentiment runtime. See
  `backend/docs/THESIS_RESULTS_RUNTIME_DRAFT.md`,
  `backend/docs/THESIS_ABSTRACT_CHAPTER4_POLISHED.md`, and
  `backend/docs/THESIS_VIVA_DEFENSE_BRIEF.md`.

- [x] Confirm the exact runtime configuration used for the thesis tables.
  Thesis-facing results are tied to `backend/results/runtime/route_a_live_v1/`,
  especially `manifest.json`,
  `live_runtime_benchmark_full_test.md`, and
  `offline_vs_live_reconciliation.md`.

- [x] Run and document the missing backend regression checks.
  `backend/app/tests.py` now covers NSGA-II vs PSO `weights_source`,
  `weights_optimization_requested`, `hybrid_dl` uncalibrated fallback, and
  fuzzy exact-match gate activation. The backend suite passes with `40/40`
  tests.

- [x] Verify frontend surfacing for uncertainty and calibration.
  Frontend dependencies install with `npm ci`; `npm test -- --run` now passes
  with `81/81` tests. Monitoring and Dashboard tests cover confidence,
  uncertainty, and calibration surfacing/fallback behavior.

- [x] Compare live runtime predictions against offline artifact predictions on
  the same held-out split.
  `backend/research/ci/prediction_level_reconciliation.py` writes
  `backend/results/runtime/route_a_live_v1/prediction_level_reconciliation.*`.
  The current artifact shows 100% label-level agreement for `logreg` and `svm`
  against the offline benchmark CPU probability cube, with probability drift
  reported separately.

- [x] Decide the calibration policy model-by-model.
  The safe policy is now documented: do not claim universal calibration gains;
  only make calibration claims supported by the pinned runtime evidence.

- [x] State clearly that `hybrid_dl` is not calibrated unless a real artifact
  row is added.
  This is now explicit in the audit/results drafts.

- [x] Lock the final benchmark headline.
  The headline remains: `meta_learner` is best by macro-F1 on the pinned live
  runtime, while `ensemble_nsga2` is the strongest calibrated ensemble.

## P1 - Strongly Recommended For Thesis Credibility

- [ ] Build a small human-labeled gold set.
  This still requires real human annotation. The tooling and evaluator are now
  runnable: `scripts/annotate.py`, `scripts/prepare/merge_annotations.py`, and
  `research/ci/gold_set_evaluation.py`. The exact command path is documented in
  `backend/docs/THESIS_CLAIM_ARTIFACT_AUDIT.md`. Current regenerated results are
  still source/silver-label evidence, not human IAA evidence.

- [x] Add a cross-domain or domain-shift evaluation utility.
  `research/evaluation/domain_shift.py` now writes metadata-backed reports.
  The source dataset metadata is retained in `data/route_a_domain_10k/`, and
  the current thesis-facing artifacts are
  `results/domain_shift/category_domain_shift.*` and
  `results/domain_shift/country_domain_shift.*`. Exact per-video and
  per-timestamp slices are still too sparse in the 10k sample, so CategoryID
  and CountryCode are the defensible domain slices.

- [x] Write a clean "Threats to Validity" section.
  See `backend/docs/THESIS_RISKS_GAPS.md`.

- [x] Write a short ethics and data-governance section.
  See the ethics/data-governance material in
  `backend/docs/THESIS_RISKS_GAPS.md`.

- [x] Make the final thesis text honest about aspect analysis.
  Thesis-facing docs now treat the current feature as a keyword-level aspect
  proxy rather than full ABSA.

- [x] Ensure every thesis claim maps to a runnable script or stored artifact.
  `backend/docs/THESIS_CLAIM_ARTIFACT_AUDIT.md` now maps supported claims to
  concrete artifacts and commands, and explicitly marks unsupported claims as
  blocked or future work.

## P2 - Good To Finish If Time Allows

- [x] Add a near-duplicate leakage audit.
  `scripts/prepare/near_duplicate_audit.py` writes
  `results/leakage/near_duplicate_audit.*`. The current benchmark CPU audit
  found no exact cross-split duplicates and 9 near-duplicate candidates for
  review, mostly repeated emoji-heavy comments.

- [x] Promote key provenance metrics to a more stable schema if needed.
  `backend/docs/PROVENANCE_SCHEMA.md` now defines the lightweight result
  provenance contract used by thesis-facing artifacts.

- [x] Expand thesis-grade validation around selective prediction / abstention.
  `research/ci/coverage_accuracy_curve.py` and
  `research/ci/entropy_gated_prediction.py` now regenerate
  `results/route_a_benchmark_cpu_ci/coverage_accuracy_curve.*` and
  `results/route_a_benchmark_cpu_ci/entropy_gated_prediction.*`.

- [x] Tighten the encoder-first thesis narrative.
  `backend/docs/ROUTE_A_ENCODER_POSITION.md` now states that Route A encoder
  work is implemented but should remain future work unless rerun with
  `transformers`, `torch`, and suitable compute.

## Suggested Finish Order

1. Add human gold-set evidence.
2. Do a final thesis-text pass against the pinned runtime artifacts.
3. Treat Route A encoder-first work as future work unless stronger evidence is added.

## Do Not Overclaim

- [x] Do not say the project solves full ABSA.
- [x] Do not say all calibrated models improved.
- [x] Do not say `hybrid_dl` is calibrated unless you add a real artifact row.
- [x] Do not say the thesis is fully reproducible unless the final written
      results match the pinned runtime artifacts.
