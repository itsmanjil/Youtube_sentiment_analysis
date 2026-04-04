# Thesis Final Checklist

Status date: 2026-04-04

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

- [ ] Verify frontend surfacing for uncertainty and calibration.
  Still blocked here because `node` and `npm` are unavailable, so `vitest`
  and UI snapshot checks were not run.

- [ ] Compare live runtime predictions against offline artifact predictions on
  the same held-out split.
  The benchmark-level reconciliation artifact was refreshed, but per-sample
  prediction equality is still not proven.

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
  This still requires human annotation even though repo support scripts exist.

- [ ] Add a cross-domain or domain-shift evaluation.
  This still needs a chosen evaluation set or selected target channels/videos.

- [x] Write a clean "Threats to Validity" section.
  See `backend/docs/THESIS_RISKS_GAPS.md`.

- [x] Write a short ethics and data-governance section.
  See the ethics/data-governance material in
  `backend/docs/THESIS_RISKS_GAPS.md`.

- [x] Make the final thesis text honest about aspect analysis.
  Thesis-facing docs now treat the current feature as a keyword-level aspect
  proxy rather than full ABSA.

- [ ] Ensure every thesis claim maps to a runnable script or stored artifact.
  This is improved, but still not fully closed until the final written thesis is
  checked claim-by-claim against the repo artifacts.

## P2 - Good To Finish If Time Allows

- [ ] Add a near-duplicate leakage audit.
  Exact dedupe already exists, but near-duplicate spam/paraphrase leakage can
  still inflate results.

- [ ] Promote key provenance metrics to a more stable schema if needed.
  Useful if dashboard querying becomes part of the final thesis story.

- [ ] Expand thesis-grade validation around selective prediction / abstention.
  Useful if you want the uncertainty-aware story to be stronger than confidence
  reporting alone.

- [ ] Tighten the encoder-first thesis narrative.
  The roadmap still recommends an encoder-centered Route A contribution, but the
  current thesis headline remains classical/ensemble-first.

## Suggested Finish Order

1. Add the missing prediction-level live-vs-offline comparison.
2. Run frontend `vitest` and snapshot checks on a machine with `node`.
3. Add gold-set and domain-shift evidence.
4. Do a final thesis-text pass against the pinned runtime artifacts.
5. Treat Route A encoder-first work as future work unless stronger evidence is added.

## Do Not Overclaim

- [x] Do not say the project solves full ABSA.
- [x] Do not say all calibrated models improved.
- [x] Do not say `hybrid_dl` is calibrated unless you add a real artifact row.
- [x] Do not say the thesis is fully reproducible unless the final written
      results match the pinned runtime artifacts.
