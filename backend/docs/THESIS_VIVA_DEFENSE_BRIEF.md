# Thesis Viva / Defense Brief

Date: 2026-04-02

This document is a concise viva-preparation brief aligned with the pinned
runtime evidence in `backend/results/runtime/route_a_live_v1/`.

## 1. One-Minute Thesis Position

This thesis develops a reproducible YouTube sentiment analysis system and
evaluates it under a pinned live runtime configuration rather than relying only
on historical offline experiments. The main contribution is not a generic
state-of-the-art claim. The contribution is a calibration-aware, artifact-pinned
runtime pipeline in which classical models, ensemble variants, and
computational-intelligence components can be compared under the same held-out
protocol. On the full held-out test set of 165,110 comments, the live
meta-learner gives the best macro-F1 score of 0.6945, while the NSGA-II
ensemble gives the best calibrated ensemble behavior with 0.6959 accuracy and
ECE of 0.004601 under pinned artifact version `route_a_live_v1`.

## 2. Core Claims to Defend

1. The system is reproducible at runtime because the deployed inference path is
   tied to a pinned artifact manifest in
   `backend/results/runtime/route_a_live_v1/manifest.json`.
2. The evaluation is stronger than a typical prototype because live benchmark
   numbers were regenerated from the runtime path and reconciled against the
   historical offline thesis table.
3. Calibration and uncertainty are treated as first-class evaluation targets,
   not secondary diagnostics.

## 3. Claims to Avoid

- Do not say the project is a generic “state-of-the-art NLP system”.
- Do not say the fuzzy ensemble is the best full-test model.
- Do not say Route A transformer-centered CI has been fully validated on the
  large benchmark; that work is still incomplete.
- Do not treat historical offline `ensemble` numbers as the current deployed
  default without clarifying the runtime split into `ensemble_pso` and
  `ensemble_nsga2`.

## 4. Numbers to Memorize

### Historical Offline Table

- Source: `backend/results/thesis_model_performance_youtube_filtered.md`
- Best offline historical macro-F1: `meta_learner = 0.6946`
- Historical offline `ensemble = 0.6909` macro-F1
- Best single classical baseline: `logreg = 0.6928` macro-F1

### Pinned Live Runtime Table

- Source: `backend/results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`
- Test size: `165110`
- `meta_learner`: accuracy `0.6953`, macro-F1 `0.6945`, ECE `0.015711`
- `ensemble_nsga2`: accuracy `0.6959`, macro-F1 `0.6940`, ECE `0.004601`
- `logreg`: accuracy `0.6946`, macro-F1 `0.6928`, ECE `0.003900`
- `fuzzy_ensemble`: accuracy `0.6622`, macro-F1 `0.6567`

### Reconciliation Table

- Source:
  `backend/results/runtime/route_a_live_v1/offline_vs_live_reconciliation.md`
- Same-name model rows are numerically stable across offline and live runs.
- The live `ensemble_nsga2` improves on the historical offline `ensemble` by
  `+0.0031` macro-F1 and `-0.014459` ECE.

## 5. Likely Examiner Questions

### Q1. What is the actual contribution of the thesis?

The contribution is a reproducible, calibration-aware YouTube sentiment
analysis pipeline whose runtime claims are tied to pinned executable artifacts.
The work also reconciles historical offline results with the current deployed
stack instead of leaving them disconnected.

### Q2. Why is the headline not a transformer model?

Because the strongest validated full-test evidence in the current repository is
still classical-first. The transformer Route A infrastructure was built, but
the large-scale encoder-centered evidence is not yet the strongest runtime
result. It would be methodologically wrong to headline a weaker model just
because it is newer.

### Q3. Why is calibration important here?

Sentiment systems used in dashboards or decision support should not only rank
labels correctly. They should also produce confidence values that are
meaningful. That is why ECE, Brier score, entropy, and temperature scaling were
integrated into the live path.

### Q4. What is the value of the CI part if fuzzy is not the best full-test row?

The CI contribution is broader than a single winning row. It includes
multi-objective ensemble optimization, uncertainty-aware routing, calibration,
and live wiring of those artifacts into the runtime stack. The full-test fuzzy
row is not the headline, but the CI methods are still implemented, benchmarked,
and available for controlled comparison.

### Q5. What is the main limitation?

The main limitation is that transformer-centered Route A evidence remains
incomplete on the larger benchmark. The current full-test headline is therefore
best framed as a calibrated and reproducible runtime benchmark, not as a final
transformer-led CI advance.

### Q6. Why should the examiners trust the numbers?

Because the thesis numbers are tied to concrete files: the manifest, the live
runtime benchmark table, the offline-vs-live reconciliation table, and the
saved API/runtime metadata. The results are not just written in prose; they are
backed by versioned artifacts in the repository.

## 6. Recommended Defense Structure

1. Problem: YouTube sentiment analysis often reports metrics without runtime
   reproducibility or calibration discipline.
2. Method: Build a multi-model system, wire calibration and CI artifacts into
   the live path, and pin runtime state.
3. Evidence: Show the pinned live benchmark and the reconciliation with
   historical offline results.
4. Limitation: Acknowledge that Route A transformer-centered validation is not
   yet the final headline.
5. Contribution: Emphasize technical rigor, reproducibility, and deployment
   credibility.

## 7. Safe Final Answer for the Defense

If asked for the single-sentence thesis conclusion:

> This thesis delivers a reproducible, calibration-aware YouTube sentiment
> analysis runtime with benchmark-scoped evidence on a large held-out test set,
> where the live meta-learner is the best macro-F1 model and the NSGA-II
> ensemble is the strongest calibrated ensemble under pinned artifact version
> `route_a_live_v1`.

## 8. Final Pre-Viva Checklist

- Be ready to point to `backend/results/runtime/route_a_live_v1/manifest.json`
- Be ready to explain why `meta_learner` and `ensemble_nsga2` are different
  thesis headlines
- Be ready to explain why calibration metrics matter
- Be ready to state why generic SOTA language is not appropriate
- Be ready to describe Route A as future work rather than the completed
  headline result
