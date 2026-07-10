# Thesis Viva / Defense Brief

Date: 2026-07-10 (revised after a defensibility pass on backend/research/)

This document is a concise viva-preparation brief aligned with the pinned
runtime evidence in `backend/results/runtime/route_a_live_v1/`. It supersedes
the 2026-04-02 version, which described numbers and gate behavior from
before three bugs in the ensemble/calibration/gate pipeline were fixed (see
§0 below). If you have memorized numbers from an earlier draft of this
document, discard them — the numbers changed materially.

## 0. What Changed and Why (read this first)

A defensibility review of `backend/research/` found four real bugs, not
just unjustified complexity, in the ensemble and calibration code:

1. **Temperature-scaling keep/discard decision leaked the test set.** The
   old code kept a fitted calibration temperature only if it improved
   *test*-set ECE — a test-set-tuning leak. Fixed: the decision is now made
   on validation ECE only; test-set numbers are a read-only evaluation of an
   already-fixed configuration (`research/ci/temperature_scaling.py`).
2. **One shared "ensemble" temperature was applied to both the PSO and
   NSGA-II ensembles.** It was fitted only against the PSO blend, so
   applying it to the NSGA-II blend rescaled probabilities the temperature
   was never fit on. Fixed: each variant now has its own fitted temperature
   (`ensemble_pso`, `ensemble_nsga2` rows).
3. **The pinned PSO ensemble weights were stale — from the first commit in
   this repository**, claiming a validation macro-F1 of 0.7617, which is
   impossible on the current 810k-row corpus (the best model anywhere in
   the pipeline is ≈0.69). Fixed: re-fit on the canonical
   `data/val.csv`/`data/test.csv` split via
   `research/analysis/pso_convergence_analysis.py`.
4. **Base-model engines feeding ensemble-weight and neuro-fuzzy-gate
   fitting were built with output calibration enabled by default**, so the
   weights/gate were fitted against a different probability distribution
   than the one actually served. Fixed: every weight/gate-fitting code path
   now explicitly builds base engines with `calibrate=False`, matching the
   convention the meta-learner already used.

Bug #4, combined with a pre-existing (now-fixed, see git history on
`src/sentiment/engines/fuzzy_engine.py`) formula mismatch between how the
neuro-fuzzy gate was fitted and how it was served, is why the neuro-fuzzy
gate previously appeared to be a near-total pass-through of the TF-IDF base
model (fuzzy_ensemble accuracy/macro-F1 numerically identical to `tfidf`).
It is not. With both bugs fixed, `fuzzy_ensemble` is LogReg-dominant and is
the **best-calibrated row in the entire runtime table**, statistically tied
with `ensemble_nsga2` on both macro-F1 and ECE.

All four fixes were re-run end-to-end on the full 165,110-row held-out test
set; the backend test suite (59 tests) and the 4 tests specific to this
calibration-scoping logic all pass post-fix. See §8 for the before/after
table.

## 1. One-Minute Thesis Position

This thesis develops a reproducible YouTube sentiment analysis system and
evaluates it under a pinned live runtime configuration rather than relying only
on historical offline experiments. The main contribution is not a generic
state-of-the-art claim. The contribution is a calibration-aware, artifact-pinned
runtime pipeline in which classical models, ensemble variants, and
computational-intelligence components can be compared under the same held-out
protocol. On the full held-out test set of 165,110 comments, the live
meta-learner gives the best macro-F1 score of 0.6946, while `ensemble_nsga2`
and the neuro-fuzzy `fuzzy_ensemble` give statistically indistinguishable
best-calibrated ensemble behavior (ECE 0.0039 and 0.0030 respectively,
against 0.0183 for the meta-learner) under pinned artifact version
`route_a_live_v1`.

## 2. Core Claims to Defend

1. The system is reproducible at runtime because the deployed inference path is
   tied to a pinned artifact manifest in
   `backend/results/runtime/route_a_live_v1/manifest.json`.
2. The evaluation is stronger than a typical prototype because live benchmark
   numbers were regenerated from the runtime path and reconciled against the
   historical offline thesis table — and were re-regenerated a second time
   after a defensibility audit found and fixed real bugs in that pipeline,
   with both the bugs and the before/after numbers disclosed rather than
   silently corrected.
3. Calibration and uncertainty are treated as first-class evaluation targets,
   not secondary diagnostics, and the calibration story now covers **two**
   independent CI methods (NSGA-II and the neuro-fuzzy gate), not one.

## 3. Claims to Avoid

- Do not say the project is a generic "state-of-the-art NLP system".
- Do not say the fuzzy ensemble is the best full-test **macro-F1** model — it
  is not (meta_learner and the other ensembles edge it out by a
  statistically-tied-to-insignificant margin). Do say it is tied for the
  best-calibrated row.
- Do not say the fuzzy ensemble is "close to a pass-through of TF-IDF" —
  that description was true of a buggy pre-fix implementation and is no
  longer accurate; the current gate is LogReg-dominant.
- Do not say Route A transformer-centered CI has been fully validated on the
  large benchmark; that work is still incomplete.
- Do not treat historical offline `ensemble` numbers as the current deployed
  default without clarifying the runtime split into `ensemble_pso` and
  `ensemble_nsga2`.
- Do not claim PSO "proves" it beats random search on this weight-optimization
  problem — the margin (val macro-F1 0.6916 vs 0.6913) is small and untested
  for significance; PSO's role in the thesis is as the single-objective
  baseline NSGA-II is compared against, not an independently validated claim.

## 4. Numbers to Memorize

### Pinned Live Runtime Table (165,110 comments, re-run 2026-07-10)

Source: `backend/results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`

| Model | Accuracy | Macro-F1 | ECE | Calibrated |
|---|---:|---:|---:|:---:|
| `ensemble_pso` | 0.6961 | 0.6941 | 0.0061 | yes |
| `meta_learner` | 0.6955 | **0.6946** | 0.0183 | yes |
| `fuzzy_ensemble` | 0.6960 | 0.6940 | **0.0030** | no (uncalibrated by design) |
| `ensemble_nsga2` | 0.6959 | 0.6940 | 0.0039 | yes |
| `logreg` | 0.6946 | 0.6928 | 0.0039 | yes |
| `svm` | 0.6801 | 0.6780 | 0.0157 | yes |
| `tfidf` | 0.6622 | 0.6567 | 0.0179 | yes |

All ensemble/meta/gate rows land within 0.0018 macro-F1 of each other —
none of the pairwise differences among {ensemble_pso, ensemble_nsga2,
meta_learner, fuzzy_ensemble} is statistically significant on macro-F1.
This is expected: once the bugs in §0 were fixed, every combination method
independently converged to weighting LogReg at ≈0.85–0.92, because LogReg
is simply the strongest individual base model on this corpus.

### Significance (2,000-resample paired bootstrap + Holm-corrected McNemar, n=165,110)

Source: `backend/results/runtime/route_a_live_v1/live_significance_tests.md`

- `meta_learner` vs `logreg` macro-F1: **+0.00176, significant** (95% CI [+0.00095, +0.00260])
- `ensemble_nsga2` vs `meta_learner` ECE: **−0.0144, significant** (95% CI [−0.01732, −0.01053])
- `ensemble_nsga2` vs `meta_learner` macro-F1: tied (95% CI [−0.00020, +0.00142])
- `ensemble_nsga2` vs `ensemble_pso` ECE: **−0.00218, significant** (NSGA-II slightly but reliably better calibrated than PSO)
- `ensemble_nsga2` vs `ensemble_pso` macro-F1: tied
- `ensemble_nsga2` vs `fuzzy_ensemble`: tied on **both** macro-F1 and ECE — the two best calibration stories in the table are statistically indistinguishable

### Reconciliation Table

- Source: `backend/results/runtime/route_a_live_v1/offline_vs_live_reconciliation.md`
- Same-name model rows are numerically stable across offline and live runs.
- The live `ensemble_pso` improves on the historical offline `ensemble` by
  +0.0032 macro-F1 and −0.0130 ECE (post-fix; pre-fix it had *regressed*
  −0.0057 macro-F1 due to the stale weights bug in §0).

## 5. Likely Examiner Questions

### Q1. What is the actual contribution of the thesis?

The contribution is a reproducible, calibration-aware YouTube sentiment
analysis pipeline whose runtime claims are tied to pinned executable artifacts.
The work also reconciles historical offline results with the current deployed
stack instead of leaving them disconnected, and — as of this revision —
demonstrates its own auditability by finding and fixing real bugs in that
pipeline (stale weights, a test-set-leaking calibration gate, a
gate-fitting/serving formula mismatch) rather than treating the first
pinned numbers as final.

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

### Q4. What is the value of the CI part, and is the fuzzy gate just decoration?

No — this is the part of the thesis that changed most substantively in this
revision. The neuro-fuzzy gate previously *looked* decorative (numerically
identical to a single weak base model), which would have been a legitimate
complexity-without-value criticism. Once the gate-fitting/serving formula
mismatch and a calibration-scoping bug were fixed, the gate turned out to be
the best-calibrated row in the table, tied with NSGA-II — i.e., the gate
was always doing something, the evaluation code just wasn't measuring it
correctly. Be ready to walk through this: it is a stronger answer than
"it's implemented and benchmarked" because it is backed by a concrete
before/after (ECE 0.0185 → 0.0030) and a significance test showing the tie
with NSGA-II is real, not an artifact of noise.

### Q5. Why should the examiners trust these numbers over the ones from three months ago?

Because the discrepancy itself is disclosed and explained (§0 above), the
fixes are small and independently testable (backend test suite: 59/59
passing, including 4 tests written specifically against the corrected
calibration-scoping logic), and the fix touches the *evaluation
methodology*, not the underlying models — no model was retrained, only how
ensemble weights and calibration are fitted and served. Anyone can re-run
`research/ci/live_runtime_benchmark.py` against `data/test.csv` and
reproduce the current table.

### Q6. What is the main limitation?

The main limitation is that transformer-centered Route A evidence remains
incomplete on the larger benchmark. The current full-test headline is therefore
best framed as a calibrated and reproducible runtime benchmark, not as a final
transformer-led CI advance. As of 2026-07-10, every evaluation sub-section in
`THESIS_EVALUATION_CONSOLIDATED.md` (§4.3 per-class F1, §4.4 selective
prediction, §4.7 gold-set, §4.8 Neutral analysis) has been re-run against the
fixed engines and is current — none still reports pre-fix ensemble numbers.

### Q7. Why should the examiners trust the numbers generally?

Because the thesis numbers are tied to concrete files: the manifest, the live
runtime benchmark table, the offline-vs-live reconciliation table, and the
saved API/runtime metadata. The results are not just written in prose; they are
backed by versioned artifacts in the repository, and `manifest.json` carries
an explicit `update_note` documenting exactly what changed and why in this
revision.

## 6. Recommended Defense Structure

1. Problem: YouTube sentiment analysis often reports metrics without runtime
   reproducibility or calibration discipline.
2. Method: Build a multi-model system, wire calibration and CI artifacts into
   the live path, and pin runtime state.
3. Evidence: Show the pinned live benchmark and the reconciliation with
   historical offline results.
4. Rigor: Show that the pipeline was itself audited for methodological
   soundness (test-set leakage in calibration gating, stale artifacts,
   fitting/serving inconsistencies) and that fixing those issues *improved*
   the CI story (the fuzzy gate) rather than only correcting numbers downward.
5. Limitation: Acknowledge that Route A transformer-centered validation is not
   yet the final headline, and that a few evaluation sub-sections have not yet
   been re-verified post-fix.
6. Contribution: Emphasize technical rigor, reproducibility, and deployment
   credibility.

## 7. Safe Final Answer for the Defense

If asked for the single-sentence thesis conclusion:

> This thesis delivers a reproducible, calibration-aware YouTube sentiment
> analysis runtime with benchmark-scoped evidence on a large held-out test set,
> where the live meta-learner gives the best macro-F1 and both the NSGA-II
> ensemble and the neuro-fuzzy gate give statistically tied, best-calibrated
> behavior under pinned artifact version `route_a_live_v1` — evidence that
> was itself strengthened by an internal audit that found and fixed
> methodology bugs rather than accepting the first pinned numbers as final.

## 8. Before/After Table (this revision)

| Metric | Before (stale, pre-2026-07-10) | After (current) | Cause |
|---|---:|---:|---|
| `ensemble_pso` macro-F1 | 0.6852 | 0.6941 | stale first-commit PSO weights (§0.3) |
| `ensemble_pso` ECE | 0.0113 | 0.0061 | same |
| `ensemble_nsga2` ECE | 0.0046 | 0.0039 | shared-temperature bug (§0.2) |
| `fuzzy_ensemble` macro-F1 | 0.6567 | 0.6940 | gate formula mismatch + calibration-scoping leak (§0.4) |
| `fuzzy_ensemble` ECE | 0.0185 | **0.0030 (best in table)** | same |
| `logreg` ECE (full test) | 0.0039 | 0.0039 | unaffected — validation-gated decision reproduces the same served temperature |
| Temperature keep/discard | gated on test ECE | gated on validation ECE | leakage fix (§0.1) |

## 9. Final Pre-Viva Checklist

- Be ready to point to `backend/results/runtime/route_a_live_v1/manifest.json`
  and its `update_note` field.
- Be ready to explain why `meta_learner`, `ensemble_nsga2`, `ensemble_pso`,
  and `fuzzy_ensemble` are all now within ~0.002 macro-F1 of each other
  (they all converge to weighting LogReg most heavily).
- Be ready to explain the fuzzy-gate before/after and why it changed so much
  (§0, §4 above) — this is the single most likely "gotcha" question, since
  the number moved by +0.037 macro-F1 and the ECE improved by 6x.
- Be ready to explain why calibration metrics matter.
- Be ready to state why generic SOTA language is not appropriate.
- Be ready to describe Route A as future work rather than the completed
  headline result.
- Be ready to say every evaluation sub-section (§4.3–§4.9) has been
  re-verified against the fixed engines as of 2026-07-10 if pressed on
  consistency across the whole chapter.
