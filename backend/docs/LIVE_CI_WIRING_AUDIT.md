# Live CI Wiring Audit

Date: 2026-05-17

## Scope

This audit checks whether the offline computational-intelligence artifacts
(temperature scaling, optimized ensemble weights, neuro-fuzzy gating, and
uncertainty reporting) are connected to the live inference path, surfaced to
the frontend/API payloads, and backed by direct backend regression tests.

## Verified Wiring

### Backend request -> inference path

- `app/views.py`
  - Accepts `ensemble_weights_optimization`
  - Passes optimized ensemble settings into the live engine
  - Computes per-analysis entropy statistics
  - Persists confidence, uncertainty, calibration, transformer, ensemble, and
    fuzzy metadata into `analysis_meta`
  - Persists both `weights_source` and
    `weights_optimization_requested` for ensemble runs
  - Returns `uncertainty_stats` and `calibration` in the analysis detail API

### Runtime artifact loading

- `src/sentiment/engines/logreg_engine.py`
- `src/sentiment/engines/svm_engine.py`
- `src/sentiment/engines/tfidf_engine.py`
- `src/sentiment/engines/meta_learner_engine.py`
- `src/sentiment/engines/hybrid_dl_engine.py`
- `src/sentiment/engines/ensemble_engine.py`
- `src/sentiment/engines/fuzzy_engine.py`

These engines load thesis-facing runtime artifacts through the pinned runtime
artifact resolver, with `route_a_live_v1` as the current default artifact
version.

### Frontend surfacing

- `frontend/src/Views/Pages/Search.jsx`
  - Sends `ensemble_weights_optimization`
  - Exposes PSO vs NSGA-II selection
  - Exposes `hybrid_dl`, transformer, ensemble, and fuzzy options
- `frontend/src/Views/Pages/Dashboard.jsx`
  - Shows entropy summary
  - Shows temperature badge when calibration is applied
  - Shows neuro-fuzzy gate active badge
- `frontend/src/Views/Pages/Report.jsx`
  - Shows calibration rows
  - Shows uncertainty table
  - Shows fuzzy routing mode
- `frontend/src/Views/Pages/Monitoring.jsx`
  - Carries uncertainty and calibration into dashboard navigation
  - Displays uncertainty and temperature on cards

## Validation Status

### Confirmed

- Backend tests pass: `40/40`
- Live API code persists uncertainty and calibration metadata
- Live API code persists ensemble `weights_source` and
  `weights_optimization_requested`
- Ensemble runtime switches between PSO and NSGA-II artifact weights through
  direct backend regression tests
- `hybrid_dl` remains uncalibrated (`T=1.0`,
  `calibration_applied=false`) when no runtime artifact row exists
- Fuzzy runtime only activates the learned neuro-fuzzy gate when the requested
  base-model set matches the trained gate model set
- The live runtime benchmark, manifest, and offline-vs-live reconciliation are
  present under `backend/results/runtime/route_a_live_v1/`
- Frontend `vitest` passes with `81/81` tests after `npm ci`
- Prediction-level live-vs-offline label equivalence is now documented in
  `backend/results/runtime/route_a_live_v1/prediction_level_reconciliation.md`

### Not yet validated end-to-end

- Gold-set evaluation and domain-shift evaluation remain outside the scope of
  this wiring audit

## Remaining Material Gaps

### 1. `hybrid_dl` calibration is wired, but no artifact entry exists

`hybrid_dl_engine.py` looks for a `hybrid_dl` row inside the pinned
temperature-scaling artifact. The current artifact has no such entry, so
runtime falls back to `T=1.0` and `calibration_applied=false`.

### 2. Calibration claims must remain model-specific

The runtime supports calibration-aware inference, but the thesis should not say
that calibration improved every model. The safe claim is narrower: calibration
metadata is wired into the runtime, and the strongest deployment-oriented
calibration result in the pinned benchmark is currently `ensemble_nsga2`.

### 3. Neuro-fuzzy gate activation is exact-match dependent

`fuzzy_engine.py` activates the learned gate only when the requested base-model
set matches the artifact model set. Any different combination falls back to the
static fuzzy path.

## Thesis-Ready Validation Checklist

### Artifact governance

- [x] Freeze runtime artifacts under a versioned directory
- [x] Record artifact SHA256 hashes in a manifest
- [x] Record which artifact version the live API used for each benchmark table
- [x] Stop pointing thesis-critical runtime at mutable root result files

### Backend verification

- [x] Add API tests for `uncertainty_stats`, `calibration`, `weights_source`,
      `weights_optimization_requested`, and `nf_gate_active`
- [x] Add one test that verifies NSGA-II vs PSO changes `weights_source`
- [x] Add one test that proves `hybrid_dl` remains uncalibrated when no artifact
      row exists
- [x] Add one test that proves fuzzy gate activation requires matching base
      models

### Frontend verification

- [x] Run `vitest` once `node`/`npm` are available
- [x] Component-check Dashboard, Report, and Monitoring for confidence,
      calibration, and uncertainty rendering
- [x] Verify empty-state behavior when calibration metadata is absent
- [x] Verify fallback behavior when calibration metadata arrives outside
      `analysis_meta`

### Benchmark validation

- [x] Regenerate one benchmark table using the live runtime path, not only the
      offline research scripts
- [x] Refresh benchmark-level offline-vs-live reconciliation under the pinned
      runtime directory
- [x] Compare live runtime predictions against offline artifact predictions on
      the same held-out split
- [x] Confirm that the live configuration used in the thesis matches the pinned
      manifest and benchmark artifacts

### Claim discipline

- [x] Claim runtime support for calibrated uncertainty-aware inference
- [x] Limit calibration claims to models/artifacts supported by stored runtime
      evidence
- [x] State clearly that `hybrid_dl` is not calibrated until a real artifact row
      exists
- [x] Cite the exact pinned runtime version for thesis-facing results

## Recommended Order

1. Add gold-set and domain-shift evidence for thesis credibility.
2. Keep the final thesis wording tied to `route_a_live_v1`.
