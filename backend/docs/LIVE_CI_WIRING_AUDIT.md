# Live CI Wiring Audit

Date: 2026-04-02

## Scope

This audit checks whether the offline computational-intelligence artifacts
(temperature scaling, optimized ensemble weights, neuro-fuzzy gating, and
uncertainty reporting) are actually connected to the live inference path and
surfaced to the frontend.

## Verified Wiring

### Backend request → inference path

- `app/views.py`
  - Accepts `ensemble_weights_optimization`
  - Passes optimized ensemble settings into the live engine
  - Computes per-analysis entropy statistics
  - Persists confidence, uncertainty, calibration, transformer, ensemble,
    meta-learner, and fuzzy metadata into `analysis_meta`
  - Returns `uncertainty_stats` and `calibration` in the analysis detail API

### Runtime artifact loading

- `src/sentiment/engines/logreg_engine.py`
- `src/sentiment/engines/svm_engine.py`
- `src/sentiment/engines/tfidf_engine.py`
- `src/sentiment/engines/meta_learner_engine.py`
- `src/sentiment/engines/hybrid_dl_engine.py`
- `src/sentiment/engines/ensemble_engine.py`
- `src/sentiment/engines/fuzzy_engine.py`

These engines now read runtime research artifacts from `backend/results/`.

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

- Backend tests pass: `31/31`
- Live API code persists uncertainty and calibration metadata
- Ensemble runtime can switch between PSO and NSGA-II artifact weights
- Fuzzy runtime loads learned ANFIS membership functions when model sets match

### Not yet validated end-to-end

- Frontend automated tests were not run in this environment because `node` is
  unavailable
- No audited live benchmark table has been regenerated from the runtime path
- No artifact version locking exists between thesis results and deployed runtime

## Material Gaps

### 1. Runtime uses mutable root result artifacts

Live engines load from shared root files such as:

- `backend/results/temperature_scaling.json`
- `backend/results/pso_ensemble_weights.json`
- `backend/results/multi_objective_ensemble.json`
- `backend/results/neuro_fuzzy_gate.json`

This is practical, but weak for thesis reproducibility. A later research run can
silently change live behavior without changing code.

### 2. `hybrid_dl` calibration is wired, but no artifact entry exists

`hybrid_dl_engine.py` looks for a `hybrid_dl` row inside
`backend/results/temperature_scaling.json`. The current artifact has no such
entry, so runtime falls back to `T=1.0` and `calibration_applied=false`.

### 3. Some loaded calibration artifacts are not obviously beneficial

Current root `temperature_scaling.json` reports:

- `logreg`: ECE `0.006789 -> 0.00741`
- `svm`: ECE `0.012565 -> 0.016255`
- `tfidf`: ECE `0.01308 -> 0.017405`
- `meta_learner`: ECE `0.02029 -> 0.022973`
- `ensemble`: ECE `0.021617 -> 0.011672`

So live calibration clearly helps `ensemble`, but the stored artifact appears to
worsen ECE for several other models. This needs an explicit thesis decision:
disable those temperatures, or regenerate them.

### 4. Requested optimization mode is not explicitly persisted

The API accepts `ensemble_weights_optimization`, and the ensemble engine uses it,
but `analysis_meta["ensemble"]` persists `weights_source`, not the original
request parameter. That is enough to explain what was applied, but not enough to
fully reconstruct the request.

### 5. Test coverage does not directly lock this live wiring

Backend tests cover parts of the transformer/calibration pipeline, but there is
no explicit regression test that asserts:

- `analysis_meta["calibration"]` is returned for calibrated live models
- `analysis_meta["uncertainty_stats"]` is present and shaped correctly
- `analysis_meta["ensemble"]["weights_source"]` is preserved
- `analysis_meta["fuzzy"]["nf_gate_active"]` is exposed when the ANFIS gate is active

### 6. Neuro-fuzzy gate activation is exact-match dependent

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
      and `nf_gate_active`
- [ ] Add one test that verifies NSGA-II vs PSO changes `weights_source`
- [ ] Add one test that proves `hybrid_dl` remains uncalibrated when no artifact
      row exists
- [ ] Add one test that proves fuzzy gate activation requires matching base models

### Frontend verification

- [ ] Run `vitest` once `node` is available
- [ ] Snapshot-check Search, Dashboard, Report, and Monitoring for calibration
      and uncertainty rendering
- [ ] Verify empty-state behavior when calibration metadata is absent
- [ ] Verify fallback behavior when fuzzy gate is inactive

### Benchmark validation

- [x] Regenerate one benchmark table using the live runtime path, not only the
      offline research scripts
- [ ] Compare live runtime predictions against offline artifact predictions on
      the same held-out split
- [x] Recompute calibration metrics after live wiring
- [ ] Confirm that the live configuration used in the thesis matches the final
      stored artifacts

### Claim discipline

- [ ] Claim runtime support for calibrated uncertainty-aware inference
- [ ] Do not claim all models are improved by calibration unless the stored
      calibration artifacts show that
- [ ] Do not claim `hybrid_dl` is calibrated until a real artifact row exists
- [ ] Do not claim full reproducibility until artifact versions are pinned

## Recommended Order

1. Freeze and version the runtime artifacts
2. Add backend regression tests for live metadata exposure
3. Run frontend `vitest`
4. Generate one live benchmark table from the deployed stack
5. Update thesis wording to match the frozen runtime artifacts
