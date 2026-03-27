# Route A Implementation Roadmap

This document turns the recommended thesis path into a repo-specific execution plan.

## Goal

Build a **validated Computational Intelligence contribution** on top of **strong pretrained encoders**, not on top of the current classical-only headline stack.

Recommended thesis claim:

> An adaptive uncertainty-aware multi-objective ensemble over strong pretrained encoders improves calibration and robustness under domain shift for YouTube sentiment classification.

## Why this route fits this repo

The repository already has three useful ingredients:

1. A leakage-aware split pipeline in `backend/scripts/prepare/prepare_hf_dataset.py`
2. A reusable API/runtime inference path in `backend/app/views.py`
3. CI research modules for calibration, selective prediction, NSGA-II, and neuro-fuzzy gating under `backend/research/ci/`

The main gap is that the current best stored benchmark is still classical-first:

- `backend/results/thesis_model_performance_youtube_filtered.md`: `meta_learner` macro-F1 `0.6946`
- `backend/results/fuzzy_best_test.json`: current fuzzy headline macro-F1 `0.6676`

That means the repo already has CI scaffolding, but it still needs **strong neural baselines** and a **clean runtime path** before the CI contribution is defensible.

## Scope decision

Pick one primary scope and keep the thesis tight:

- **English-first**: `ModernBERT` + `DeBERTa-v3`
- **Multilingual-first**: `XLM-V` + `mDeBERTa-v3`

Do not try to prove English, multilingual, multimodal, ABSA, fuzzy logic, and LLM adaptation in one thesis. The clean thesis path is:

1. strong encoder baseline
2. domain adaptation on YouTube text
3. adaptive CI layer for calibration / abstention / robustness

## Target architecture

```text
YouTube comments + metadata
    -> dual preprocessing path
         -> classical cleaned text
         -> raw / lightly normalised transformer text
    -> strong encoder inference
         -> per-model probabilities + logits + entropy + margin
    -> calibration layer
         -> temperature scaling
         -> optional conformal / selective prediction
    -> adaptive CI layer
         -> per-sample routing / gating
         -> multi-objective optimisation
    -> persistence
         -> model provenance + split id + calibration + coverage metadata
    -> dashboard
         -> validated metrics, uncertainty, abstention, provenance
```

## Phase plan

### Phase 1 — Strong encoder baseline

Objective: replace the current headline baselines with a real transformer benchmark.

Deliverables:

- one English or multilingual encoder family selected as primary
- one held-out test benchmark that beats or matches the current `0.6946` macro-F1 baseline
- saved model artifact, config, and reproducibility log

Recommended initial models:

- English: `answerdotai/ModernBERT-base`, `microsoft/deberta-v3-base`
- Multilingual: `facebook/xlm-v-base`, `microsoft/mdeberta-v3-base`

### Phase 2 — Domain adaptation

Objective: adapt the encoder to unlabeled YouTube text before supervised fine-tuning.

Deliverables:

- domain-adaptive pretraining run on collected YouTube comments / titles / descriptions / transcripts
- comparison against the Phase 1 encoder without adaptation
- fixed train/val/test protocol preserved

### Phase 3 — Adaptive CI layer

Objective: make the thesis contribution about reliability and robustness, not raw voting.

Deliverables:

- temperature-scaled encoder outputs
- selective prediction or conformal confidence sets
- adaptive gate over strong encoders
- NSGA-II or PSO optimisation over macro-F1, calibration, coverage, and latency

### Phase 4 — Hard validation

Objective: convert a strong system into a defensible thesis result.

Deliverables:

- bootstrap confidence intervals
- McNemar or paired bootstrap significance tests
- cross-channel / cross-topic / temporal OOD evaluation
- human gold set evaluation
- ablation study on gate features and calibration layers

## File-by-file roadmap

### 1) `backend/src/sentiment/engines/transformer_engine.py`

Current state:

- generic Hugging Face sequence classifier wrapper
- still documented as a BERT/RoBERTa engine
- no explicit calibration or provenance support

Change:

- keep the file, but reposition it as the main **encoder runtime engine**
- add named presets for `modernbert`, `deberta_v3`, `xlm_v`, and `mdeberta_v3`
- return richer metadata per sample:
  - logits
  - entropy
  - top-2 margin
  - tokenizer/model artifact name
  - calibrated vs uncalibrated probabilities
- allow a saved temperature parameter or calibrator artifact to be loaded at inference time

Reason:

The repo already has one reusable transformer inference surface. Reuse it instead of creating parallel encoder runtimes.

### 2) `backend/src/sentiment/factory.py`

Current state:

- registry is classical-first
- modern encoder aliases are missing
- CI runtime entry points are not first-class

Change:

- add engine aliases:
  - `modernbert`
  - `deberta_v3`
  - `xlm_v`
  - `mdeberta_v3`
  - `adaptive_ci`
- make `transformer` accept a preset name instead of only a raw model path
- move classical models from “headline” to “baseline / fallback”

Reason:

This is the central routing layer. If it stays classical-first, the repo will remain classical-first.

### 3) `backend/app/youtube_preprocessor.py`

Current state:

- good spam / language filtering hooks
- current cleaning removes emojis, punctuation, hashtags, and non-English characters

Change:

- split preprocessing into two explicit paths:
  - `preprocess_for_classical(...)`
  - `preprocess_for_transformer(...)`
- keep raw text cues for transformers:
  - punctuation
  - repeated characters
  - emojis
  - casing signals
  - multilingual characters when multilingual mode is enabled
- emit lightweight feature metadata for CI gating:
  - language
  - character length
  - token length
  - emoji density
  - uppercase ratio
  - punctuation ratio

Reason:

The current regex cleaning is appropriate for TF-IDF models but removes useful signal for modern encoders and sarcasm-sensitive routing.

### 4) `backend/app/views.py`

Current state:

- request path already supports model selection, confidence stats, fuzzy config, and persistence
- analysis metadata is flexible JSON, which is useful

Change:

- add request-level options for the new runtime:
  - `sentiment_model=modernbert|deberta_v3|xlm_v|mdeberta_v3|adaptive_ci`
  - `calibration_profile`
  - `allow_abstain`
  - `coverage_target`
  - `language_mode`
- when using strong encoders, use the transformer preprocessing path
- standardise `analysis_meta` keys so results are thesis-grade and queryable:
  - `model_family`
  - `model_artifact`
  - `dataset_split_id`
  - `training_run_id`
  - `calibration`
  - `selective_prediction`
  - `ece`
  - `brier`
  - `abstained_count`
  - `coverage`
  - `latency_ms`
- reject client-supplied “model comparison” payloads as evidence unless generated by backend evaluation code

Reason:

This file is the production analysis path. It must expose the same model family and reliability story that the thesis reports.

### 5) `backend/app/models.py`

Current state:

- `YouTubeAnalysis.analysis_meta` is flexible enough for early iteration
- there are no first-class provenance or reliability fields

Change:

Short-term:

- keep using `analysis_meta`, but define a stable schema for:
  - model artifact/version
  - split provenance
  - calibration method and parameters
  - abstention counts and coverage
  - CI gate configuration

Long-term, if dashboard filtering/querying becomes important:

- promote the following to columns in a migration:
  - `model_family`
  - `model_artifact`
  - `dataset_split_id`
  - `ece`
  - `brier`
  - `coverage`
  - `abstained_count`

Reason:

For the thesis, provenance must be reproducible before it is queryable. JSON is acceptable if the schema is fixed and documented.

### 6) `backend/scripts/prepare/prepare_hf_dataset.py`

Current state:

- this is already the strongest methodological part of the repo
- it handles label conflict removal, dedupe, and group-aware splitting

Change:

- emit both:
  - raw transformer text
  - classical-cleaned text
- preserve metadata columns needed for domain-shift experiments:
  - `VideoID`
  - `channel_id`
  - `published_at`
  - language flag
- optionally export:
  - train/val/test CSV for classical pipelines
  - Hugging Face dataset files for encoder training

Reason:

One split pipeline should feed both the classical baseline and the encoder path so the comparison stays fair.

### 7) `backend/research/ci/temperature_scaling.py`

Current state:

- already usable
- assumes the current model list

Change:

- make the script model-agnostic
- support encoder probability exports, not only live scoring
- save a small calibration artifact per model:
  - temperature
  - validation metrics
  - fitted date
  - split id

Reason:

This script should become the canonical post-hoc calibration step between model training and deployment.

### 8) `backend/research/ci/multi_objective_ensemble.py`

Current state:

- already has the right optimisation framing
- currently optimises global weights across `logreg`, `svm`, `tfidf`

Change:

- reuse the same NSGA-II machinery but switch inputs from classical-only probability cubes to strong encoder probability cubes
- add latency as an explicit objective
- optionally add selective-coverage utility instead of plain coverage
- report Pareto solutions for:
  - encoder-only ensemble
  - encoder + fallback classical model

Reason:

The NSGA-II formulation is already one of the most thesis-worthy parts of the repo. The main issue is the weak base model set.

### 9) `backend/research/ci/neuro_fuzzy_gate.py`

Current state:

- already implements a per-sample adaptive gate
- gating is based only on confidence values from classical models

Change:

- extend the gate feature set beyond confidence:
  - entropy
  - top-2 margin
  - model disagreement
  - language
  - text length
  - emoji density
  - uppercase ratio
  - punctuation ratio
  - optional channel/topic metadata
- compare three gates:
  - confidence-only fuzzy gate
  - richer fuzzy gate
  - lightweight learned gate without fuzzy logic

Reason:

This is where the CI contribution becomes defendable. The current version is a good starting point, not the final thesis model.

### 10) `backend/research/ci/entropy_gated_prediction.py`

Current state:

- already implements abstention/cascade logic from entropy

Change:

- generalise it to encoder ensembles
- add conformal or threshold-calibrated abstention
- output:
  - risk-coverage curve
  - AURC
  - coverage-accuracy curve
  - abstention confusion analysis

Reason:

Selective prediction is the cleanest bridge between CI and deployment reliability.

### 11) Add `backend/research/transformers/`

Add a new subpackage instead of mixing encoder training into classical scripts.

Recommended files:

- `backend/research/transformers/train_encoder.py`
  - supervised fine-tuning entry point
- `backend/research/transformers/domain_adapt_encoder.py`
  - DAPT/TAPT on unlabeled YouTube text
- `backend/research/transformers/export_prob_cube.py`
  - exports per-sample probabilities/logits for CI modules
- `backend/research/transformers/model_registry.py`
  - preset names, tokenizer config, label mapping, default max length

Reason:

The repo has a research layer already. Encoder work should live beside it, not be hidden inside the API code.

### 12) Add `backend/research/evaluation/` utilities for Route A

Some evaluation utilities already exist. Add missing thesis-critical scripts:

- `backend/research/evaluation/domain_shift.py`
  - cross-channel, cross-topic, temporal evaluation
- `backend/research/evaluation/reliability_diagrams.py`
  - calibration plots and tables
- `backend/research/evaluation/conformal.py`
  - conformal prediction / set-valued evaluation
- `backend/research/evaluation/human_gold_analysis.py`
  - gold-set agreement and error slices

Reason:

The thesis argument will fail on validation if these checks stay ad hoc.

### 13) `frontend/src/Views/Pages/Search.jsx`

Current state:

- the UI presents classical, ensemble, fuzzy, and meta-learner options as the main choices

Change:

- make the main choices:
  - `ModernBERT`
  - `DeBERTa-v3`
  - `XLM-V` / `mDeBERTa-v3` if multilingual
  - `Adaptive CI Ensemble`
- move `logreg`, `svm`, `tfidf`, and `fuzzy_ensemble` under a “baseline / research legacy” section
- expose reliability controls only when relevant:
  - confidence threshold
  - abstention
  - calibration profile

Reason:

The frontend should reflect the actual thesis system hierarchy.

### 14) `frontend/src/Views/Pages/Dashboard.jsx`

Current state:

- the dashboard mixes exploratory visuals with experimental metadata
- the summary cards are sentiment counts, not thesis-grade model reporting

Change:

- make the top summary thesis-focused:
  - model family
  - model artifact
  - split id
  - total analyzed
  - ECE
  - Brier
  - coverage
  - abstained count
- add a dedicated provenance / uncertainty section
- de-emphasise word-cloud style views
- rename the aspect section unless proper ASTE is implemented

Reason:

The dashboard should present validated evidence, not just descriptive charts.

## Experiment matrix

Use one fixed split and compare incrementally.

### Baselines

1. `logreg`
2. best existing classical ensemble or meta-learner
3. encoder A
4. encoder B
5. encoder A + temperature scaling
6. encoder B + temperature scaling

### CI models

7. static encoder ensemble
8. NSGA-II encoder ensemble
9. entropy-gated selective predictor
10. adaptive fuzzy gate
11. adaptive learned gate without fuzzy logic

### Validation slices

Run each of the above on:

- in-domain held-out test
- cross-channel split
- temporal split
- human gold set
- optional multilingual slice if multilingual scope is selected

## Minimum success criteria

Do not claim the route worked unless all of the following are true:

1. A strong encoder baseline is reproducible and stored
2. The CI layer is compared against that encoder baseline, not only against classical models
3. Calibration improves materially (`ECE`, `Brier`, or both)
4. Coverage / abstention trade-offs are reported
5. Domain-shift performance is measured
6. Statistical significance or confidence intervals are reported

## Recommended implementation order

Follow this order to minimise rework:

1. fix preprocessing split between classical and transformer paths
2. add encoder presets to the runtime factory
3. add supervised encoder training
4. add temperature scaling artifacts
5. export encoder probability cubes
6. upgrade NSGA-II and adaptive gate modules to encoder inputs
7. add selective prediction / conformal evaluation
8. surface provenance and uncertainty in the dashboard

## What not to do

- Do not lead the thesis with the current fuzzy grid-search result
- Do not compare models trained on different preprocessing without stating it
- Do not call the current keyword aspect module “ABSA”
- Do not claim generic “state-of-the-art”; claim benchmark-scoped improvement
- Do not add multimodal scope until the text-only route is validated

## Primary references

- ModernBERT: https://aclanthology.org/2025.acl-long.127/
- DeBERTa-v3 / mDeBERTa-v3: https://arxiv.org/abs/2111.09543
- XLM-V: https://aclanthology.org/2023.emnlp-main.813/
- Domain-adaptive pretraining: https://arxiv.org/abs/2004.10964
- Temperature scaling: https://proceedings.mlr.press/v70/guo17a.html
- Selective classification: https://proceedings.mlr.press/v130/gangrade21a.html

## Immediate next step

If you only do one implementation step next, do this:

1. add a dual preprocessing path
2. fine-tune one strong encoder on the existing leakage-safe split
3. save uncalibrated and calibrated test-set metrics as the new thesis baseline
