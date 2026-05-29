# Incomplete Work Audit — YouTube Sentiment Analysis

**Audit date:** 2026-05-21  
**Project:** YouTube sentiment analysis thesis (Django backend + React frontend + ML research pipeline)

---

## What Is Complete

Before listing gaps, here is what the project has firmly finished:

- All P0 thesis checklist items — backend passes 40/40 tests, frontend passes 81/81 tests
- Classical ML pipeline (LogReg, SVM, TF-IDF, meta-learner, PSO/NSGA-II ensemble)
- Full CI research layer: temperature scaling, neuro-fuzzy gate, selective prediction, entropy gating, Pareto front
- DeBERTa-v3 model artifact saved (`backend/models/transformers/deberta_v3/`)
- Dual preprocessing path (`preprocess_for_classical` / `preprocess_for_transformer` in `youtube_preprocessor.py`)
- Transformer factory aliases added (`modernbert`, `deberta_v3`, `xlm_v`, `mdeberta_v3`)
- `views.py` handles transformer model routing
- `Search.jsx` exposes transformer model options in the UI
- All 6 "Should-do" deliverables completed (effect sizes, error characterization, fold variance, seed sensitivity, preprocessing ablation, Pareto SVGs)
- Leakage audit, provenance schema, domain shift evaluation
- Threats to validity, ethics section, thesis abstract, viva defense brief

---

## Incomplete Work

### 1. Human Gold Set — BLOCKED (P1, last critical thesis gap)

**Location:** `backend/data/gold_set_annotator_*.csv`

The gold set tooling is fully built, but the actual human annotation has barely begun:

| File | Rows | Labeled | Status |
|------|------|---------|--------|
| `gold_set_annotator_1.csv` | 300 | **4** | Annotation barely started |
| `gold_set_annotator_2.csv` | 300 | **0** | Completely empty |
| `gold_set_human_reconciled.csv` | — | — | Does not exist yet |

Because of this, the current gold set evaluation (`results/gold_set/gold_set_evaluation.md`) runs against **silver labels** (auto-annotated by the PSO ensemble), not real human IAA evidence. The result is misleading — `ensemble_pso` scores 1.000 F1 because it was used to generate the labels it is being tested against.

The `THESIS_CLAIM_ARTIFACT_AUDIT.md` explicitly marks both **"Human-level sentiment accuracy"** and **"Inter-annotator agreement"** as `Blocked`.

**What needs to be done:**
```bash
cd backend
python scripts/annotate.py --input data/gold_set_template.csv --output data/gold_set_annotator_1.csv
python scripts/annotate.py --input data/gold_set_template.csv --output data/gold_set_annotator_2.csv
python scripts/prepare/merge_annotations.py \
    --annotator_a data/gold_set_annotator_1.csv \
    --annotator_b data/gold_set_annotator_2.csv \
    --output data/gold_set_human_reconciled.csv
python research/ci/gold_set_evaluation.py
```

---

### 2. Three Missing Evaluation Scripts (from Route A Roadmap)

**Location:** `backend/research/evaluation/`

The `ROUTE_A_IMPLEMENTATION_ROADMAP.md` lists these as required for full thesis validation. All three are absent:

| Script | Purpose | Status |
|--------|---------|--------|
| `reliability_diagrams.py` | Calibration plots and tables for the thesis | **Missing** |
| `conformal.py` | Conformal prediction / set-valued evaluation | **Missing** |
| `human_gold_analysis.py` | Gold-set agreement and error slices | **Missing** |

The existing `evaluation/` folder only has: `ablation.py`, `calibration.py`, `domain_shift.py`, `statistical_tests.py`.

---

### 3. `app_api/models.py` Is Empty

**Location:** `backend/app_api/models.py`

The file contains only a placeholder comment and no model definitions. The `app_api` Django app has a working JWT token view and tests, but the models file was never populated. This is not a thesis-blocker but is a dead stub that should either be filled in or intentionally removed.

---

### 4. `backend/tests/` Directory Is Completely Empty

**Location:** `backend/tests/integration/` and `backend/tests/unit/`

Both subdirectories exist but contain no Python test files. All backend tests currently live in:
- `backend/app/tests.py` (1,018 lines, 40 tests)
- `backend/app_api/tests.py` (JWT tests)
- `backend/users/tests.py` (user profile + auth tests)

The `tests/` folder structure was set up but never populated. Not a blocker, but the empty directory is misleading.

---

### 5. Route A Transformer Pipeline — Partially Done, Not Fully Validated

**Status:** DeBERTa-v3 artifact exists (smoke/CPU run only). Full thesis-grade evaluation is flagged as future work.

Specifically:

- Only a **smoke/CPU benchmark** of DeBERTa-v3 has been run — the result files are `deberta_v3_smoke_metrics.json` and `deberta_v3_benchmark_cpu_metrics.json`
- No full-dataset (10k+) transformer training benchmark exists
- ModernBERT, XLM-V, and mDeBERTa-v3 have **no saved model artifacts** — only DeBERTa-v3 is saved
- `ROUTE_A_ENCODER_POSITION.md` explicitly states: *"Route A encoder work is implemented but should remain future work unless rerun with `transformers`, `torch`, and suitable compute"*
- The `THESIS_CLAIM_ARTIFACT_AUDIT.md` marks "Transformer-first Route A superiority" as **Future work**

The CI modules (NSGA-II, neuro-fuzzy gate) have not been upgraded to use encoder probability cubes — they still operate on classical model outputs as documented in the roadmap.

---

### 6. Final Thesis Text Pass Not Confirmed

The `THESIS_FINAL_CHECKLIST.md` lists as its **Suggested Finish Order item #2**:

> *"Do a final thesis-text pass against the pinned runtime artifacts."*

The `thesis.docx` file is present in the repo root but there is no checklist item marking this pass as done. Given the gold set is incomplete and the human IAA evidence is missing, any thesis chapter referencing gold set validation will need updating once annotation is complete.

---

## Summary Table

| Item | Location | Severity |
|------|----------|----------|
| Human gold set annotation (4/300 done, annotator 2 empty) | `data/gold_set_annotator_*.csv` | **High** — explicitly Blocked in audit |
| Missing: `reliability_diagrams.py`, `conformal.py`, `human_gold_analysis.py` | `research/evaluation/` | Medium — listed in roadmap |
| `app_api/models.py` empty | `backend/app_api/models.py` | Low — dead stub |
| `backend/tests/` directory empty | `backend/tests/` | Low — structural gap |
| Transformer Route A not fully validated (smoke only, no full run) | `results/deberta_v3_*` | Medium — documented as future work |
| Final thesis text pass against pinned artifacts | `thesis.docx` | Medium — depends on gold set completion |

---

## Recommended Finish Order

1. **Complete human gold set annotation** — annotate both CSV files, run merge + IAA, re-run gold set evaluation with real human labels. This is the only P1 item still blocked.
2. **Do final thesis text pass** — once gold set is done, update any chapter referencing it and verify all claims in `THESIS_CLAIM_ARTIFACT_AUDIT.md` are met.
3. **Add missing evaluation scripts** — `reliability_diagrams.py` and `conformal.py` if time allows; `human_gold_analysis.py` is needed after gold set is complete anyway.
4. **Clean up dead stubs** (`app_api/models.py`, empty `tests/` folder) — low effort, keeps the repo tidy.
