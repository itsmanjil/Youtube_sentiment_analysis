# Incomplete Work Audit — YouTube Sentiment Analysis

**Audit date:** 2026-06-02 (regenerated)
**Project:** YouTube sentiment analysis thesis (Django backend + React frontend + ML research pipeline)

> This file was regenerated after the gold-set annotation, live-runtime significance
> testing, and environment-pinning work were completed. The prior version (dated
> 2026-05-21) is superseded; items it listed as **Blocked** are now resolved (see below).

---

## What Is Complete

- All P0 thesis checklist items — backend passes its test suite, frontend passes its suite
- Classical ML pipeline (LogReg, SVM, TF-IDF, meta-learner, PSO/NSGA-II ensemble)
- Full CI research layer: temperature scaling, neuro-fuzzy gate, selective prediction, entropy gating, Pareto front
- Dual preprocessing path (`preprocess_for_classical` / `preprocess_for_transformer`)
- Leakage audit, provenance schema, domain-shift evaluation
- Threats to validity, ethics section, thesis abstract, viva defense brief
- **Human gold set — COMPLETE** (see below)
- **Live-runtime significance testing — COMPLETE** (see below)
- **Environment pinning — COMPLETE** (see below)

---

## Recently Resolved (since the 2026-05-21 audit)

### 1. Human Gold Set — RESOLVED ✅ (was P1 "Blocked")

The 300-item gold set is now fully and independently annotated:

| File | Rows | Labeled | Status |
|------|------|---------|--------|
| `gold_set_annotator_1.csv` | 300 | 300 | Complete |
| `gold_set_annotator_2.csv` | 300 | 300 | Complete |
| `gold_set_human_reconciled.csv` | 300 | 300 | Complete (gold_label + is_disputed) |

Inter-annotator agreement (`results/gold_set/iaa_report.md`): **Krippendorff's α = 0.9547,
Cohen's/Fleiss' κ = 0.9546, 97.0% percent agreement, 9 disputed items excluded.** The
gold-set model evaluation (`results/gold_set/gold_set_evaluation.md`) now runs against the
reconciled **human** labels, replacing the earlier circular silver-label result (where
`ensemble_pso` trivially scored 1.000 F1 because it generated the labels). Against human
labels the ensemble scores ~0.70 F1 — a credible, non-circular figure.

The thesis Chapter 3 §3, Chapter 4 §4.7, and Appendix Table 13 have been updated to report
this as a **Supported** claim. (Verify a final read-through after any further edits.)

### 2. Live-Runtime Significance Testing — RESOLVED ✅ (was a documented gap)

The earlier draft stated that no paired significance test had been computed for the pinned
live `ensemble_nsga2` variant. This is now done:
`results/runtime/route_a_live_v1/live_significance_tests.{md,json}`
(script: `research/ci/live_significance_tests.py`). Highlights (n = 165,110; 2,000-resample
paired bootstrap, seed 42; Holm-adjusted McNemar):

- Every reconstructed model validates **exactly** against the pinned benchmark.
- NSGA-II ensemble ECE vs meta-learner: −0.0111, 95% CI [−0.0126, −0.0084] (**excludes 0**).
- NSGA-II vs meta-learner macro-F1: tied (CI [−0.0002, +0.0014]).
- NSGA-II ECE vs logistic regression: tied (CI [−0.0021, +0.0031]).
- Meta-learner vs logistic regression macro-F1: +0.0017 (CI [+0.0009, +0.0025], significant).

Thesis §4.6 and §4.5 (Discussion) and the Chapter 4/5 limitations have been updated to report
the NSGA-II calibration advantage as significance-backed rather than descriptive.

### 3. Environment Pinning — RESOLVED ✅

- `backend/requirements.txt` is now pinned to `backend/Pipfile.lock` versions
  (scikit-learn 1.8.0, numpy 1.26.4, pandas 3.0.0, scipy 1.17.0, Django 5.2.10, …).
- `.python-version` corrected from `3.8.18` to `3.11`, consistent with the README ("3.11+")
  and `Pipfile` (`python_version = "3.11"`).

---

## Still Open (non-blocking)

### A. Two evaluation scripts from the Route A roadmap are still absent

`research/evaluation/` contains `ablation.py`, `calibration.py`, `confusion_matrices.py`,
`domain_shift.py`, `reliability_diagrams.py`, `roc_auc.py`, `statistical_tests.py`.
`reliability_diagrams.py` is done (`backend/figures/reliability_diagrams.png`,
used in the docx as Figure 8). Still missing:

| Script | Purpose | Status |
|--------|---------|--------|
| `conformal.py` | Conformal prediction / set-valued evaluation | **Missing** (optional) |
| `human_gold_analysis.py` | Gold-set error slices | **Missing** (gold set now exists, so this is now feasible) |

### B. `app_api/models.py` is an empty stub (Low)

Contains only a placeholder comment. The `app_api` app has a working JWT view and tests but
no models. Either populate or remove.

### C. `backend/tests/` directory tree is empty (Low)

`tests/unit`, `tests/integration`, `tests/fixtures` exist but contain no files. All real
tests live in `app/tests.py`, `app_api/tests.py`, `users/tests.py`. Either populate or remove
the empty scaffold.

### D. Transformer Route A — future work (Medium, intentional)

Only a smoke/CPU DeBERTa-v3 benchmark exists; no full-dataset fine-tuned transformer result is
claimed, and `ROUTE_A_ENCODER_POSITION.md` flags this as future work. The deployed headline is
deliberately classical-and-ensemble-first. This is stated honestly in the thesis and is not a
blocker, but it remains the main avenue for future contribution.

### E. Local cache artifacts (cosmetic)

Stray `*.cpython-314.pyc` files exist under `__pycache__/` directories from an earlier Python
3.14 run. They are already covered by `.gitignore` and are **not** git-tracked, so they will
not be committed; delete locally with
`find . -path '*__pycache__*' -name '*.pyc' -delete` if desired.

---

## Summary Table

| Item | Location | Severity | Status |
|------|----------|----------|--------|
| Human gold set + IAA | `data/gold_set_*`, `results/gold_set/` | was High | **Resolved** |
| Live-runtime significance tests | `results/runtime/route_a_live_v1/live_significance_tests.*` | was Medium | **Resolved** |
| Dependency pinning + Python version | `requirements.txt`, `.python-version` | was Medium | **Resolved** |
| Missing `reliability_diagrams.py` / `conformal.py` / `human_gold_analysis.py` | `research/evaluation/` | Low–Medium | Open |
| `app_api/models.py` empty | `backend/app_api/models.py` | Low | Open |
| `backend/tests/` empty | `backend/tests/` | Low | Open |
| Transformer Route A (smoke only) | `results/deberta_v3_*` | Medium | Future work (intentional) |

---

## Recommended Finish Order

1. **Final thesis text read-through** — confirm every gold-set/IAA and significance reference
   is internally consistent after the recent edits (the contradictions have been fixed; this is
   a verification pass).
2. **Add `reliability_diagrams.py`** and drop one reliability diagram into Chapter 4 §4.5.
3. **(Optional) `conformal.py` and `human_gold_analysis.py`** to deepen the uncertainty/gold-set
   analysis.
4. **Clean up dead stubs** (`app_api/models.py`, empty `tests/`).
