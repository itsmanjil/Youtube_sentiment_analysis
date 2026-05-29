# Provenance Schema

Status date: 2026-05-18

The project uses a lightweight provenance schema for thesis-facing artifacts.
This is not a database migration; it is the stable contract for result files and
runtime audit documents.

## Required Fields

Every thesis-facing result artifact should expose these fields directly or via a
nearby manifest:

| Field | Meaning | Example |
| --- | --- | --- |
| `created_at` or `created_at_utc` | UTC generation time | `2026-05-18T01:47:02Z` |
| `dataset_path` or `split_dir` | Input dataset or split directory | `data/route_a_benchmark_cpu/test.csv` |
| `models` | Models evaluated | `["logreg", "svm", "ensemble_nsga2"]` |
| `n_samples` | Number of evaluated samples | `180` |
| `metrics` or `results` | Main evaluation payload | Accuracy, macro-F1, ECE, AUCA |
| `interpretation` | Thesis-facing limitation or finding | Metadata-backed domain slices by CategoryID/CountryCode |

## Runtime Provenance

The pinned runtime uses:

- `results/runtime/route_a_live_v1/manifest.json`
- `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.json`
- `results/runtime/route_a_live_v1/prediction_level_reconciliation.json`

These files are the source of truth for runtime claims.

## Research Provenance

Research artifacts should keep their command path in `backend/research/README.md`
or in `backend/docs/THESIS_CLAIM_ARTIFACT_AUDIT.md`. New thesis claims should
not be added unless they can be mapped to one of those command paths.

