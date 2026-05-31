# Thesis Claim Artifact Audit

Status date: 2026-05-31 (updated)

This audit maps the final defensible thesis claims to runnable scripts or stored
artifacts in the repository. Claims not backed by an artifact are marked as
blocked or future work.

## Supported Claims

| Claim | Evidence | Reproduction |
| --- | --- | --- |
| The final runtime is artifact-pinned and auditable. | `results/runtime/route_a_live_v1/manifest.json`, `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md` | `python research/ci/live_runtime_benchmark.py --data data/test.csv --text_column text --label_column label` |
| Live runtime labels match offline probability-cube labels for checked models. | `results/runtime/route_a_live_v1/prediction_level_reconciliation.md` | `python research/ci/prediction_level_reconciliation.py --models logreg,svm` |
| Calibration claims are model-specific, not universal. | `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`, `docs/LIVE_CI_WIRING_AUDIT.md` | Review runtime benchmark calibration fields and wiring audit. |
| NSGA-II ensemble improves calibration at no accuracy cost vs baseline. | `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md` (ECE 0.0046 vs meta 0.0157) | `python research/ci/live_runtime_benchmark.py --data data/test.csv` |
| ROC-AUC (OvR) computed for all main models; Neutral is weakest class. | `results/roc_auc/roc_auc.md`, `results/roc_auc/roc_auc.json` | `python research/evaluation/roc_auc.py --test data/test.csv --sample 5000` |
| Full confusion matrices + per-class P/R/F1 for all main models. | `results/confusion_matrices/confusion_matrices.md` | `python research/evaluation/confusion_matrices.py --test data/test.csv --sample 5000` |
| Exploratory data analysis (class/length/lexical/metadata distributions). | `results/eda/eda_report.md`, `results/eda/eda_report.json` | `python research/analysis/eda_report.py --test data/test.csv --sample 50000` |
| Neutral-class weakness characterised + intervention tested honestly. | `results/neutral_analysis/neutral_analysis.md` | `python research/analysis/neutral_class_analysis.py --model logreg --sample 8000` |
| The thesis includes real metadata-backed domain-slice evidence. | `results/domain_shift/category_domain_shift.md`, `results/domain_shift/country_domain_shift.md` | `python research/evaluation/domain_shift.py --data data/route_a_domain_10k/test.csv --slice_column CategoryID` |
| The checked benchmark split has no exact cross-split duplicates and has reviewed near-duplicate candidates. | `results/leakage/near_duplicate_audit.md` | `python scripts/prepare/near_duplicate_audit.py --split_dir data/route_a_benchmark_cpu` |
| The frontend surfaces confidence, uncertainty, and calibration metadata. | `frontend/src/Views/Pages/Dashboard.test.jsx`, `frontend/src/Views/Pages/Monitoring.test.jsx` | `cd frontend && npm test -- --run` |
| Selective prediction / abstention has a runnable validation artifact. | `results/route_a_benchmark_cpu_ci/coverage_accuracy_curve.md`, `results/route_a_benchmark_cpu_ci/entropy_gated_prediction.md` | `python research/ci/coverage_accuracy_curve.py --test data/route_a_benchmark_cpu/test.csv --sample 180 --points 20 --output results/route_a_benchmark_cpu_ci` |
| Human-level gold-set evaluation with strong inter-annotator agreement. | `results/gold_set/gold_set_evaluation.md`, `results/gold_set/iaa_report.md` (α = 0.9547) | `python research/ci/gold_set_evaluation.py` |
| The app build is split into route and vendor chunks. | `frontend/src/App.jsx`, `frontend/vite.config.mjs` | `cd frontend && npm run build` |

## Thesis-Section to Artifact Map

| Thesis section | Document | Backing artifact(s) |
| --- | --- | --- |
| Ch.1 Research questions & framing | `docs/THESIS_CHAPTER1_RESEARCH_QUESTIONS.md` | — (framing) |
| Ch.2 Literature review | `docs/THESIS_LITERATURE_REVIEW.md` | — (citations) |
| Ch.3 Data provenance | `docs/LABEL_PROVENANCE.md`, `data/split_metadata.json` | dataset card, split metadata |
| Ch.3 EDA | `docs/THESIS_EDA.md` | `results/eda/` |
| Ch.4 Headline performance | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.1 | `results/runtime/route_a_live_v1/` |
| Ch.4 ROC-AUC | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.2 | `results/roc_auc/` |
| Ch.4 Confusion matrices | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.3 | `results/confusion_matrices/` |
| Ch.4 Selective prediction | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.4 | `results/route_a_benchmark_cpu_ci/` |
| Ch.4 Gold set | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.7 | `results/gold_set/` |
| Ch.4 Neutral analysis | `docs/THESIS_EVALUATION_CONSOLIDATED.md` §4.8 | `results/neutral_analysis/` |

## Blocked Or Future Claims

| Claim | Status | Required Input |
| --- | --- | --- |
| Human-level sentiment accuracy on an independently labeled gold set. | Blocked | At least one real human annotation pass, preferably two annotators plus adjudication. |
| Inter-annotator agreement from independent humans. | Blocked | Two independent annotation CSVs and merge/adjudication output. |
| Transformer-first Route A superiority. | Future work | Full encoder training/evaluation with `transformers` and `torch` installed, ideally on GPU. |
| `hybrid_dl` calibration. | Future work | A trained hybrid DL checkpoint plus validation logits/probabilities for calibration fitting. |
| Full ABSA. | Out of scope | Current implementation is keyword-level aspect sentiment, not full aspect-based sentiment analysis. |

## Human Gold-Set Command Path

The repo has the tooling, but the labels must come from humans:

```bash
cd backend
python scripts/annotate.py --input data/gold_set_template.csv --output data/gold_set_annotator_1.csv
python scripts/annotate.py --input data/gold_set_template.csv --output data/gold_set_annotator_2.csv
python scripts/prepare/merge_annotations.py --annotator_a data/gold_set_annotator_1.csv --annotator_b data/gold_set_annotator_2.csv --output data/gold_set_human_reconciled.csv
python research/ci/gold_set_evaluation.py
```

