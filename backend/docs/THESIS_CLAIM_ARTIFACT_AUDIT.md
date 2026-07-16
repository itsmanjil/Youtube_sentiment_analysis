# Thesis Claim Artifact Audit

Status date: 2026-07-11 (re-verified after the 2026-07-10 defensibility-pass
fixes and the 2026-07-11 `calibration_applied` fix; the two rows that cited
pre-fix numbers were updated to the regenerated artifacts)

This audit maps the final defensible thesis claims to runnable scripts or stored
artifacts in the repository. Claims not backed by an artifact are marked as
blocked or future work.

## Supported Claims

| Claim | Evidence | Reproduction |
| --- | --- | --- |
| The final runtime is artifact-pinned and auditable, including the trained model binaries (not just calibration/ensemble/fuzzy configs). | `results/runtime/route_a_live_v1/manifest.json` (SHA-256 for temperature scaling, PSO/NSGA-II weights, neuro-fuzzy gate, and the logreg/svm/tfidf/meta_learner model + vectorizer binaries), `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md` | `python research/ci/live_runtime_benchmark.py --data data/test.csv --text_column text --label_column label` |
| Live runtime labels match offline probability-cube labels for checked models. | `results/runtime/route_a_live_v1/prediction_level_reconciliation.md` | `python research/ci/prediction_level_reconciliation.py --models logreg,svm` |
| Calibration claims are model-specific, not universal. | `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`, `docs/LIVE_CI_WIRING_AUDIT.md` | Review runtime benchmark calibration fields and wiring audit. |
| NSGA-II ensemble improves calibration at no accuracy cost vs baseline. | `results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md` (ECE 0.0039 vs meta 0.0183, post-fix re-run of 2026-07-10/11) | `python research/ci/live_runtime_benchmark.py --data data/test.csv` |
| ROC-AUC (OvR) computed for all main models; Neutral is weakest class. | `results/roc_auc/roc_auc.md`, `results/roc_auc/roc_auc.json` | `python research/evaluation/roc_auc.py --test data/test.csv --sample 5000` |
| Full confusion matrices + per-class P/R/F1 for all main models. | `results/confusion_matrices/confusion_matrices.md` | `python research/evaluation/confusion_matrices.py --test data/test.csv --sample 5000` |
| Exploratory data analysis (class/length/lexical/metadata distributions). | `results/eda/eda_report.md`, `results/eda/eda_report.json` | `python research/analysis/eda_report.py --test data/test.csv --sample 50000` |
| Neutral-class weakness characterised + intervention tested honestly. | `results/neutral_analysis/neutral_analysis.md` | `python research/analysis/neutral_class_analysis.py --model logreg --sample 8000` |
| The neuro-fuzzy gate changes the base classifier's (logreg's) argmax on 2.74% of comments (1,096/40,000: 456 corrections, 412 regressions, 228 wrong-to-wrong flips) — a small net-positive edit rate. The earlier "0.18%, fuzzy_ensemble/tfidf parity" reading was an artifact of two since-fixed bugs (THESIS_VIVA_DEFENSE_BRIEF.md §0). | `results/neuro_fuzzy_gate_ablation/fuzzy_gate_ablation.md` (regenerated 2026-07-10) | `python research/ci/fuzzy_gate_ablation.py --sample 40000 --seed 42 --base_model logreg` |
| The thesis includes real metadata-backed domain-slice evidence. | `results/domain_shift/category_domain_shift.md`, `results/domain_shift/country_domain_shift.md` | `python research/evaluation/domain_shift.py --data data/route_a_domain_10k/test.csv --slice_column CategoryID` |
| The checked benchmark split has no exact cross-split duplicates and has reviewed near-duplicate candidates. | `results/leakage/near_duplicate_audit.md` | `python scripts/prepare/near_duplicate_audit.py --split_dir data/route_a_benchmark_cpu` |
| The frontend surfaces confidence, uncertainty, and calibration metadata. | `frontend/src/Views/Pages/Dashboard.test.jsx`, `frontend/src/Views/Pages/Monitoring.test.jsx` | `cd frontend && npm test -- --run` |
| Selective prediction / abstention has a runnable validation artifact on the real held-out test split. | `results/route_a_live_v1_ci/coverage_accuracy_curve.md` (20,000-comment sample of the real 165,110-row test split; supersedes the earlier 180-comment smoke-split table) | `python research/ci/coverage_accuracy_curve.py --test data/test.csv --sample 20000 --points 20 --output results/route_a_live_v1_ci` |
| Human-level gold-set evaluation with strong inter-annotator agreement. | `results/gold_set/gold_set_evaluation.md`, `results/gold_set/iaa_report.md` (α = 0.9547) | `python research/ci/gold_set_evaluation.py` |
| Gold-set metrics are not inflated by training-split overlap. | `data/gold_set_split_membership.csv`, `results/gold_set/gold_set_evaluation_holdout.md` | `python research/ci/gold_set_train_membership.py && python research/ci/gold_set_evaluation_holdout.py` |
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
| Transformer-first Route A superiority. | Future work | Full encoder training/evaluation with `transformers` and `torch` installed, ideally on GPU. |
| `hybrid_dl` calibration. | Future work | A trained hybrid DL checkpoint plus validation logits/probabilities for calibration fitting. |
| Full ABSA. | Out of scope | Current implementation is keyword-level aspect sentiment, not full aspect-based sentiment analysis. |

Human-level sentiment accuracy on an independently labeled gold set, and
inter-annotator agreement from independent humans, were previously listed here
as Blocked. Both are now resolved and appear in Supported Claims above
(α = 0.9547, 97.0% agreement, two independent annotators plus reconciliation).

