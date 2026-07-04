# Threats to Validity

Status date: 2026-07-02 (updated)

For a master's thesis, the main risk is not missing models but threats to
validity: leakage, train/inference skew, non-reproducible preprocessing, and
evaluation mistakes that inflate results. This chapter states the principal
threats and the mitigations applied.

## Data Leakage

Exact-duplicate or near-duplicate texts appearing across train/val/test can
inflate test metrics. Mitigations applied: a leakage-checking script
(`scripts/prepare/check_split_leakage.py`), a split builder
(`scripts/prepare/prepare_hf_dataset.py`) that drops conflicting-label texts
and de-duplicates by final model-input text before splitting, and a
group-aware split by VideoID that reduces within-video topical leakage; split
provenance is saved to `split_metadata.json`.

**Residual gap:** the current mitigation is exact-text de-duplication. A
near-duplicate audit (`scripts/prepare/near_duplicate_audit.py`,
MinHash/SimHash) has been run, but only on the 810-row
`route_a_benchmark_cpu` smoke split (9 near-duplicate cross-split pairs found,
status REVIEW) — it has **not** been run on the full 810,850-row corpus used
for the headline benchmark. Near-duplicate leakage across the full train/val/test
split is therefore an open, unquantified risk and is reported as a limitation
rather than a closed item; extending the audit to the full split is
recommended future work.

## Train/Inference Preprocessing Skew

If training data is raw but the serving API cleans aggressively (or vice
versa), evaluation becomes misleading. Production API preprocessing
(`app/youtube_preprocessor.py`) is shared with dataset preparation via an
explicit flag so the same pipeline can be applied at both stages. The residual
requirement is discipline: if classical preprocessing is enabled during
training it must also be enabled during inference, or skew is reintroduced.

## Label Noise / Construct Validity

Public sentiment datasets contain sarcasm, topic-dependent sentiment words,
ambiguous examples, and inconsistent label definitions. Conflicting-label
texts are dropped during preparation. The principal mitigation is the human
gold set with chance-corrected agreement (Krippendorff's α = 0.9547), which
lets the thesis report human-grounded performance and bound the share of
measured error attributable to the automated labelling scheme. The gold set
was originally sampled from the training split rather than the held-out test
split; a post-hoc membership audit
(`research/ci/gold_set_train_membership.py`) found 95/300 items are exact-text
training-split members, and a held-out-only re-evaluation
(`results/gold_set/gold_set_evaluation_holdout.md`) shows no material change
in the headline gold-set figures, so this does not appear to inflate the
reported numbers.

## Evaluation Methodology / Conclusion Validity

Common thesis-killers are tuning on the test set, reporting a single split
without uncertainty, claiming significance without tests, and comparing
models trained on different preprocessing or splits. Mitigations: no
hyperparameter was selected on the test set — hyperparameters and the Neutral
threshold-tuning α were selected on the validation split only, and every
number reported against the test split is a read-only evaluation of an
already-fixed configuration. The test split **is** reused across multiple
independent read-only analyses (ROC-AUC, confusion matrices, coverage-accuracy,
significance testing, Neutral analysis) at different sample sizes; this is a
deliberate reuse of a fixed held-out set for descriptive/inferential reporting,
not test-set tuning, and is disclosed here rather than glossed as "evaluated
once." Uncertainty is reported via bootstrap confidence intervals on macro-F1
and ECE, and via Holm-corrected McNemar tests for paired comparison; group-aware
splitting is used where VideoID is available.

## External Validity (Domain Shift)

A model strong on a large mixed-topic dataset may not generalise to a
specific target channel, to newer comments (temporal drift), or to
non-English/code-mixed comments. The metadata-backed domain-slice evaluation
quantifies the category and country performance spread, and the English-only
filtering is stated as an explicit external-validity limitation.

## Bias, Ethics, and Data Governance

Language filtering systematically excludes non-English speakers, and
aggressive regex cleaning can distort dialects, slang, and named entities.
YouTube comments can contain personal data or hate speech. The thesis
documents dataset licensing and source, what is stored and retained, and a
bias-analysis plan across language, topic category, and toxicity proxies.

## Reproducibility (Engineering Validity)

Split provenance is written to metadata, language detection is made
deterministic via a fixed seed, and the pipeline supports a command log plus
a timestamped reproducibility bundle (git commit/dirty state, runtime
metadata, pip-freeze and environment lock files, and SHA-256 artifact
checksums covering both configuration files and trained model binaries). This
is what makes the artifact-pinned claims auditable.

## Claims-vs-Code Discipline

Every claim in the thesis is backed by a runnable script, stored outputs, and
a documented configuration. Full-test runtime claims cite the live runtime
benchmark; the historical generic ensemble row is explicitly superseded by
the offline-vs-live reconciliation. The headline is phrased as either best
live macro-F1 (meta_learner) or best live calibrated ensemble
(ensemble_nsga2), and features that are easy to overstate — keyword-level
aspect extraction, the uncalibrated hybrid_dl runtime, and the gold set's
train-split sampling frame — are scoped accordingly.
