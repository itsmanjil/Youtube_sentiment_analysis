# Thesis Risks / Gaps Checklist (YouTube Sentiment Analysis)

This project is already fairly feature-rich. For a Master's thesis, the main risk is not "missing models" but **threats to validity**: leakage, train/inference skew, non-reproducible preprocessing, and evaluation mistakes that inflate results.

Below is a prioritized checklist you can reuse in a "Threats to Validity" / "Limitations" chapter section.

## 1) Data Leakage (Highest Priority)

**Risk:** Exact-duplicate texts (or near-duplicates) appearing across `train/val/test` can inflate test metrics.

**What we did in this repo:**
- Leakage checking script: `backend/scripts/prepare/check_split_leakage.py`
- Split builder that **drops conflicting-label texts** + **dedupes by final model-input text** before splitting:
  `backend/scripts/prepare/prepare_hf_dataset.py`
- Group-aware split by `VideoID` when available (reduces within-video topical leakage).
- Split provenance saved to: `backend/data/split_metadata.json`

**Residual gap:** The current mitigation is **exact-text** dedupe. Near-duplicate leakage (copy/paste with minor edits, templated spam, paraphrases) can still exist. If you have time, add a near-duplicate audit using MinHash/SimHash or sentence embeddings; otherwise, report this as a limitation.

## 2) Train/Inference Preprocessing Skew

**Risk:** If training data is "raw" but the API cleans aggressively (or vice versa), you get inconsistent behavior and misleading evaluation.

**Repo reality:**
- Production API preprocessing is `backend/app/youtube_preprocessor.py` (spam/lang filters optional but enabled by default in `backend/app/views.py`).
- Dataset preparation can apply the same pipeline via `--youtube_preprocess`:
  `backend/scripts/prepare/prepare_hf_dataset.py`

**Residual gap:** If you enable the *classical* preprocessing (`backend/src/preprocessing/classical.py`) during training (`backend/train_*.py --preprocess`), you must also enable it during inference (engine `preprocess=True`). Otherwise you reintroduce skew.

## 3) Label Noise / Construct Validity

**Risk:** Public sentiment datasets often contain:
- sarcasm / irony / context dependence,
- topic-dependent sentiment words,
- mislabeled or ambiguous examples,
- inconsistent label definitions ("Neutral" vs "Mixed/Unclear").

**What we did:**
- Drop texts with conflicting labels in `prepare_hf_dataset.py` (good first-order cleanup).

**Gap to address for thesis-grade credibility:**
- Build a small, human-labeled **gold set** (e.g., 300-1000 comments), compute agreement (Cohen's kappa or Krippendorff's alpha), and report performance on it.
- Script exists to start annotation: `backend/scripts/prepare/create_gold_set.py`

## 4) Evaluation Methodology / Conclusion Validity

**Risk:** Common thesis-killers:
- tuning hyperparameters on the test set,
- reporting a single split without uncertainty,
- claiming significance without tests,
- comparing models trained on different preprocessing/splits.

**What exists in repo:**
- Metrics runner(s): `backend/research/experiment_runner.py`
- Statistical tests + bootstrap CI framework: `backend/research/evaluation_framework.py`,
  `backend/research/evaluation/statistical_tests.py`

**Gaps to close:**
- Only tune on `val`, evaluate once on `test`.
- Report uncertainty: bootstrap CI on macro-F1, plus McNemar for paired comparison.
- If you use group splitting (`VideoID`), use group-aware CV (GroupKFold) for CV experiments.

## 5) External Validity (Domain Shift)

**Risk:** A model that performs well on a large mixed-topic dataset may not generalize to:
- your target channel/topics,
- newer comments (temporal drift),
- non-English / code-mixed comments (especially if you filter to English).

**Recommended thesis add-ons:**
- Cross-domain test: evaluate on comments collected from a small set of videos you choose (and label a subset).
- Report performance degradation vs the benchmark dataset.

## 6) Bias / Ethics / Data Governance

**Risk areas:**
- Language filtering systematically excludes non-English speakers.
- Aggressive regex cleaning can disproportionately distort dialects, slang, named entities.
- YouTube comments can contain PII / hate speech; thesis should include handling & governance.

**Minimum thesis content:**
- Dataset licensing/source and what is stored.
- Whether you persist raw comments; if yes, retention policy and anonymization.
- A short bias analysis plan (even if small): language, topic category, toxicity proxies.

## 7) Reproducibility (Engineering Validity)

**What helps already:**
- Split provenance is written to `backend/data/split_metadata.json`.
- Language detection made deterministic via `DetectorFactory.seed` in `backend/app/youtube_preprocessor.py`.

**Gaps to close:**
- Pin package versions more tightly for the thesis environment (not only `scikit-learn==1.8.0`).
- Record exact commands used to generate each dataset/model/result JSON (append to an experiment log).

## 8) “Claims vs Code” Gap (Avoid Overclaiming)

**Risk:** `backend/README_THESIS.md` mentions BERT results and extensive evaluation. Make sure every claim in the thesis is backed by:
- a runnable script in the repo,
- stored outputs (metrics JSON / plots),
- and a documented config.

If a model (e.g., BERT fine-tuning) is not implemented end-to-end, either implement it or explicitly scope it out.

