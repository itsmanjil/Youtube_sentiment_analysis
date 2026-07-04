# Label Provenance & Quality Assessment

## 1. Dataset Source

| Field | Value |
|-------|-------|
| **Source** | HuggingFace Hub |
| **Dataset ID** | `AmaanP314/youtube-comment-sentiment` |
| **File** | `youtube-comments-sentiment.csv` |
| **Raw rows** | 1,032,225 |
| **Classes** | Positive / Neutral / Negative |
| **Label column** | `Sentiment` |

## 2. How the Source Labels Were Generated

The source labels in this dataset were produced by an **automated sentiment
classifier** applied to YouTube comment text. This means:

- Labels were **not assigned by human annotators**
- The labelling process is not fully documented in the dataset card
- Typical automated labellers (VADER, TextBlob, fine-tuned BERT) achieve
  60–80% agreement with human labels on social media text

**Implication for thesis:** When you report metrics like 69.46% Macro-F1, you
are measuring how well your model *replicates the automated labeller*, not
absolute human-level sentiment accuracy. This distinction must be stated in
your Methodology and Threats to Validity sections.

## 3. Gold Set Status — COMPLETED

A stratified gold set of 300 comments was independently annotated using a blind
command-line tool (`scripts/annotate.py`) that presents each comment **without** its
automated source label (the annotation template contains only `text,label`). Two
independent annotation passes were collected (`data/gold_set_annotator_1.csv`,
`data/gold_set_annotator_2.csv`) and reconciled into `data/gold_set_human_reconciled.csv`.

**Annotator disclosure:** one annotation pass was completed by the thesis
author and the second by an independent second annotator not otherwise
involved in model development. Both passes were conducted blind to the
automated source labels via the same template and tool. The author's pass is
not fully arms-length in the way a third-party-only annotation would be, and
this is disclosed as a limitation; the reported Krippendorff's alpha
nonetheless reflects genuine agreement between two distinct people, not
self-consistency.

Results (`results/gold_set/iaa_report.md`):

- Krippendorff's alpha = 0.9547; Cohen's/Fleiss' kappa = 0.9546; percent agreement 97.0%
- 9 of 300 items had no agreed majority and were marked **disputed** and excluded
  (291 reconciled gold labels)
- The reconciled human labels agree with the dataset's automated source labels
  only **73.5%** (`gold_set_evaluation.json: human_ref.human_vs_source.accuracy
  = 0.7354`), confirming the gold labels are genuinely human-derived rather
  than copies of the source scheme. (Do not confuse this with the ~69–70%
  figure below, which is model-vs-human agreement — a different comparison.)

The reconciled human labels are used as an independent reference that separates label
error from model error in the gold-set evaluation (`results/gold_set/gold_set_evaluation.md`).
Against these human labels the models score ~0.70 macro-F1 (a credible, non-circular figure),
versus an inflated 0.92–0.97 against the silver/auto labels — the contrast is itself reported.

**Sampling frame note.** The gold set was originally sampled from `train.csv`,
not the held-out test split. A post-hoc membership audit
(`research/ci/gold_set_train_membership.py`) found 95/300 items (31.7%) are
exact-text members of the training split, 26 in validation, 36 in test, 143
unmatched. Re-running the gold-set evaluation on the 205-item held-out-only
subset (`results/gold_set/gold_set_evaluation_holdout.md`) shows no material
change (e.g. `ensemble_pso` macro-F1 0.7042 → 0.7128), so training-set
memorisation is not inflating the headline gold-set numbers. This is
reported as a methodological check, not hidden.

## 4. Data Processing Steps Applied

After sourcing from HuggingFace, the following transformations were applied
before splitting (documented in `backend/data/split_metadata.json`):

| Step | Details |
|------|---------|
| Label normalisation | `str.title()` → Positive / Neutral / Negative |
| Whitespace collapse | Multi-space → single space, strip |
| Conflicting-label removal | 1,228 texts with disagreeing labels dropped |
| Exact-duplicate removal | 12,679 duplicate rows removed |
| YouTube preprocessing | Emoji convert, spam filter, language filter, min 3 words |
| Spam filtered | 34,266 rows |
| Non-English filtered | 138,590 rows |
| Too-short filtered | 25,221 rows |
| NaN text/label dropped | 9,391 rows (see note) |
| Split strategy | Group-aware by VideoID (prevents topical leakage) |

**Row-accounting note.** 1,032,225 − 1,228 (conflicting) − 12,679 (exact
duplicate) − 34,266 (spam) − 138,590 (non-English) − 25,221 (too short) =
820,241, not 810,850. The residual 9,391 rows were dropped by an
un-instrumented `dropna(subset=[text, label])` step that ran before the
YouTube-specific filters in `scripts/prepare/prepare_hf_dataset.py`; the
original pipeline run did not persist this count to
`split_metadata.json`. The script now records it as
`dedupe.nan_text_or_label_rows_dropped` for all future runs. For this
thesis's existing split, the correct statement is that 9,391 rows had a
missing text or label value and were dropped prior to YouTube-preprocessing,
not that they are unaccounted for.

## 5. Final Split Sizes

| Split | Rows | Negative | Neutral | Positive |
|-------|------|----------|---------|----------|
| Train | 516,886 | 184,266 (35.7%) | 160,744 (31.1%) | 171,876 (33.2%) |
| Val | 128,854 | 47,023 (36.5%) | 39,037 (30.3%) | 42,794 (33.2%) |
| Test | 165,110 | 59,614 (36.1%) | 50,540 (30.6%) | 54,956 (33.3%) |

**The class distribution is approximately balanced** (±5% across classes in
all splits), meaning reported Macro-F1 and Accuracy are directly comparable
and class imbalance is not a confound.

## 6. Recommended Thesis Language (Methodology Section)

> Data were sourced from the `AmaanP314/youtube-comment-sentiment` dataset on
> HuggingFace Hub, comprising 1,032,225 YouTube comments labelled with
> Positive, Neutral, and Negative sentiment via automated classification.
> After deduplication, conflicting-label removal, and YouTube-specific
> preprocessing (spam filtering, language detection, minimum length filtering),
> the final corpus contains 810,850 samples partitioned into train (516,886),
> validation (128,854), and test (165,110) splits using a group-aware strategy
> by VideoID to prevent topical data leakage. The resulting class distribution
> is approximately balanced (Negative 35.7%, Positive 33.2%, Neutral 31.1%).
> Label reliability was assessed by re-annotating a stratified gold set of 300
> comments in two independent, source-label-blind passes, yielding Krippendorff's
> α = 0.9547 (Cohen's κ = 0.9546; 97.0% agreement); the reconciled human labels
> serve as an independent reference distinct from the automated source labels.

## 7. References

- Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational
  linguistics. *Computational Linguistics*, 34(4), 555–596.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
  for categorical data. *Biometrics*, 33(1), 159–174.
