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

## 3. Current Gold Set Status — ACTION REQUIRED

The file `backend/data/gold_set_labeled_from_dataset.csv` (300 samples) was
generated from the dataset but was **not independently re-annotated**. The
`source_label` and `label` columns are identical across all 300 rows, yielding
a trivial κ = 1.0 that does not measure real label quality.

### What This Means

Without genuine independent annotation:
- You cannot report a meaningful Cohen's Kappa
- You cannot separate "model error" from "label error" on the gold set
- Your thesis faces a **construct validity gap** that a committee will identify

### What To Do

**Option A — Perform independent re-annotation (Recommended)**

Re-annotate the 300 gold set comments yourself, recording your labels in the
`label` column while keeping `source_label` as-is. Then re-run:

```bash
cd backend
python research/analysis/label_quality_report.py
```

A κ ≥ 0.60 (Substantial agreement) is the accepted threshold for thesis-grade
annotation quality (Artstein & Poesio, 2008).

Even with a single annotator, this is valid for a Master's thesis — report it
as "single-annotator agreement with automated source labels" rather than
"inter-annotator agreement."

**Option B — Use two annotators**

If you have a colleague available, have them annotate 50–100 comments
independently, then compute κ between both human annotations. This measures
true inter-annotator reliability and is the gold standard.

**Option C — Cite the dataset's existing validation**

If time does not allow re-annotation, clearly state in your thesis:

> "Labels were sourced from the AmaanP314/youtube-comment-sentiment dataset,
> which uses automated annotation. No independent human re-annotation was
> performed as part of this study. All reported performance metrics therefore
> measure agreement with the automated labelling scheme rather than absolute
> human-judged sentiment accuracy. This constitutes a construct validity
> limitation acknowledged in Section X.X."

Option C is the weakest choice — Option A is achievable in 2–3 hours.

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
| Split strategy | Group-aware by VideoID (prevents topical leakage) |

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
> Label reliability was assessed by [Option A: re-annotating a stratified
> gold set of 300 comments, yielding Cohen's κ = X.XX / Option C: the
> automated label provenance constitutes a construct validity limitation
> acknowledged in the Threats to Validity section].

## 7. References

- Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational
  linguistics. *Computational Linguistics*, 34(4), 555–596.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
  for categorical data. *Biometrics*, 33(1), 159–174.
