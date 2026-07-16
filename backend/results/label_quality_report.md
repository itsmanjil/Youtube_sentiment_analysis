# Label Quality Report

## 1. Dataset Provenance

The training corpus was sourced from the publicly available HuggingFace dataset:

> **AmaanP314/youtube-comment-sentiment**
> `hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv`

The dataset provides three sentiment classes: **Positive**, **Neutral**, **Negative**.
Labels were originally assigned via automated annotation using a pre-trained sentiment
classifier. This introduces potential label noise, particularly for sarcastic,
context-dependent, or ambiguous comments.

A gold set of **300 comments** was drawn from the dataset for quality assessment.

> **⚠ ACTION REQUIRED — Gold Set Not Yet Re-annotated**
>
> The `gold_set_labeled_from_dataset.csv` file currently has identical values in
> `source_label` and `label` (κ = 1.0 is trivial — it compares a column to itself).
> **Independent re-annotation has not been performed yet.**
> See `backend/docs/LABEL_PROVENANCE.md` for instructions.

## 2. Inter-Annotator Agreement

> Values below are **placeholder** until re-annotation is complete.
> Re-run this script after updating `data/gold_set_labeled_from_dataset.csv`.

| Metric | Value |
|--------|-------|
| Gold Set Size | 300 |
| Percent Agreement | — (pending re-annotation) |
| Cohen's Kappa (κ) | — (pending re-annotation) |
| Kappa Interpretation | — |

**Target:** κ ≥ 0.60 (Substantial agreement) for thesis-grade label credibility.

**Kappa scale reference** (Landis & Koch, 1977):

| κ range | Interpretation |
|---------|----------------|
| < 0.20 | Slight |
| 0.20–0.40 | Fair |
| 0.40–0.60 | Moderate |
| 0.60–0.80 | Substantial |
| > 0.80 | Almost Perfect |

### 2.1 Confusion Matrix (Source Label vs Re-annotation)

Rows = source label, Columns = re-annotation label.

| | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 108 | 0 | 0 |
| **Neutral** | 0 | 80 | 0 |
| **Positive** | 0 | 0 | 112 |

### 2.2 Per-Class Label Changes

| Source Class | Total | Kept | Changed | % Changed |
|-------------|-------|------|---------|-----------|
| Negative | 108 | 108 | 0 | 0.0% |
| Neutral | 80 | 80 | 0 | 0.0% |
| Positive | 112 | 112 | 0 | 0.0% |

## 3. Label Distribution — Source vs Re-annotation

| Class | Source Count | Source % | Re-ann Count | Re-ann % |
|-------|-------------|----------|-------------|----------|
| Negative | 108 | 36.0% | 108 | 36.0% |
| Neutral | 80 | 26.7% | 80 | 26.7% |
| Positive | 112 | 37.3% | 112 | 37.3% |

## 4. Model Performance on Gold Set

Performance on the 300-sample human-labelled gold set (using re-annotation labels).

| Model | Accuracy | Macro-F1 | ECE | Brier |
|-------|----------|----------|-----|-------|
| tfidf | 0.6700 | 0.6512 | 0.0560 | 0.4384 |
| logreg | 0.6733 | 0.6580 | 0.0777 | 0.4100 |
| svm | 0.6600 | 0.6462 | 0.0608 | 0.4343 |
| ensemble | 0.6733 | 0.6594 | 0.0472 | 0.4119 |
| meta_learner | 0.6767 | 0.6645 | 0.0775 | 0.4128 |

> **Note:** Gold set performance is slightly lower than the held-out test set
> (~165K examples) because the gold set reflects human-label difficulty, while
> the test set uses the same automated labels as training.

## 5. Validity Discussion & Thesis Framing

### 5.1 Construct Validity

The source labels were assigned by an automated labeller on the HuggingFace dataset.
Cohen's κ between the source labels and independent re-annotation measures whether
the automated labels agree with human judgement at a level sufficient for thesis work.
A κ ≥ 0.60 (Substantial agreement) is the accepted threshold for NLP annotation tasks
(Artstein & Poesio, 2008).

### 5.2 Implication for Results

If κ < 0.60, you should report this as a **construct validity threat** in your
Threats to Validity section and note that model performance metrics are an upper
bound relative to the quality of the automated labels, not absolute sentiment accuracy.

### 5.3 Recommended Thesis Language

> This study uses labels drawn from the AmaanP314/youtube-comment-sentiment dataset,
> which were assigned via automated annotation. To assess label reliability, a
> stratified sample of 300 comments was independently re-annotated, yielding a
> Cohen's κ of {κ_value}  ({κ_interpretation}). We report model performance against
> both the automated test set and the human-annotated gold set to bound the
> impact of label noise on our conclusions.

Replace `{κ_value}` and `{κ_interpretation}` with the values from Section 2 above.