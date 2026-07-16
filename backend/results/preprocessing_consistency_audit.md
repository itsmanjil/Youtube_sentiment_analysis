# Preprocessing Consistency Audit

## Purpose

This document verifies that preprocessing applied during model training is
consistent with preprocessing applied during API inference — a critical
requirement for valid test metrics and production reliability.

## Methodology

Traced the full text transformation chain for both paths by inspecting:
- Model metadata files (`models/*/metadata.json`)
- Data preparation metadata (`data/split_metadata.json`)
- Sentiment engine source code (`src/sentiment/engines/*.py`)
- API view layer (`app/views.py`)
- YouTube preprocessor (`app/youtube_preprocessor.py`)

## Training Path

### Stage 1: Data Preparation (before model training)

YouTube preprocessing was applied to the raw HuggingFace dataset
(`split_metadata.json`, `youtube_preprocess.enabled: true`):

| Step | Applied |
|------|---------|
| Emoji conversion (demojize) | ✓ |
| Timestamp removal | ✓ |
| Channel mention removal | ✓ |
| URL/bracket cleanup | ✓ |
| Lowercase conversion | ✓ |
| Elongated word normalization | ✓ |
| Non-alphanumeric removal | ✓ |
| Single-letter word removal | ✓ |
| Spam filter | ✓ (34,266 removed) |
| Language filter (English only) | ✓ (138,590 removed) |
| Short comment filter (< 3 words) | ✓ (25,221 removed) |

### Stage 2: Model Training

Classical preprocessing (negation expansion, stopword removal, negation
tagging) was **disabled** for all three base models:

| Model | `preprocessing.enabled` | Evidence |
|-------|------------------------|----------|
| LogReg | `false` | `models/logreg_youtube_filtered/logreg_metadata.json` |
| SVM | `false` | `models/svm_youtube_filtered/svm_metadata.json` |
| TF-IDF NB | `false` | `models/tfidf_youtube_filtered/tfidf_metadata.json` |

TF-IDF vectorizer settings (identical across all models):
- `max_features`: 75,000
- `min_df`: 2, `max_df`: 0.95
- `ngram_range`: (1, 2)
- `lowercase`: true
- `strip_accents`: false

## Inference Path

### Stage 1: API YouTube Preprocessing

When a user submits a video URL to `/api/youtube/analyze/`, the API layer
(`app/views.py`, lines 283–293) applies `YouTubePreprocessor.batch_preprocess()`
with `profile='classical'`, performing the **identical** transformation steps
as the data preparation stage.

### Stage 2: Sentiment Engine

All engines are instantiated via the factory (`src/sentiment/factory.py`)
with `preprocess=False` (default), matching the training configuration.
Text passes directly to the stored TF-IDF vectorizer.

## Consistency Verdict

| Preprocessing Stage | Training | Inference | Match |
|---------------------|----------|-----------|-------|
| YouTube text cleaning | ✓ (data prep) | ✓ (API layer) | ✓ |
| Classical preprocessing | ✗ disabled | ✗ disabled | ✓ |
| TF-IDF vectorizer params | 75k/bigram/lc | Same pickle loaded | ✓ |
| Spam/language/length filters | ✓ (data prep) | ✓ (configurable) | ✓* |

*\*Note: At inference, spam/language/length filtering is configurable via
API parameters (`filter_spam`, `filter_language`). If a user disables these
filters, comments that would have been removed during training may reach the
model. This is a known limitation but not a systematic bias — it simply means
some low-quality inputs may receive less reliable predictions.*

## Conclusion

**No train/inference preprocessing skew detected.** Both paths apply YouTube
text cleaning followed by direct TF-IDF vectorization with classical
preprocessing disabled. The consistency ensures that reported test metrics
are representative of production performance.

---
*Audit generated: 2026-04-09*
