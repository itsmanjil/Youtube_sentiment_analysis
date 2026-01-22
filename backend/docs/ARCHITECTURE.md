# System Architecture

## Overview

This document describes the architecture of the YouTube Sentiment Analysis system,
designed for Master's thesis research in Computational Intelligence and NLP.

## Directory Structure

```
backend/
├── src/                          # Core application package
│   ├── sentiment/                # Sentiment analysis engines
│   │   ├── base.py              # SentimentResult, utilities
│   │   ├── factory.py           # Engine factory functions
│   │   └── engines/             # Individual engine implementations
│   │       ├── tfidf_engine.py  # TF-IDF + Naive Bayes
│   │       ├── logreg_engine.py # TF-IDF + Logistic Regression
│   │       ├── svm_engine.py    # TF-IDF + Linear SVM
│   │       ├── ensemble_engine.py    # Weighted voting
│   │       ├── meta_learner_engine.py # Stacked ensemble
│   │       ├── hybrid_dl_engine.py   # CNN-BiLSTM-Attention
│   │       └── transformer_engine.py # BERT-based
│   ├── preprocessing/           # Text preprocessing
│   ├── services/               # Business logic layer
│   └── utils/                  # Shared utilities
│       ├── analysis_utils.py   # Metrics, confidence intervals
│       └── config.py           # Centralized configuration
│
├── research/                    # Thesis research components
│   ├── architectures/          # Neural network architectures
│   │   ├── hybrid_cnn_bilstm.py
│   │   ├── attention.py
│   │   └── transformers/
│   │       └── bert_classifier.py
│   ├── evaluation/             # Evaluation framework
│   │   ├── statistical_tests.py # McNemar, Wilcoxon, Friedman
│   │   └── ablation.py         # Ablation study framework
│   ├── explainability/         # XAI module
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── attention_explainer.py
│   ├── training/               # Training infrastructure
│   └── absa/                   # Aspect-Based Sentiment Analysis
│
├── app/                        # Django application
├── scripts/                    # Training/evaluation scripts
├── tests/                      # Test suite
├── configs/                    # Configuration files
├── models/                     # Trained model artifacts
└── docs/                       # Documentation
```

## Model Architecture

### 1. Classical ML Models

```
Input Text
    ↓
TF-IDF Vectorization (unigrams + bigrams, max 5000 features)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Naive Bayes     │ Logistic Reg    │ Linear SVM      │
│ (baseline)      │ (calibrated)    │ (best accuracy) │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                    ↓                  ↓
Probability Distribution: P(Neg), P(Neu), P(Pos)
```

### 2. Hybrid CNN-BiLSTM-Attention

```
Input Text (max 200 tokens)
    ↓
Word Embeddings (GloVe 300d)
    ↓
┌─────────────────────────────────────────────────────┐
│                 PARALLEL BRANCHES                    │
├────────────────────────┬────────────────────────────┤
│      CNN Branch        │      BiLSTM Branch         │
│  ┌─────────────────┐   │  ┌─────────────────────┐   │
│  │ Conv1d (k=3,4,5)│   │  │ BiLSTM (128 hidden) │   │
│  │ 128 filters each│   │  │ 2 layers            │   │
│  └────────┬────────┘   │  └──────────┬──────────┘   │
│           ↓            │             ↓              │
│  Global MaxPool        │  Multi-Head Attention      │
│  Output: 384-dim       │  (4 heads)                 │
│                        │  Output: 256-dim           │
└────────────────────────┴────────────────────────────┘
                    ↓
            Feature Fusion
            (Concatenate: 640-dim)
                    ↓
            Dense Classifier
            (640 → 256 → 128 → 3)
                    ↓
            Softmax Output
```

### 3. BERT Transformer

```
Input Text
    ↓
BERT Tokenizer (WordPiece)
    ↓
[CLS] token1 token2 ... [SEP]
    ↓
BERT Encoder (12 layers, 768-dim)
    ↓
[CLS] Embedding (768-dim)
    ↓
Dropout (0.1)
    ↓
Linear Classifier (768 → 3)
    ↓
Softmax Output
```

### 4. Ensemble Methods

#### Weighted Soft Voting
```
┌─────────────────────────────────────────────────────┐
│              Base Model Predictions                  │
├──────────────┬──────────────┬───────────────────────┤
│   LogReg     │     SVM      │       TF-IDF          │
│   (w=0.4)    │   (w=0.4)    │       (w=0.2)         │
└──────────────┴──────────────┴───────────────────────┘
                        ↓
        P_ensemble(c) = Σ w_i × P_i(c)
                        ↓
                 Final Prediction
```

#### Meta-Learner (Stacking)
```
Level 0: Base Models
┌─────────────────────────────────────────────────────┐
│ LogReg → P(neg), P(neu), P(pos)                     │
│ SVM    → P(neg), P(neu), P(pos)                     │
│ TF-IDF → P(neg), P(neu), P(pos)                     │
└─────────────────────────────────────────────────────┘
                        ↓
            Feature Vector (9-dim)
                        ↓
Level 1: Meta-Classifier (Logistic Regression)
                        ↓
            Final Prediction
```

## Evaluation Framework

### Metrics Computed

1. **Classification Metrics**
   - Accuracy
   - Precision (macro, micro, per-class)
   - Recall (macro, micro, per-class)
   - F1-Score (macro, micro, per-class)
   - Cohen's Kappa

2. **Statistical Tests**
   - McNemar's Test: Compare two classifiers
   - Wilcoxon Signed-Rank: Compare fold-wise scores
   - Friedman Test: Compare 3+ classifiers
   - Nemenyi Post-hoc: Pairwise after Friedman

3. **Confidence Estimation**
   - Bootstrap Confidence Intervals (95%)
   - Entropy-based Confidence Scoring

### Cross-Validation Protocol

```
Dataset (N samples)
        ↓
Stratified 10-Fold CV
        ↓
┌─────────────────────────────────────────────────────┐
│ For each fold k:                                    │
│   1. Train on folds ≠ k                            │
│   2. Evaluate on fold k                            │
│   3. Store predictions and metrics                  │
└─────────────────────────────────────────────────────┘
        ↓
Aggregate Results:
- Mean ± Std for each metric
- Statistical significance tests
- Confusion matrices
```

## Explainability (XAI)

### SHAP Explanations
```
Input: "This video is amazing!"
        ↓
SHAP Explainer
        ↓
Token-level SHAP values:
  "This"    → +0.02 (neutral)
  "video"   → +0.01 (neutral)
  "is"      → +0.00 (neutral)
  "amazing" → +0.35 (strongly positive)
        ↓
Visualization: Bar plot / Force plot
```

### LIME Explanations
```
Input: "Terrible video, waste of time"
        ↓
Generate perturbations (5000 samples)
        ↓
Get model predictions for perturbations
        ↓
Fit local linear model
        ↓
Feature weights:
  "Terrible" → -0.42 (strongly negative)
  "waste"    → -0.28 (negative)
  "time"     → -0.15 (slightly negative)
```

### Attention Visualization
```
Hybrid Model / BERT
        ↓
Extract attention weights
        ↓
Token importance heatmap
        ↓
Highlighted text visualization
```

## Data Flow

### Analysis Pipeline

```
1. Input: YouTube Video URL
        ↓
2. YouTube API / Scraper
   - Fetch video metadata
   - Fetch comments (max 200)
        ↓
3. Preprocessing Pipeline
   - Spam detection
   - Language filtering (English)
   - Emoji handling
   - Text normalization
   - Short comment filtering
        ↓
4. Sentiment Analysis
   - Select engine (logreg, svm, bert, etc.)
   - Batch prediction
   - Confidence scoring
        ↓
5. Analytics
   - Sentiment distribution
   - Timeline analysis
   - Aspect extraction
   - Bootstrap confidence intervals
        ↓
6. Response
   - JSON with all results
   - Stored in database
```

## Configuration

### Model Paths (src/utils/config.py)

```python
Config.MODELS_DIR / "logreg" / "model.sav"
Config.MODELS_DIR / "svm" / "model.sav"
Config.MODELS_DIR / "hybrid_dl" / "hybrid_v1.pt"
Config.MODELS_DIR / "transformers" / "bert"
```

### Training Configuration (YAML)

```yaml
model:
  type: hybrid_dl
  embed_dim: 300
  hidden_dim: 128
  num_heads: 4
  dropout: 0.5

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 7

evaluation:
  cv_folds: 10
  metrics: [accuracy, f1_macro, f1_micro]
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/` | POST | Analyze YouTube video |
| `/api/analyses/` | GET | List user analyses |
| `/api/analyses/{id}/` | GET | Get specific analysis |

### Request Format

```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "max_comments": 200,
  "model_type": "logreg",
  "include_aspects": true
}
```

### Response Format

```json
{
  "video": {
    "title": "...",
    "channel": "...",
    "views": 1000000
  },
  "sentiment_data": {
    "Positive": 120,
    "Neutral": 50,
    "Negative": 30
  },
  "confidence_intervals": {
    "Positive": {"lower": 0.55, "upper": 0.65}
  },
  "aspects": [
    {"aspect": "content", "sentiment": {"Positive": 0.7, ...}}
  ]
}
```

## Performance Benchmarks

| Model | Accuracy | F1-Macro | Inference (ms/sample) |
|-------|----------|----------|----------------------|
| TF-IDF | 67.71% | 67.70% | ~1ms |
| LogReg | 74.27% | 74.34% | ~1ms |
| SVM | 75.08% | 75.14% | ~1ms |
| Hybrid-DL | ~78% | ~77% | ~10ms (CPU) |
| BERT | ~85-90% | ~84-89% | ~50ms (GPU) |

## Dependencies

### Core Dependencies
- Django 4.0+
- PyTorch 2.0+
- scikit-learn 1.0+
- transformers 4.30+

### XAI Dependencies
- shap 0.42+
- lime 0.2+

### Statistical Analysis
- scipy 1.10+
- statsmodels 0.14+

## References

1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions
3. Demsar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets
