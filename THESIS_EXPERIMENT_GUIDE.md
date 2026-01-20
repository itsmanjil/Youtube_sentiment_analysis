# Thesis Experiment Guide
## YouTube Sentiment Analysis with Computational Intelligence Techniques

**Complete guide for running thesis-grade experiments**

---

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Data Preparation](#data-preparation)
4. [Experiment 1: Baseline Models](#experiment-1-baseline-models)
5. [Experiment 2: Deep Learning Model](#experiment-2-deep-learning-model)
6. [Experiment 3: Ensemble Methods](#experiment-3-ensemble-methods)
7. [Experiment 4: Meta-Learner](#experiment-4-meta-learner)
8. [Evaluation & Analysis](#evaluation--analysis)
9. [Results Reporting](#results-reporting)

---

## Overview

### Research Objectives

This guide supports thesis research on:
- **Sentiment Analysis** of YouTube comments
- **Comparative evaluation** of ML/DL models
- **Ensemble methods** (soft voting, stacking)
- **Optimization techniques** (PSO for weight tuning)
- **Statistical validation** (10-fold CV, bootstrap CI, McNemar's test)

### Available Models

| Model | Type | Description |
|-------|------|-------------|
| **LogReg** | Classical ML | TF-IDF + Logistic Regression baseline |
| **SVM** | Classical ML | TF-IDF + Linear SVM baseline |
| **TF-IDF** | Classical ML | Multinomial NB baseline |
| **Hybrid CNN-BiLSTM** | Deep Learning | Novel architecture with attention |
| **PSO Ensemble** | CI Optimization | Particle swarm optimized weights |
| **Meta-Learner** | Stacking | 2-level ensemble with meta-classifier |

---

## Infrastructure Setup

### 1. Test Training Infrastructure

Run the test suite to verify everything works:

```bash
cd backend
python test_training_infrastructure.py
```

**Expected output:**
```
✅ PASS  Imports
✅ PASS  Vocabulary
✅ PASS  Dataset & DataLoader
✅ PASS  Model Architecture
✅ PASS  Training Loop
✅ PASS  Checkpointing
✅ PASS  CSV Loading

7/7 tests passed
🎉 All tests passed! Training infrastructure is ready.
```

### 2. Install Dependencies

Ensure all required packages are installed:

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install scikit-learn==0.24.2
pip install pandas numpy
pip install nltk

# Optional (for advanced ensembles)
pip install xgboost lightgbm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### 3. GPU Setup (Optional)

For faster training, use GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Data Preparation

### Option A: Use Existing Labeled Dataset

If you have a pre-labeled dataset (e.g., from Kaggle, your own manual labels):

```bash
# Ensure CSV has 'text' and 'label' columns
# Labels: "Positive", "Neutral", "Negative"

# Split into train/val/test
python -c "
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('your_labeled_data.csv')
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

train.to_csv('data/train.csv', index=False)
val.to_csv('data/val.csv', index=False)
test.to_csv('data/test.csv', index=False)
"
```

### Option B: Fetch & Auto-Label YouTube Comments

Use LogReg to automatically label high-confidence comments:

#### 1. Create a video list file

Create `videos.txt` with YouTube URLs (one per line):

```text
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://www.youtube.com/watch?v=example123
https://www.youtube.com/watch?v=example456
# Add more videos (aim for 20-50 videos)
```

#### 2. Fetch and prepare data

```bash
python prepare_youtube_training_data.py \
    --video_list videos.txt \
    --max_comments 200 \
    --label_method auto \
    --confidence_threshold 0.7 \
    --output_dir ./data/youtube_training \
    --use_api
```

**What this does:**
1. Fetches comments from all videos in the list
2. Preprocesses (removes spam, non-English, etc.)
3. Auto-labels using LogReg with confidence filtering
4. Splits into train/val/test (70/15/15)
5. Exports to CSV files

**Expected output:**
```
✅ DATA PREPARATION COMPLETE

Train: 3500 samples
Val:   750 samples
Test:  750 samples

Files saved to: ./data/youtube_training/
```

---

## Experiment 1: Baseline Models

### Objective
Establish baseline performance with traditional models.

### 1.1 Evaluate LogReg, SVM, TF-IDF

```bash
python research/experiment_runner.py \
    --data data/test.csv \
    --models logreg,svm,tfidf \
    --test_size 1.0 \
    --output results/baseline_results.json
```

**Output:** `baseline_results.json` with accuracy, F1, precision, recall for each model.

### 1.2 10-Fold Cross-Validation (Thesis-grade evaluation)

For rigorous evaluation, run 10-fold CV:

```python
# Run 10-fold CV evaluation
python -c "
import sys
sys.path.insert(0, 'backend')

from research.evaluation_framework import evaluate_model_cv
import pandas as pd

df = pd.read_csv('data/train.csv')  # Use full dataset for CV
texts = df['text'].tolist()
labels = df['label'].tolist()

for model in ['logreg', 'svm', 'tfidf']:
    print(f'\n=== {model.upper()} ===')
    results = evaluate_model_cv(
        model_name=model,
        texts=texts,
        labels=labels,
        n_folds=10,
        random_state=42
    )
    print(f'Accuracy: {results[\"accuracy_mean\"]:.4f} ± {results[\"accuracy_std\"]:.4f}')
    print(f'F1 (macro): {results[\"f1_macro_mean\"]:.4f} ± {results[\"f1_macro_std\"]:.4f}')
"
```

---

## Experiment 2: Deep Learning Model

### Objective
Train and evaluate the novel Hybrid CNN-BiLSTM-Attention model.

### 2.1 Quick Test Run (2 epochs)

```bash
cd backend

python research/train_hybrid_dl.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --test_csv data/test.csv \
    --max_epochs 2 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir output/test_run \
    --experiment_name quick_test
```

### 2.2 Full Training with Config File

Edit `research/config/hybrid_dl_config.yaml` to set:
- `max_epochs: 50`
- `early_stopping.patience: 7`
- Embedding paths (if using GloVe)

```bash
python research/train_hybrid_dl.py \
    --config research/config/hybrid_dl_config.yaml \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --test_csv data/test.csv \
    --experiment_name thesis_hybrid_v1
```

**Monitor training:**

```bash
# In a separate terminal
tensorboard --logdir=model/hybrid_dl/training_logs
```

Visit `http://localhost:6006` to view training curves.

### 2.3 Evaluate Trained Model

After training completes:

```bash
python -c "
import sys
sys.path.insert(0, 'backend')

from app.deep_models import HybridDLSentimentEngine
import pandas as pd

# Load test data
df = pd.read_csv('data/test.csv')

# Load trained model
engine = HybridDLSentimentEngine(
    model_path='output/thesis_hybrid_v1/final_model.pt',
    vocab_path='output/thesis_hybrid_v1/vocab.pkl',
    device='auto'
)

# Evaluate
results = engine.batch_analyze(df['text'].tolist())
predictions = [r.label for r in results]

from sklearn.metrics import accuracy_score, f1_score, classification_report
print('Hybrid-DL Test Results:')
print(f'Accuracy: {accuracy_score(df[\"label\"], predictions):.4f}')
print(f'F1 (macro): {f1_score(df[\"label\"], predictions, average=\"macro\"):.4f}')
print(classification_report(df['label'], predictions))
"
```

---

## Experiment 3: Ensemble Methods

### Objective
Compare ensemble techniques: equal weights, PSO-optimized weights, soft voting.

### 3.1 Equal-Weight Ensemble

```bash
python research/experiment_runner.py \
    --data data/test.csv \
    --models ensemble \
    --ensemble_models logreg,svm,tfidf \
    --output results/ensemble_equal.json
```

### 3.2 PSO-Optimized Ensemble

Optimize ensemble weights using Particle Swarm Optimization:

```bash
python research/optimize_ensemble.py \
    --data data/train.csv \
    --models logreg,svm,tfidf \
    --particles 30 \
    --iterations 50 \
    --test_size 0.2 \
    --output results/pso_weights.json
```

**Output example:**
```json
{
  "weights": {
    "logreg": 0.25,
    "svm": 0.55,
    "tfidf": 0.20
  },
  "macro_f1": 0.7845
}
```

### 3.3 Evaluate PSO Ensemble

```bash
python research/experiment_runner.py \
    --data data/test.csv \
    --models ensemble \
    --ensemble_models logreg,svm,tfidf \
    --ensemble_weights '{"logreg": 0.25, "svm": 0.55, "tfidf": 0.20}' \
    --output results/ensemble_pso.json
```

---

## Experiment 4: Meta-Learner (Stacking)

### Objective
Train a 2-level stacked ensemble where a meta-learner combines base models.

### 4.1 Train Meta-Learner

```bash
python research/meta_learner.py \
    --data data/full_dataset.csv \
    --base_models logreg,svm,tfidf \
    --meta_learner logistic_regression \
    --n_folds 5 \
    --test_size 0.2 \
    --output models/meta_learner.pkl
```

**What happens:**
1. Splits data into train/test
2. Uses 5-fold CV to generate out-of-fold predictions from base models
3. Trains meta-learner (Logistic Regression) on OOF predictions
4. Evaluates on test set
5. Saves trained meta-learner

**Expected output:**
```
Training accuracy (OOF): 0.7892
Training F1 (OOF):      0.7856

META-LEARNER EVALUATION
Accuracy:  0.8012
F1 (macro): 0.7978
```

### 4.2 Try Different Meta-Learners

```bash
# XGBoost
python research/meta_learner.py \
    --data data/full_dataset.csv \
    --base_models logreg,svm,tfidf \
    --meta_learner xgboost \
    --n_folds 5 \
    --test_size 0.2 \
    --output models/meta_learner_xgb.pkl

# LightGBM
python research/meta_learner.py \
    --data data/full_dataset.csv \
    --base_models logreg,svm,tfidf \
    --meta_learner lightgbm \
    --n_folds 5 \
    --test_size 0.2 \
    --output models/meta_learner_lgbm.pkl
```

### 4.3 Compare All Meta-Learners

Create a comparison script:

```python
import pandas as pd

results = []

for meta_type in ['logistic_regression', 'xgboost', 'lightgbm']:
    # Load and evaluate
    # ... (evaluation code)
    results.append({
        'Meta-Learner': meta_type,
        'Accuracy': acc,
        'F1': f1
    })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

---

## Evaluation & Analysis

### Statistical Comparison (McNemar's Test)

Compare two models statistically:

```python
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

# Get predictions from two models
preds1 = model1.predict(test_texts)
preds2 = model2.predict(test_texts)

# Create contingency table
n01 = sum((preds1[i] != true_labels[i]) and (preds2[i] == true_labels[i])
          for i in range(len(true_labels)))
n10 = sum((preds1[i] == true_labels[i]) and (preds2[i] != true_labels[i])
          for i in range(len(true_labels)))

table = [[0, n01], [n10, 0]]
result = mcnemar(table, exact=False, correction=True)

print(f"McNemar statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")
print(f"Significant at α=0.05: {result.pvalue < 0.05}")
```

### Bootstrap Confidence Intervals

```python
from app.analysis_utils import bootstrap_confidence_intervals

# Get predictions
predictions = model.predict(test_texts)

# Compute 95% CI for each class proportion
ci = bootstrap_confidence_intervals(
    predictions,
    n_boot=1000,
    alpha=0.05,
    seed=42
)

print("95% Confidence Intervals:")
for label, bounds in ci.items():
    print(f"  {label}: [{bounds['lower']:.4f}, {bounds['upper']:.4f}]")
```

---

## Results Reporting

### Generate Thesis Tables

#### Table 1: Baseline Model Comparison

```python
import pandas as pd

results = {
    'Model': ['LogReg', 'SVM', 'TF-IDF'],
    'Accuracy': [0.6829, 0.7234, 0.6829],
    'Precision': [0.6877, 0.7301, 0.6877],
    'Recall': [0.6828, 0.7198, 0.6828],
    'F1 (macro)': [0.6822, 0.7189, 0.6822]
}

df = pd.DataFrame(results)
print(df.to_latex(index=False))  # For LaTeX thesis
```

#### Table 2: Ensemble Comparison

```python
results = {
    'Ensemble Type': ['Equal Weights', 'PSO-Optimized', 'Meta-Learner (LR)', 'Meta-Learner (XGB)'],
    'Base Models': ['LR+SVM+TF'] * 4,
    'Accuracy': [0.7120, 0.7456, 0.8012, 0.8134],
    'F1 (macro)': [0.7089, 0.7423, 0.7978, 0.8101],
    'Training Time (s)': ['-', 145, 234, 198]
}

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

### Create Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Model comparison bar chart
models = ['LogReg', 'SVM', 'TF-IDF', 'Hybrid-DL', 'PSO-Ens', 'Meta-LR']
f1_scores = [0.68, 0.72, 0.68, 0.76, 0.74, 0.80]

plt.figure(figsize=(10, 6))
plt.bar(models, f1_scores)
plt.ylabel('F1 Score (macro)')
plt.title('Model Comparison on YouTube Sentiment Analysis')
plt.ylim(0.6, 0.85)
plt.savefig('results/model_comparison.png', dpi=300)
```

---

## Complete Experiment Workflow

### Full Thesis Experiment Pipeline

```bash
#!/bin/bash
# Complete experiment pipeline for thesis

# 1. Test infrastructure
echo "=== Testing Infrastructure ==="
python test_training_infrastructure.py

# 2. Prepare data
echo "=== Preparing Data ==="
python prepare_youtube_training_data.py \
    --video_list videos.txt \
    --max_comments 200 \
    --output_dir data/youtube_training

# 3. Baseline models
echo "=== Baseline Evaluation ==="
python research/experiment_runner.py \
    --data data/youtube_training/test_*.csv \
    --models logreg,svm,tfidf \
    --output results/baseline.json

# 4. Train Hybrid-DL
echo "=== Training Hybrid-DL ==="
python research/train_hybrid_dl.py \
    --config research/config/hybrid_dl_config.yaml \
    --train_csv data/youtube_training/train_*.csv \
    --val_csv data/youtube_training/val_*.csv \
    --test_csv data/youtube_training/test_*.csv \
    --experiment_name thesis_final

# 5. PSO Ensemble
echo "=== PSO Optimization ==="
python research/optimize_ensemble.py \
    --data data/youtube_training/train_*.csv \
    --models logreg,svm,tfidf \
    --particles 30 \
    --iterations 50 \
    --output results/pso_weights.json

# 6. Meta-Learner
echo "=== Meta-Learner Training ==="
python research/meta_learner.py \
    --data data/youtube_training/full_data.csv \
    --base_models logreg,svm,tfidf \
    --meta_learner logistic_regression \
    --n_folds 5 \
    --output models/meta_learner.pkl

echo "=== Experiments Complete ==="
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:** Reduce batch size

```bash
python research/train_hybrid_dl.py --batch_size 16  # Instead of 32
```

#### 2. LogReg/SVM Model Not Found

**Solution:** Train the classic baselines first:

```bash
python train_logreg_youtube.py --data data/raw/youtube_comments_cleaned.csv
python train_svm_youtube.py --data data/raw/youtube_comments_cleaned.csv
```

#### 3. TF-IDF Model Not Found

**Solution:** Train TF-IDF baseline first or use existing model

```bash
# Check if model files exist
ls backend/model/model.sav
ls backend/model/tfidfVectorizer.pickle
```

#### 4. Training Too Slow on CPU

**Options:**
- Use smaller model architecture
- Reduce vocabulary size
- Use fewer training samples for initial experiments
- Rent GPU (Google Colab, AWS)

---

## Citation

If you use this framework in your thesis, consider citing:

```bibtex
@mastersthesis{yourname2024youtube,
  title={Computational Intelligence Approaches for YouTube Comment Sentiment Analysis},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Test infrastructure | `python test_training_infrastructure.py` |
| Prepare data | `python prepare_youtube_training_data.py --video_list videos.txt` |
| Train Hybrid-DL | `python research/train_hybrid_dl.py --config ...` |
| Optimize ensemble | `python research/optimize_ensemble.py --data ...` |
| Train meta-learner | `python research/meta_learner.py --data ...` |
| Run baseline eval | `python research/experiment_runner.py --models logreg,svm,tfidf` |

---

**Good luck with your thesis research! 🎓**
