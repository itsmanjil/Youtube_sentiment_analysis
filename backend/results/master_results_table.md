# Master Results Table

> Generated automatically from results/ JSON files.

## Part 1 — Classical Ensemble (Full Dataset: 165k test samples)

### 1a. Per-Model Metrics

| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | Temperature |
|-------|----------|-----------|------------------|-------------|
| logreg | 0.6943 | 0.0068 | 0.0074 | 1.031 |
| svm | 0.6817 | 0.0126 | 0.0163 | 1.033 |
| tfidf | 0.6579 | 0.0131 | 0.0174 | 1.131 |
| ensemble | 0.6938 | 0.0216 | 0.0117 | 0.935 |
| meta_learner | 0.6967 | 0.0203 | 0.0230 | 0.984 |

### 1b. Ensemble Optimisation (NSGA-II Multi-Objective)

| Method | Macro-F1 | ECE | Coverage@70% |
|--------|----------|-----|--------------|
| NSGA-II knee (logreg=0.916  svm=0.003  tfidf=0.081) | 0.6940 | 0.0039 | 0.4711 |
| Single-obj PSO (best val) | — | — | — |

### 1c. Neuro-Fuzzy Gating

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Uniform ensemble | 0.6938 | 0.6959 | 0.0260 | 0.4123 |
| **Neuro-fuzzy gate** | **0.6955** | 0.6972 | **0.0070** | 0.4076 |
| Δ (NF − uniform) | +0.0017 | — | -0.0190 | -0.0047 |

### 1d. Selective Prediction (Entropy-Gated)

Full ensemble (neuro-fuzzy mode)  F1=0.6949  AURC=0.1521

| τ | Coverage | Accuracy | Macro-F1 |
|---|----------|----------|----------|
| 0.04 | 0.018 | 0.9918 | 0.8618 |
| 0.16 | 0.078 | 0.9852 | 0.9351 |
| 0.28 | 0.140 | 0.9662 | 0.9326 |
| 0.40 | 0.210 | 0.9428 | 0.9151 |
| 0.52 | 0.296 | 0.9162 | 0.8941 |
| 0.64 | 0.404 | 0.8781 | 0.8597 |
| 0.76 | 0.574 | 0.8189 | 0.8083 |
| 0.88 | 0.786 | 0.7552 | 0.7507 |
| 1.00 | 1.000 | 0.6964 | 0.6949 |

### 1e. Coverage-Accuracy (AUCA)

| Rank | Model | AUCA | AUC-F1 | Acc@100% |
|------|-------|------|--------|----------|
| 1 | logreg | 0.8300 | 0.8124 | 0.6957 |
| 2 | svm | 0.8152 | 0.8029 | 0.6835 |
| 3 | tfidf | 0.7975 | 0.7672 | 0.6630 |
| 4 | ensemble | 0.8288 | 0.8135 | 0.6956 |
| 5 | meta_learner | 0.8309 | 0.7479 | 0.6973 |
| 6 | neuro_fuzzy | 0.8304 | 0.8138 | 0.6972 |

---

## Part 2 — Route A: DeBERTa-v3 + Classical (450-sample fine-tune, 180 test)

> ⚠️  Small test set (n=180). Results are directional; full-scale fine-tuning pending GPU access.

### 2a. Per-Model Metrics

| Model | Macro-F1 | ECE (raw) | ECE (calibrated) | Temperature |
|-------|----------|-----------|------------------|-------------|
| deberta_v3 | 0.6579 | 0.1084 | 0.1084 | 1.000 |
| logreg | 0.7544 | 0.0789 | 0.0621 | 0.861 |
| svm | 0.7863 | 0.1004 | 0.0649 | 0.735 |

### 2b. Neuro-Fuzzy Gating

| Method | Macro-F1 | Accuracy | ECE | Brier |
|--------|----------|----------|-----|-------|
| Uniform ensemble | 0.7770 | 0.7778 | 0.1385 | 0.3375 |
| **Neuro-fuzzy gate** | **0.7976** | 0.8000 | **0.0853** | 0.3048 |
| Δ | +0.0206 | — | -0.0532 | -0.0327 |

### 2c. Coverage-Accuracy (AUCA)

| Rank | Model | AUCA | AUC-F1 | Acc@100% |
|------|-------|------|--------|----------|
| 1 | svm | 0.9122 | 0.9095 | 0.7889 |
| 2 | neuro_fuzzy | 0.9070 | 0.9015 | 0.8000 |
| 3 | ensemble_uniform | 0.9048 | 0.9002 | 0.7778 |
| 4 | logreg | 0.9003 | 0.8951 | 0.7556 |
| 5 | deberta_v3 | 0.7125 | 0.5642 | 0.6556 |

---

## Part 3 — Cross-System Comparison

| System | Method | Macro-F1 | ECE | AUCA | Notes |
|--------|--------|----------|-----|------|-------|
| Classical (165k) | logreg | 0.6943 | 0.0068 | — | Full test set |
| Classical (165k) | svm | 0.6817 | 0.0126 | — | Full test set |
| Classical (165k) | tfidf | 0.6579 | 0.0131 | — | Full test set |
| Classical (165k) | ensemble | 0.6938 | 0.0216 | — | Full test set |
| Classical (165k) | meta_learner | 0.6967 | 0.0203 | — | Full test set |
| Classical (165k) | Neuro-fuzzy gate | 0.6955 | 0.0070 | — | Adaptive routing |
| Classical (165k) | NSGA-II ensemble | 0.6940 | 0.0039 | — | Pareto knee-point |
| Route A (450 train) | NF gate (DeBERTa+LogReg+SVM) | 0.7976 | 0.0853 | — | n=180 test, directional |
| Route A (450 train) | Best AUCA (svm) | 0.7889 | — | 0.9122 | n=180 test |