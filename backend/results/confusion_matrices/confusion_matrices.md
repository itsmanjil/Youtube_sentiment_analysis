# Confusion Matrices — All Models

- Dataset: `data\test.csv`
- Samples: `5000`
- Label order: Negative / Neutral / Positive

## Per-Class Precision / Recall / F1 Summary

| Model | Neg P | Neg R | Neg F1 | Neu P | Neu R | Neu F1 | Pos P | Pos R | Pos F1 | Macro F1 |
|-------|-------|-------|--------|-------|-------|--------|-------|-------|--------|----------|
| logreg | 0.697 | 0.719 | 0.708 | 0.619 | 0.604 | 0.611 | 0.759 | 0.748 | 0.753 | 0.6909 |
| svm | 0.691 | 0.707 | 0.699 | 0.603 | 0.596 | 0.600 | 0.744 | 0.733 | 0.739 | 0.6791 |
| tfidf | 0.625 | 0.754 | 0.684 | 0.613 | 0.513 | 0.558 | 0.750 | 0.694 | 0.721 | 0.6545 |
| ensemble_pso | 0.695 | 0.712 | 0.704 | 0.613 | 0.604 | 0.608 | 0.756 | 0.745 | 0.750 | 0.6873 |
| ensemble_nsga2 | 0.695 | 0.724 | 0.710 | 0.625 | 0.606 | 0.615 | 0.762 | 0.748 | 0.755 | 0.6933 |
| meta_learner | 0.713 | 0.701 | 0.707 | 0.613 | 0.633 | 0.623 | 0.763 | 0.754 | 0.758 | 0.6960 |
| fuzzy_ensemble | 0.626 | 0.753 | 0.684 | 0.611 | 0.513 | 0.558 | 0.751 | 0.694 | 0.722 | 0.6543 |

---

## logreg

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1312 | 351 | 161 |
| **Neutral** | 379 | 929 | 229 |
| **Positive** | 191 | 222 | 1226 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.719 | 0.192 | 0.088 |
| **Neutral** | 0.247 | 0.604 | 0.149 |
| **Positive** | 0.117 | 0.135 | 0.748 |


## svm

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1290 | 367 | 167 |
| **Neutral** | 375 | 916 | 246 |
| **Positive** | 201 | 236 | 1202 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.707 | 0.201 | 0.092 |
| **Neutral** | 0.244 | 0.596 | 0.160 |
| **Positive** | 0.123 | 0.144 | 0.733 |


## tfidf

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1376 | 299 | 149 |
| **Neutral** | 520 | 788 | 229 |
| **Positive** | 304 | 198 | 1137 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.754 | 0.164 | 0.082 |
| **Neutral** | 0.338 | 0.513 | 0.149 |
| **Positive** | 0.185 | 0.121 | 0.694 |


## ensemble_pso

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1299 | 360 | 165 |
| **Neutral** | 379 | 928 | 230 |
| **Positive** | 191 | 227 | 1221 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.712 | 0.197 | 0.090 |
| **Neutral** | 0.247 | 0.604 | 0.150 |
| **Positive** | 0.117 | 0.138 | 0.745 |


## ensemble_nsga2

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1321 | 343 | 160 |
| **Neutral** | 383 | 932 | 222 |
| **Positive** | 196 | 217 | 1226 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.724 | 0.188 | 0.088 |
| **Neutral** | 0.249 | 0.606 | 0.144 |
| **Positive** | 0.120 | 0.132 | 0.748 |


## meta_learner

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1278 | 381 | 165 |
| **Neutral** | 345 | 973 | 219 |
| **Positive** | 169 | 234 | 1236 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.701 | 0.209 | 0.090 |
| **Neutral** | 0.224 | 0.633 | 0.142 |
| **Positive** | 0.103 | 0.143 | 0.754 |


## fuzzy_ensemble

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1374 | 303 | 147 |
| **Neutral** | 519 | 788 | 230 |
| **Positive** | 303 | 198 | 1138 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.753 | 0.166 | 0.081 |
| **Neutral** | 0.338 | 0.513 | 0.150 |
| **Positive** | 0.185 | 0.121 | 0.694 |


## Thesis Interpretation

The confusion matrices reveal a consistent pattern across all models:
- Neutral class has the highest off-diagonal mass (most confused with Positive and Negative)
- Positive and Negative are generally well-separated from each other
- The Neutral row in the normalised matrix shows the lowest diagonal value for every model,
  confirming that neutral sentiment is the primary source of classification error.
