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
| ensemble_pso | 0.694 | 0.726 | 0.710 | 0.626 | 0.607 | 0.616 | 0.765 | 0.747 | 0.756 | 0.6940 |
| ensemble_nsga2 | 0.695 | 0.725 | 0.710 | 0.625 | 0.606 | 0.616 | 0.763 | 0.749 | 0.755 | 0.6937 |
| meta_learner | 0.713 | 0.701 | 0.707 | 0.614 | 0.632 | 0.623 | 0.763 | 0.755 | 0.759 | 0.6964 |
| fuzzy_ensemble | 0.690 | 0.733 | 0.710 | 0.627 | 0.600 | 0.613 | 0.766 | 0.744 | 0.755 | 0.6927 |

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
| **Negative** | 1324 | 342 | 158 |
| **Neutral** | 385 | 933 | 219 |
| **Positive** | 198 | 216 | 1225 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.726 | 0.188 | 0.087 |
| **Neutral** | 0.250 | 0.607 | 0.142 |
| **Positive** | 0.121 | 0.132 | 0.747 |


## ensemble_nsga2

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1322 | 342 | 160 |
| **Neutral** | 383 | 932 | 222 |
| **Positive** | 196 | 216 | 1227 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.725 | 0.188 | 0.088 |
| **Neutral** | 0.249 | 0.606 | 0.144 |
| **Positive** | 0.120 | 0.132 | 0.749 |


## meta_learner

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1279 | 380 | 165 |
| **Neutral** | 346 | 972 | 219 |
| **Positive** | 170 | 231 | 1238 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.701 | 0.208 | 0.090 |
| **Neutral** | 0.225 | 0.632 | 0.142 |
| **Positive** | 0.104 | 0.141 | 0.755 |


## fuzzy_ensemble

### Confusion Matrix (counts)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 1336 | 334 | 154 |
| **Neutral** | 396 | 922 | 219 |
| **Positive** | 205 | 215 | 1219 |

### Confusion Matrix (Row-Normalised Recall)

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 0.732 | 0.183 | 0.084 |
| **Neutral** | 0.258 | 0.600 | 0.142 |
| **Positive** | 0.125 | 0.131 | 0.744 |


## Thesis Interpretation

The confusion matrices reveal a consistent pattern across all models:
- Neutral class has the highest off-diagonal mass (most confused with Positive and Negative)
- Positive and Negative are generally well-separated from each other
- The Neutral row in the normalised matrix shows the lowest diagonal value for every model,
  confirming that neutral sentiment is the primary source of classification error.
