# Confusion Matrices

Rows = True Label, Columns = Predicted Label.
Values shown as **count (normalised row %)**.

## LOGREG

Total test samples: 9,998

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 2,590 (71.8%) | 734 (20.3%) | 286 (7.9%) |
| **Neutral** | 734 (24.0%) | 1,929 (63.0%) | 397 (13.0%) |
| **Positive** | 426 (12.8%) | 466 (14.0%) | 2,436 (73.2%) |

**Overall accuracy:** 0.6956

**Per-class recall (sensitivity):**

- Negative: 0.7175
- Neutral: 0.6304
- Positive: 0.7320

---

## SVM

Total test samples: 9,998

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 2,550 (70.6%) | 764 (21.2%) | 296 (8.2%) |
| **Neutral** | 730 (23.9%) | 1,895 (61.9%) | 435 (14.2%) |
| **Positive** | 432 (13.0%) | 477 (14.3%) | 2,419 (72.7%) |

**Overall accuracy:** 0.6865

**Per-class recall (sensitivity):**

- Negative: 0.7064
- Neutral: 0.6193
- Positive: 0.7269

---

## TFIDF

Total test samples: 9,998

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 2,730 (75.6%) | 656 (18.2%) | 224 (6.2%) |
| **Neutral** | 998 (32.6%) | 1,619 (52.9%) | 443 (14.5%) |
| **Positive** | 615 (18.5%) | 401 (12.0%) | 2,312 (69.5%) |

**Overall accuracy:** 0.6662

**Per-class recall (sensitivity):**

- Negative: 0.7562
- Neutral: 0.5291
- Positive: 0.6947

---

## ENSEMBLE

Total test samples: 9,998

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 2,634 (73.0%) | 716 (19.8%) | 260 (7.2%) |
| **Neutral** | 757 (24.7%) | 1,903 (62.2%) | 400 (13.1%) |
| **Positive** | 448 (13.5%) | 456 (13.7%) | 2,424 (72.8%) |

**Overall accuracy:** 0.6962

**Per-class recall (sensitivity):**

- Negative: 0.7296
- Neutral: 0.6219
- Positive: 0.7284

---

## META_LEARNER

Total test samples: 9,998

| True \ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| **Negative** | 2,498 (69.2%) | 816 (22.6%) | 296 (8.2%) |
| **Neutral** | 672 (22.0%) | 1,994 (65.2%) | 394 (12.9%) |
| **Positive** | 384 (11.5%) | 488 (14.7%) | 2,456 (73.8%) |

**Overall accuracy:** 0.6949

**Per-class recall (sensitivity):**

- Negative: 0.6920
- Neutral: 0.6516
- Positive: 0.7380

---
